"""
Locust Test Generator

Generates Locust performance test files from parsed OpenAPI endpoints.
"""

import json
import re
import secrets
from typing import List, Tuple, Dict, Any, Optional
import black
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import logging
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime


from devdox_ai_locust.utils.open_ai_parser import Endpoint, Parameter

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types for testing"""

    MONGO = "mongo"
    POSTGRES = "postgres"


@dataclass
class MongoDBConfig:
    """MongoDB-specific configuration"""

    use_realistic_data: str = "true"
    enable_mongodb: str = "false"
    use_mongodb_for_test_data: str = "false"
    mongodb_uri: str = "mongodb://localhost:27017/"
    mongodb_database: str = "locust_test_data"
    MONGODB_MAX_POOL_SIZE: int = 100
    MONGODB_MIN_POOL_SIZE: int = 10

    # MongoDB Timeout Settings
    MONGODB_CONNECT_TIMEOUT_MS: int = 5000
    MONGODB_SERVER_SELECTION_TIMEOUT_MS: int = 5000
    MONGODB_SOCKET_TIMEOUT_MS: int = 10000
    MONGODB_MAX_IDLE_TIME_MS: int = 60000
    MONGODB_WAIT_QUEUE_TIMEOUT_MS: int = 10000

    # MongoDB Collection Names to be added


@dataclass
class PostgreSQLConfig:
    """PostgreSQL-specific configuration"""

    host: str = "localhost"
    port: str = "5432"
    database: str = "test_db"
    user: str = "test_user"
    password: str = "test_password"
    pool_size: str = "10"
    max_overflow: str = "20"


@dataclass
class TestDataConfig:
    """Configuration for test data generation"""

    string_length: int = 10
    integer_min: int = 1
    integer_max: int = 1000
    array_size: int = 3
    use_realistic_data: bool = True


class LocustTestGenerator:
    """Generates Locust performance test files from OpenAPI endpoints"""

    def __init__(
        self,
        test_config: Optional[TestDataConfig] = None,
        jinja_env: Optional[Environment] = None,
    ):
        self.test_config = test_config or TestDataConfig()
        self.generated_files: Dict[str, str] = {}
        self.auth_token = None
        self.user_data: Dict[str, Any] = {}
        self.request_count = 0

        self.template_dir = self._find_project_root() / "templates"

        self.template_dir.mkdir(parents=True, exist_ok=True)

        self.jinja_env = jinja_env or self._setup_templates()

    def _find_project_root(self) -> Path:
        """Find the project root by looking for setup.py, pyproject.toml, or .git"""
        current_path = Path(__file__).parent

        return current_path

    def _setup_templates(self) -> Environment:
        """Initialize Jinja2 environment and create default templates"""
        return Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=False,
        )

    def fix_indent(self, base_files: Dict[str, str]) -> Dict[str, str]:
        """Fix indentation for generated files"""
        try:
            mode = black.Mode()
            updated_data = {}

            for key, value in base_files.items():
                try:
                    formatted_code = black.format_str(value, mode=mode)

                    updated_data[key] = formatted_code
                except black.InvalidInput:
                    # Not valid Python code, keep original
                    logger.debug(f"Skipping formatting for {key}: not valid Python")
                    updated_data[key] = value
                except Exception as format_error:
                    # Other Black formatting errors, keep original and log
                    logger.warning(f"Failed to format {key}: {format_error}")
                    updated_data[key] = value

            return updated_data

        except Exception as e:
            logger.error(f"exception occurred: {e}")
            return base_files

    def generate_from_endpoints(
        self,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        include_auth: bool = True,
        target_host: Optional[str] = None,
        db_type: str = "",
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, List[Endpoint]]]:
        """
        Generate complete Locust test suite from parsed endpoints

        Args:
            endpoints: List of parsed Endpoint objects
            api_info: API information dictionary
            output_dir: Output directory for generated files

        Returns:
            Dictionary of filename -> file content
        """
        try:
            grouped_enpoint = self._group_endpoints_by_tag(endpoints, include_auth)

            workflows_files = self.generate_workflows(grouped_enpoint, api_info)

            self.generated_files = {
                "locustfile.py": self._generate_main_locustfile(
                    endpoints, api_info, list(grouped_enpoint.keys())
                ),
                "test_data.py": self._generate_test_data_file(db_type),
                "config.py": self._generate_config_file(api_info),
                "utils.py": self._generate_utils_file(),
                "custom_flows.py": self._generate_custom_flows_file(),
                "requirements.txt": self._generate_requirements_file(),
                "README.md": self._generate_readme_file(api_info, db_type),
                ".env.example": self._generate_env_example(
                    api_info, target_host, db_type
                ),
            }
            if db_type != "":
                self.generated_files["db_config.py"] = self._generate_db_file(
                    db_type, "db_config.py.j2"
                )
                self.generated_files["data_provider.py"] = self._generate_db_file(
                    db_type, "data_provider.py.j2"
                )

            return self.generated_files, workflows_files, grouped_enpoint
        except Exception as e:
            logger.error(f"Failed to generate test suite: {e}")
            empty_files: Dict[str, str] = {}
            empty_workflows: List[Dict[str, Any]] = []
            empty_grouped: Dict[str, List[Endpoint]] = {}
            return empty_files, empty_workflows, empty_grouped

    def generate_workflows(
        self, endpoints: Dict[str, List[Any]], api_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate the workflows Locust test files with proper structure and no duplicates
        """
        try:
            workflows: List = []

            for group, group_endpoints in endpoints.items():
                task_methods: List[str] = []

                for endpoint in group_endpoints:
                    try:
                        task_method = self._generate_task_method(endpoint)
                        if task_method:
                            task_methods.append(task_method)
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to generate task method for {getattr(endpoint, 'path', '?')}: {e}"
                        )
                        continue
                if not task_methods:
                    logger.warning(f"No task methods generated from group {group}")
                    task_methods.append(self._generate_default_task_method())
                indented_task_methods = self._indent_methods(
                    task_methods, indent_level=1
                )
                file_content = self._build_endpoint_template(
                    api_info, indented_task_methods, group
                )
                file_name = f"{group}_workflow.py".replace("-", "_")

                workflows.append({file_name: file_content})

            workflows.append(
                {"base_workflow.py": self.generate_base_common_file(api_info)}
            )

            return workflows

        except Exception as e:
            logger.error(f"❌ Failed to generate test suite: {e}")

            return []

    def _build_endpoint_template(
        self, api_info: Dict[str, Any], task_methods_content: str, group: str
    ) -> str:
        template = self.jinja_env.get_template("endpoint_template.py.j2")
        return template.render(
            api_info=api_info, group=group, task_methods_content=task_methods_content
        )

    def _generate_main_locustfile(
        self, endpoints: List[Any], api_info: Dict[str, Any], groups: List[str]
    ) -> str:
        """
        Generate the main Locust test file with proper structure and no duplicates

        Args:
            endpoints: List of parsed Endpoint objects
            api_info: API information dictionary

        Returns:
            Complete locustfile.py content as string
        """
        try:
            # Generate task methods for each endpoint
            task_methods = []
            for endpoint in endpoints:
                try:
                    task_method = self._generate_task_method(endpoint)
                    if task_method:
                        task_methods.append(task_method)
                except Exception as e:
                    logger.warning(
                        f"Failed to generate task method for  {endpoint.path}: {e}"
                    )
                    continue

            if not task_methods:
                logger.warning("No task methods generated from endpoints")
                # Generate a default task method
                task_methods.append(self._generate_default_task_method())

            # Properly indent task methods for class inclusion
            indented_task_methods = self._indent_methods(task_methods, indent_level=1)
            indented_task_methods = ""
            # Generate the complete file content
            return self._build_locustfile_template(
                api_info=api_info,
                task_methods_content=indented_task_methods,
                groups=groups,
            )

        except Exception as e:
            logger.error(f"Failed to generate test suite: {e}")
            # Return fallback files
            return self._generate_fallback_locustfile(api_info)

    def _indent_methods(self, task_methods: List[str], indent_level: int = 1) -> str:
        """Properly indent task methods for class inclusion"""
        indented_methods = []
        for method in task_methods:
            lines = method.split("\n")
            indented_lines = []
            first_nonempty = True

            for line in lines:
                if line.strip():
                    stripped_line = line.lstrip()
                    if first_nonempty:
                        # First non-empty line → indent once (for @task or def)
                        indented_lines.append("    " * indent_level + stripped_line)
                        first_nonempty = False
                    else:
                        # Method body → indent deeper
                        indented_lines.append(
                            "    " * (indent_level + 1) + stripped_line
                        )
                else:
                    indented_lines.append("")
            indented_methods.append("\n".join(indented_lines))

        return "\n\n".join(indented_methods)

    def _generate_fallback_locustfile(self, api_info: Dict[str, Any]) -> str:
        """Generate a basic fallback locustfile when main generation fails"""

        template = self.jinja_env.get_template("fallback_locust.py.j2")
        return template.render(api_info=api_info)

    from typing import Dict, Any

    def _build_locustfile_template(
        self, api_info: Dict[str, Any], task_methods_content: str, groups: List[str]
    ) -> str:
        import_group_tasks = ""
        tasks = []
        for group in groups:
            file_name = group.lower().replace("-", "_")
            class_name = group.capitalize().replace("-", "")
            import_group_tasks += f"""from workflows.{file_name}_workflow import {class_name}TaskMethods\n"""
            tasks.append(f"{class_name}TaskMethods")
        tasks_str = "[" + ",".join(tasks) + "]"
        template = self.jinja_env.get_template("locust.py.j2")

        # Prepare template context
        context = {
            "import_group_tasks": import_group_tasks,
            "task_methods_content": task_methods_content,
            "tasks_str": tasks_str,
            "api_info": api_info,
            "generated_task_classes": self._generate_user_classes(),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Render the template
        content = template.render(**context)

        logger.info("README generated successfully using template")
        return content

    def _generate_default_task_method(self) -> str:
        """Generate a default task method when no endpoints are available"""
        return '''@task(1)
    def default_health_check(self):
        """Default health check task"""
        try:
            response_data = self.make_request(
                method="get",
                path="/health"
            )

            if response_data:
                self._store_response_data("health_check", response_data)

        except Exception as e:
            logger.error(f"Health check task failed: {e}")
    '''

    def _generate_task_method(self, endpoint: Any) -> str:
        """Generate a Locust task method for a single endpoint with improved structure"""
        try:
            method_name = self._generate_method_name(endpoint)
            path_with_params = self._generate_path_with_params(endpoint)
            weight = self._get_task_weight(getattr(endpoint, "method", "GET"))

            # Build the task method with proper error handling
            task_method = f'''@task({weight})
    def {method_name}(self):
        """
        {getattr(endpoint, "summary", f"{getattr(endpoint, 'method', 'GET')} {getattr(endpoint, 'path', '')}")}
        {getattr(endpoint, "description", "")}
        """
        try:
            # Generate path parameters
            {self._generate_path_params_code(endpoint)}

            # Generate query parameters
            {self._generate_query_params_code(endpoint)}

            # Generate request body
            {self._generate_request_body_code(endpoint)}

            # Make the request
            response_data = self.make_request(
                method="{getattr(endpoint, "method", "GET").lower()}",
                path=f"{path_with_params}",
                {self._generate_request_kwargs(endpoint)}
            )

            if response_data:
                # Store response data for dependent requests
                self._store_response_data("{method_name}", response_data)

        except Exception as e:
            logger.error(f"Task {method_name} failed: {{e}}")
    '''
            return task_method

        except Exception as e:
            logger.error(f"Failed to generate task method for 535 endpoint: {e}")
            return ""

    def _generate_method_name(self, endpoint: Endpoint) -> str:
        """Generate a valid Python method name from endpoint"""
        if endpoint.operation_id:
            # Use operation ID if available
            name = endpoint.operation_id
        else:
            # Generate name from method and path
            path_parts = [
                part
                for part in endpoint.path.split("/")
                if part and not part.startswith("{")
            ]
            name = f"{endpoint.method.lower()}_{'_'.join(path_parts)}"

        # Clean up the name
        name = re.sub(r"[^\w]", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")

        return name if name else f"{endpoint.method.lower()}_endpoint"

    def _generate_path_with_params(self, endpoint: Endpoint) -> str:
        """Generate path with parameter placeholders"""
        path = endpoint.path

        # Replace path parameters with f-string format
        for param in endpoint.parameters:
            if param.location.value == "path":
                path = path.replace(f"{{{param.name}}}", f"{{{param.name}}}")

        return path

    def _generate_path_params_code(self, endpoint: Endpoint) -> str:
        """Generate code for path parameters"""
        path_params = [p for p in endpoint.parameters if p.location.value == "path"]

        if not path_params:
            return "# No path parameters"

        code_lines = []
        for param in path_params:
            if param.type.startswith("integer"):
                code_lines.append(f"{param.name} = data_generator.generate_integer()")
            elif param.type == "string":
                if "id" in param.name.lower():
                    code_lines.append(f"{param.name} = data_generator.generate_id()")
                else:
                    code_lines.append(
                        f"{param.name} = data_generator.generate_string()"
                    )

            else:
                code_lines.append(
                    f'{param.name} = data_generator.generate_value("{param.type}")'
                )

        return "\n".join(code_lines)

    def _generate_query_params_code(self, endpoint: Endpoint) -> str:
        """Generate code for query parameters"""
        query_params = [p for p in endpoint.parameters if p.location.value == "query"]

        if not query_params:
            return "params = {}"

        param_lines = []
        for param in query_params:
            if self._should_skip_optional_param(param):
                continue

            param_line = self._generate_param_line(param)
            param_lines.append(param_line)

        return self._format_params_dict(param_lines)

    def _should_skip_optional_param(self, param: Parameter) -> bool:
        """Randomly skip optional parameters 30% of the time"""
        return not param.required and secrets.randbelow(100) > 70

    def _generate_param_line(self, param: Parameter) -> str:
        """Generate a single parameter line based on its type"""
        param_generators = {
            "integer": self._generate_integer_param,
            "string": self._generate_string_param,
            "boolean": self._generate_boolean_param,
        }

        # Handle integer types that might have prefixes like "integer64"
        param_type = "integer" if param.type.startswith("integer") else param.type

        generator = param_generators.get(param_type, self._generate_generic_param)
        return generator(param)

    def _generate_integer_param(self, param: Parameter) -> str:
        """Generate integer parameter line"""
        default = param.default if param.default is not None else "None"
        return f'"{param.name}": data_generator.generate_integer(default={default}),'

    def _generate_string_param(self, param: Parameter) -> str:
        """Generate string parameter line"""
        default = f'"{param.default}"' if param.default else "None"
        return f'"{param.name}": data_generator.generate_string(default={default}),'

    def _generate_boolean_param(self, param: Parameter) -> str:
        """Generate boolean parameter line"""
        return f'"{param.name}": data_generator.generate_boolean(),'

    def _generate_generic_param(self, param: Parameter) -> str:
        """Generate generic parameter line for unknown types"""
        return f'"{param.name}": data_generator.generate_value("{param.type}"),'

    def _format_params_dict(self, param_lines: list[str]) -> str:
        """Format parameter lines into a dictionary structure"""
        if not param_lines:
            return "params = {}"

        lines = ["params = {"] + [f"    {line}" for line in param_lines] + ["}"]
        return "\n".join(lines)

    def _generate_request_body_code(self, endpoint: Endpoint) -> str:
        """Generate code for request body"""
        if not endpoint.request_body:
            return "json_data = None"

        if endpoint.request_body.content_type == "application/json":
            return f"""json_data = data_generator.generate_json_data(
                schema={json.dumps(endpoint.request_body.schema, indent=16)}
            )"""
        elif endpoint.request_body.content_type == "application/x-www-form-urlencoded":
            return "data = data_generator.generate_form_data()"
        else:
            return "json_data = {}"

    def _generate_request_kwargs(self, endpoint: Endpoint) -> str:
        """Generate kwargs for the request method"""
        kwargs = []

        # Add query parameters
        query_params = [p for p in endpoint.parameters if p.location.value == "query"]
        if query_params:
            kwargs.append("params=params")

        # Add request body
        if endpoint.request_body:
            if endpoint.request_body.content_type == "application/json":
                kwargs.append("json=json_data")
            elif (
                endpoint.request_body.content_type
                == "application/x-www-form-urlencoded"
            ):
                kwargs.append("data=data")

        # Add headers if needed
        header_params = [p for p in endpoint.parameters if p.location.value == "header"]
        if header_params:
            kwargs.append("headers=headers")

        return ",\n                ".join(kwargs)

    def _get_task_weight(self, method: str) -> int:
        """Get task weight based on HTTP method"""
        weights = {
            "GET": 5,  # Most frequent
            "POST": 2,  # Common
            "PUT": 1,  # Less frequent
            "PATCH": 1,  # Less frequent
            "DELETE": 1,  # Least frequent
            "HEAD": 3,  # Moderate
            "OPTIONS": 1,  # Rare
        }
        return weights.get(method.upper(), 1)

    def _group_endpoints_by_tag(
        self,
        endpoints: List[Endpoint],
        include_auth_endpoints: bool = True,
    ) -> Dict[str, List[Endpoint]]:
        """Group endpoints by their tags"""
        grouped: Dict[str, List[Endpoint]] = {}
        # Define authentication-related keywords to check for in paths
        auth_keywords = [
            "login",
            "signin",
            "sign-in",
            "sign_in",
            "logout",
            "signout",
            "sign-out",
            "sign_out",
            "auth",
            "authenticate",
            "authorization",
            "token",
            "refresh",
            "verify",
            "password",
            "reset",
            "forgot",
            "register",
            "signup",
            "sign-up",
            "sign_up",
            "session",
            "oauth",
            "sso",
        ]

        def is_auth_endpoint(endpoint_path: str) -> bool:
            """Check if endpoint path contains authentication-related keywords"""
            path_lower = endpoint_path.lower()
            return any(keyword in path_lower for keyword in auth_keywords)

        for endpoint in endpoints:
            tags = endpoint.tags if endpoint.tags else ["default"]
            if is_auth_endpoint(endpoint.path) and include_auth_endpoints:
                # Add to authentication group regardless of tags
                if "Authentication" not in grouped:
                    grouped["Authentication"] = []
                grouped["Authentication"].append(endpoint)

            for tag in tags:
                if tag not in grouped:
                    grouped[tag] = []
                grouped[tag].append(endpoint)

        return grouped

    def _generate_all_task_methods_string(self, endpoints: List[Endpoint]) -> str:
        """Generate all task methods as a properly indented string"""
        methods = []
        for endpoint in endpoints:
            method_code = self._generate_task_method(endpoint)
            methods.append(method_code)

        return "\n".join(methods)

    def _generate_user_classes(self) -> str:
        """
        **FIXED: Generate user classes with proper structure**
        """

        return '''
    class LightUser(BaseAPIUser, BaseTaskMethods):
        """Light user with occasional API usage patterns"""
        weight = 3
        wait_time = between(3, 8)  # Longer wait times

        def on_start(self):
            super().on_start()
            self.user_type = "light"
   

    class RegularUser( BaseAPIUser, BaseTaskMethods):
        """Regular user with normal API usage patterns"""
        weight = 4
        wait_time = between(1, 4)  # Moderate wait times
        
        def on_start(self):
            super().on_start()
            self.user_type = "regular"
        


    class PowerUser( BaseAPIUser, BaseTaskMethods):
        """Power user with heavy API usage patterns"""
        weight = 3
        wait_time = between(0.5, 2)  # Shorter wait times

        def on_start(self):
            super().on_start()
            self.user_type = "power"
            

    '''

    def _generate_test_data_file(self, db_type: str = "") -> str:
        """Generate test_data.py file content"""
        data_provider_content = None
        if db_type == DatabaseType.MONGO.value:
            data_provider_content = "mongo_data_provider"
        template = self.jinja_env.get_template("test_data.py.j2")
        return template.render(data_provider_content=data_provider_content)

    def _generate_config_file(self, api_info: Dict[str, Any]) -> str:
        """Generate config.py file content"""
        template = self.jinja_env.get_template("config.py.j2")
        return template.render(api_info=api_info)

    def _generate_utils_file(self) -> str:
        """Generate utils.py file content"""
        template = self.jinja_env.get_template("utils.py.j2")
        return template.render()

    def _generate_custom_flows_file(self) -> str:
        """Generate custom_flows.py file content"""
        template = self.jinja_env.get_template("custom_flows.py.j2")
        return template.render()

    def _generate_requirements_file(self) -> str:
        """Generate requirements.txt file content"""
        template = self.jinja_env.get_template("requirement.txt.j2")
        content = template.render()
        return content

    def _generate_db_file(self, db_type: str, file_name: str) -> str:
        """Generate db file content"""
        template = self.jinja_env.get_template(db_type + "/" + file_name)
        return template.render()

    def _generate_env_example(
        self,
        api_info: Dict[str, Any],
        target_host: Optional[str] = None,
        db_type: str = "",
    ) -> str:
        """Generate .env.example file content"""
        try:
            template = self.jinja_env.get_template("env.example.j2")
            if target_host:
                base_url = target_host
                locust_host = target_host
            else:
                base_url = api_info.get("base_url", "http://localhost:8000")
                locust_host = api_info.get("base_url", "http://localhost:8000")
            # Prepare environment variables context
            environment_vars = {
                "API_BASE_URL": base_url,
                "API_VERSION": api_info.get("version", "v1"),
                "API_TITLE": api_info.get("title", "Your API Name"),
                "LOCUST_USERS": "50",
                "LOCUST_SPAWN_RATE": "5",
                "LOCUST_RUN_TIME": "10m",
                "LOCUST_HOST": locust_host,
                "DATA_SEED": "42",
                "REQUEST_TIMEOUT": "30",
                "MAX_RETRIES": "3",
            }

            if db_type == DatabaseType.MONGO.value:
                mongodb_config = MongoDBConfig()
                environment_vars.update(asdict(mongodb_config))

            context = {
                "environment_vars": environment_vars,
                "api_info": api_info,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            content = template.render(**context)
            logger.info("✅ .env.example generated successfully using template")
            return content

        except Exception as e:
            logger.error(f"❌ Failed to generate .env.example from template: {e}")
            return ""

    def _generate_readme_file(self, api_info: Dict[str, Any], db_type: str = "") -> str:
        try:
            # Get the template
            template = self.jinja_env.get_template("readme.md.j2")
            db_using = ""
            if db_type == DatabaseType.MONGO.value:
                template_db = self.jinja_env.get_template(
                    DatabaseType.MONGO.value + "/db_integration.j2"
                )

                db_using = template_db.render()

            # Prepare template context
            context = {
                "api_info": api_info,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "db_using": db_using,
            }

            # Render the template
            content = template.render(**context)

            logger.info("README generated successfully using template")
            return content

        except Exception as e:
            logger.error(f"Error generating README: {e}")
            return ""

    def generate_base_common_file(self, api_info: Dict[str, Any]) -> str:
        template = self.jinja_env.get_template("base_workflow.py.j2")
        return template.render(api_info=api_info)
