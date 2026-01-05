"""
Hybrid Locust Test Generator

Combines reliable template-based generation with LLM enhancement for creativity
and domain-specific optimizations.
"""

import ast
import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from dataclasses import dataclass
import uuid
import shutil


from devdox_ai_locust.utils.open_ai_parser import Endpoint
from devdox_ai_locust.utils.file_creation import FileCreationConfig, SafeFileCreator
from devdox_ai_locust.locust_generator import LocustTestGenerator, TestDataConfig
from together import AsyncTogether

logger = logging.getLogger(__name__)


test_data_file_path = "test_data.py"
data_provider_path = "data_provider.py"
base_workflow_path = "base_workflow.py"
workflow_jinja_path = "workflow.j2"

# Critical classes that MUST exist in each file after AI enhancement
# If these are missing, the AI has corrupted the file and we must use the original
CRITICAL_CLASSES = {
    "test_data.py": ["TestDataGenerator"],
    "utils.py": ["ResponseValidator", "RequestLogger", "PerformanceMonitor", "DataManager"],
}

# Critical functions that MUST exist in each file
CRITICAL_FUNCTIONS = {
    "test_data.py": ["generate_json_data", "generate_string", "generate_id"],
    "utils.py": ["validate_response", "log_request"],
}


class SafeCodeMerger:
    """
    Safely merges AI-generated code additions into original code.

    This approach:
    1. ALWAYS keeps the original code intact
    2. Only ADDS new methods/classes from AI output
    3. Never replaces or modifies existing code
    4. Uses AST parsing for safety
    5. Prevents duplicate class definitions (including auth classes)
    6. Prevents redefining classes that are already imported
    """

    # Classes that should never be duplicated - if they exist anywhere, don't add again
    SINGLETON_CLASSES = {
        "TestDataGenerator",
        "ResponseValidator",
        "RequestLogger",
        "PerformanceMonitor",
        "DataManager",
        "LoadTestConfig",
    }

    # Keywords that indicate auth-related classes - only one auth class should exist
    AUTH_CLASS_KEYWORDS = {"auth", "login", "logout", "token", "session", "credential"}

    @staticmethod
    def get_imported_classes(code: str) -> Set[str]:
        """
        Extract class names that are imported from other modules.

        Returns:
            Set of class names that are imported
        """
        imported_classes: Set[str] = set()

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        # Check if it looks like a class (starts with uppercase)
                        name = alias.asname if alias.asname else alias.name
                        if name and name[0].isupper():
                            imported_classes.add(name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        if name and name[0].isupper():
                            imported_classes.add(name)
        except SyntaxError:
            # Fallback to regex for imports
            # Match: from module import ClassName
            from_imports = re.findall(r"from\s+\S+\s+import\s+([A-Z]\w+)", code)
            imported_classes.update(from_imports)
            # Match: from module import ClassName as Alias
            aliased_imports = re.findall(r"from\s+\S+\s+import\s+\w+\s+as\s+([A-Z]\w+)", code)
            imported_classes.update(aliased_imports)

        return imported_classes

    @staticmethod
    def is_auth_related_class(class_name: str) -> bool:
        """Check if a class name indicates it's authentication-related."""
        lower_name = class_name.lower()
        return any(keyword in lower_name for keyword in SafeCodeMerger.AUTH_CLASS_KEYWORDS)

    @staticmethod
    def has_auth_class(code: str) -> bool:
        """Check if code already contains an auth-related class."""
        class_names, _, _ = SafeCodeMerger.get_existing_names(code)
        return any(SafeCodeMerger.is_auth_related_class(name) for name in class_names)

    @staticmethod
    def get_existing_names(code: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Extract existing class names, method names, and function names from code.

        Returns:
            Tuple of (class_names, method_names, function_names)
        """
        class_names: Set[str] = set()
        method_names: Set[str] = set()
        function_names: Set[str] = set()

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_names.add(node.name)
                    # Get methods within the class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_names.add(f"{node.name}.{item.name}")
                elif isinstance(node, ast.FunctionDef):
                    # Top-level function
                    if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                        function_names.add(node.name)
        except SyntaxError:
            logger.warning("Failed to parse code with AST, falling back to regex")
            # Fallback to regex
            class_names = set(re.findall(r'class\s+(\w+)\s*[:\(]', code))
            function_names = set(re.findall(r'^def\s+(\w+)\s*\(', code, re.MULTILINE))

        return class_names, method_names, function_names

    @staticmethod
    def extract_new_methods_only(original_code: str, ai_code: str, target_class: str = None) -> str:
        """
        Extract ONLY new methods from AI code that don't exist in original.

        This is a conservative approach that:
        1. Identifies methods in AI output
        2. Filters out any that already exist in original
        3. Returns only the truly new additions
        4. Prevents duplicate class definitions (auth classes, imported classes)
        5. Prevents redefining classes that are imported

        Handles both:
        - Full class definitions from AI
        - Standalone method definitions (method-only output)
        """
        if not ai_code or not ai_code.strip():
            return ""

        orig_classes, orig_methods, orig_functions = SafeCodeMerger.get_existing_names(original_code)

        # Get imported classes to prevent redefining them
        imported_classes = SafeCodeMerger.get_imported_classes(original_code)

        # Check if original already has auth-related classes
        original_has_auth = SafeCodeMerger.has_auth_class(original_code)

        # Get all existing method names (without class prefix) for comparison
        existing_method_names = set()
        for method in orig_methods:
            if "." in method:
                existing_method_names.add(method.split(".")[-1])
        existing_method_names.update(orig_functions)

        try:
            ai_tree = ast.parse(ai_code)
        except SyntaxError:
            logger.warning("AI code has syntax errors, trying to extract methods via regex")
            # Fallback: extract method definitions via regex
            return SafeCodeMerger._extract_methods_regex(ai_code, existing_method_names)

        new_methods = []

        for node in ast.iter_child_nodes(ai_tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name

                # Check if this is an existing class we should add methods to
                if class_name in orig_classes:
                    # Extract only new methods from this class (even for singleton classes)
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if item.name not in existing_method_names:
                                try:
                                    method_source = ast.get_source_segment(ai_code, item)
                                    if method_source:
                                        # Ensure proper indentation for class method
                                        indented = SafeCodeMerger._indent_code(method_source, 4)
                                        new_methods.append(f"    # AI-added method\n{indented}")
                                except Exception:
                                    pass
                    # Don't add the class itself, just the methods
                    continue

                # For NEW classes (not in orig_classes), apply deduplication rules:

                # Skip classes that are already imported (Bug #3 fix)
                if class_name in imported_classes:
                    logger.info(f"Skipping class '{class_name}' - already imported")
                    continue

                # Skip singleton classes that shouldn't be duplicated
                if class_name in SafeCodeMerger.SINGLETON_CLASSES:
                    logger.info(f"Skipping singleton class '{class_name}' - would create duplicate")
                    continue

                # Skip auth-related classes if original already has auth (Bug #1 fix)
                if SafeCodeMerger.is_auth_related_class(class_name) and original_has_auth:
                    logger.info(f"Skipping auth class '{class_name}' - original already has auth")
                    continue

                # This is a genuinely new class - but we only extract methods, not full classes
                # to prevent adding duplicate class definitions
                logger.debug(f"Skipping new class '{class_name}' - only methods are extracted")

            elif isinstance(node, ast.FunctionDef):
                # Standalone method definition (AI returned methods only)
                if node.name not in existing_method_names:
                    try:
                        method_source = ast.get_source_segment(ai_code, node)
                        if method_source:
                            # Add indentation for class method
                            indented = SafeCodeMerger._indent_code(method_source, 4)
                            new_methods.append(f"    # AI-added method\n{indented}")
                    except Exception:
                        pass

        return "\n\n".join(new_methods)

    @staticmethod
    def _indent_code(code: str, spaces: int) -> str:
        """Add indentation to code block."""
        indent = " " * spaces
        lines = code.split("\n")
        # Check if already indented
        if lines and lines[0].startswith(" " * spaces):
            return code
        return "\n".join(indent + line if line.strip() else line for line in lines)

    @staticmethod
    def _extract_methods_regex(ai_code: str, existing_methods: Set[str]) -> str:
        """Fallback method extraction using regex when AST fails."""
        # Match method definitions
        pattern = r'(def\s+(\w+)\s*\([^)]*\).*?(?=\ndef\s|\Z))'
        matches = re.findall(pattern, ai_code, re.DOTALL)

        new_methods = []
        for full_match, method_name in matches:
            if method_name not in existing_methods:
                indented = SafeCodeMerger._indent_code(full_match.strip(), 4)
                new_methods.append(f"    # AI-added method\n{indented}")

        return "\n\n".join(new_methods)

    @staticmethod
    def safe_merge(original_code: str, ai_additions: str, target_class: str = None) -> str:
        """
        Safely merge AI additions into original code.

        This method:
        1. ALWAYS returns the original code as base
        2. Only appends new methods that don't exist
        3. Never modifies existing code

        Args:
            original_code: The original template-generated code (ALWAYS preserved)
            ai_additions: Code generated by AI (only new parts used)
            target_class: If specified, add new methods to this class

        Returns:
            Original code with safe additions appended
        """
        if not ai_additions or not ai_additions.strip():
            logger.info("No AI additions to merge, returning original")
            return original_code

        # Extract only new methods (handles syntax errors internally)
        new_methods = SafeCodeMerger.extract_new_methods_only(
            original_code, ai_additions, target_class
        )

        if not new_methods:
            logger.info("No new methods found in AI output, returning original")
            return original_code

        # Find where to insert new methods (at the end of the target class)
        if target_class:
            # Find the class definition and locate the end of the class
            # Look for the class and find where it ends
            lines = original_code.split('\n')
            class_start = -1
            class_indent = 0

            for i, line in enumerate(lines):
                if re.match(rf'\s*class\s+{target_class}\s*[\(:]', line):
                    class_start = i
                    class_indent = len(line) - len(line.lstrip())
                    break

            if class_start >= 0:
                # Find the end of the class (next line with same or less indentation that's not empty)
                class_end = len(lines)
                for i in range(class_start + 1, len(lines)):
                    line = lines[i]
                    if line.strip() and not line.strip().startswith('#'):
                        current_indent = len(line) - len(line.lstrip())
                        if current_indent <= class_indent and not line.strip().startswith('def '):
                            # Check if this is a new class or module-level code
                            if re.match(r'\s*(class\s|def\s|if\s+__name__|@|[A-Z_]+\s*=)', line):
                                class_end = i
                                break

                # Insert new methods before the class ends
                new_lines = lines[:class_end]
                new_lines.append("")
                new_lines.append(new_methods)
                new_lines.extend(lines[class_end:])

                merged = '\n'.join(new_lines)
                logger.info(f"Added new methods to {target_class}")
                return merged

        # Fallback: append at end of file
        logger.info("Appending new methods at end of file")
        return original_code + "\n\n# AI-enhanced additions\n" + new_methods


@dataclass
class ProtectedSymbol:
    """A symbol that is protected because other files depend on it."""
    name: str
    symbol_type: str  # 'class', 'function', 'method', 'constant'
    defined_in: str  # file where it's defined
    used_by: List[str]  # files that import/use it
    reason: str  # why it's protected


def format_auth_endpoints_for_prompt(auth_endpoints: List["Endpoint"]) -> str:
    """
    Format auth_endpoints as human-readable text for AI prompts.

    Instead of rendering Python object repr like:
        [Endpoint(path='/api/v1/auth/login', method='POST', ...)]

    This produces readable text like:
        Available authentication endpoints:
        - POST /api/v1/auth/login (login_api_v1_auth_login_post)
        - POST /api/v1/auth/logout (logout_api_v1_auth_logout_post)

    This prevents the AI from copying Python code into the generated output.
    """
    if not auth_endpoints:
        return "No authentication endpoints available. Use config.api_key for authorization if needed."

    lines = ["Available authentication endpoints:"]
    for endpoint in auth_endpoints:
        method = getattr(endpoint, 'method', 'UNKNOWN')
        path = getattr(endpoint, 'path', '/unknown')
        operation_id = getattr(endpoint, 'operation_id', '')
        summary = getattr(endpoint, 'summary', '')

        line = f"  - {method} {path}"
        if operation_id:
            line += f" (operation_id: {operation_id})"
        if summary:
            line += f" - {summary}"
        lines.append(line)

    lines.append("")
    lines.append("Use these EXACT paths in your authentication methods.")
    lines.append("DO NOT invent or hardcode endpoint paths - use only what's listed above.")

    return "\n".join(lines)


class CodebaseAwareness:
    """
    Analyzes the generated codebase to understand dependencies between files.

    This creates a "sandbox" for AI enhancement by identifying:
    1. What symbols (classes, functions) each file exports
    2. What each file imports from other files
    3. Which symbols are "protected" (used by other files)
    4. Which symbols are "free" (can be modified/removed)

    The AI is then given this context so it knows what it MUST preserve.
    """

    def __init__(self):
        self.files: Dict[str, str] = {}  # filename -> content
        self.exports: Dict[str, Set[str]] = {}  # filename -> exported symbols
        self.imports: Dict[str, Dict[str, Set[str]]] = {}  # filename -> {source_file: symbols}
        self.protected: Dict[str, List[ProtectedSymbol]] = {}  # filename -> protected symbols

    def analyze_codebase(
        self,
        base_files: Dict[str, str],
        directory_files: List[Dict[str, Any]]
    ) -> None:
        """
        Analyze all generated files to build the dependency map.

        Args:
            base_files: Main files like locustfile.py, test_data.py, utils.py
            directory_files: Workflow files in subdirectories
        """
        # Collect all files
        self.files = base_files.copy()
        for file_dict in directory_files:
            self.files.update(file_dict)

        # Analyze each file
        for filename, content in self.files.items():
            self._analyze_file(filename, content)

        # Build protected symbols map
        self._build_protected_map()

    def _analyze_file(self, filename: str, content: str) -> None:
        """Analyze a single file for exports and imports."""
        if not content:
            return

        # Extract exports (classes, functions, constants defined in this file)
        self.exports[filename] = self._extract_exports(content)

        # Extract imports from other local files
        self.imports[filename] = self._extract_local_imports(content)

    def _extract_exports(self, content: str) -> Set[str]:
        """Extract all symbols defined in this file."""
        exports = set()

        try:
            tree = ast.parse(content)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    exports.add(node.name)
                    # Also add methods as ClassName.method_name
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                            exports.add(f"{node.name}.{item.name}")
                elif isinstance(node, ast.FunctionDef):
                    exports.add(node.name)
                elif isinstance(node, ast.Assign):
                    # Constants like SOME_VALUE = ...
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            exports.add(target.id)
        except SyntaxError:
            # Fallback to regex
            exports.update(re.findall(r'^class\s+(\w+)', content, re.MULTILINE))
            exports.update(re.findall(r'^def\s+(\w+)', content, re.MULTILINE))
            exports.update(re.findall(r'^([A-Z_]+)\s*=', content, re.MULTILINE))

        return exports

    def _extract_local_imports(self, content: str) -> Dict[str, Set[str]]:
        """Extract imports from other local files."""
        imports: Dict[str, Set[str]] = {}

        # Match patterns like:
        # from test_data import TestDataGenerator
        # from utils import ResponseValidator, RequestLogger
        # from workflows.base_workflow import BaseWorkflow
        patterns = [
            r'from\s+(\w+)\s+import\s+([^#\n]+)',
            r'from\s+workflows\.(\w+)\s+import\s+([^#\n]+)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content):
                source_file = match.group(1)
                if not source_file.endswith('.py'):
                    source_file += '.py'
                imported_symbols = match.group(2)

                # Parse the imported symbols
                symbols = set()
                for sym in imported_symbols.split(','):
                    sym = sym.strip()
                    # Handle 'as' aliases
                    if ' as ' in sym:
                        sym = sym.split(' as ')[0].strip()
                    if sym and sym != '*':
                        symbols.add(sym)

                if source_file not in imports:
                    imports[source_file] = set()
                imports[source_file].update(symbols)

        return imports

    def _build_protected_map(self) -> None:
        """Build the map of protected symbols for each file."""
        for filename in self.files:
            self.protected[filename] = []

        # For each file, find which of its exports are used by other files
        for filename, exports in self.exports.items():
            for symbol in exports:
                used_by = []

                # Check which other files import this symbol
                for other_file, other_imports in self.imports.items():
                    if other_file == filename:
                        continue

                    # Check if this file imports from our file
                    for source, symbols in other_imports.items():
                        if source == filename or source.replace('.py', '') in filename:
                            if symbol in symbols or symbol.split('.')[-1] in symbols:
                                used_by.append(other_file)

                if used_by:
                    # Determine symbol type
                    if '.' in symbol:
                        symbol_type = 'method'
                    elif symbol.isupper():
                        symbol_type = 'constant'
                    elif symbol[0].isupper():
                        symbol_type = 'class'
                    else:
                        symbol_type = 'function'

                    protected_symbol = ProtectedSymbol(
                        name=symbol,
                        symbol_type=symbol_type,
                        defined_in=filename,
                        used_by=used_by,
                        reason=f"Imported by: {', '.join(used_by)}"
                    )
                    self.protected[filename].append(protected_symbol)

    def get_constraints_for_file(self, filename: str) -> str:
        """
        Generate AI constraints for a specific file.

        Returns a formatted string that tells the AI what it MUST preserve.
        """
        if filename not in self.protected:
            return ""

        protected_symbols = self.protected[filename]
        if not protected_symbols:
            return "No external dependencies. You can freely modify this file."

        constraints = []
        constraints.append("🔒 PROTECTED SYMBOLS (DO NOT REMOVE - used by other files):")
        constraints.append("")

        # Group by type
        by_type: Dict[str, List[ProtectedSymbol]] = {}
        for sym in protected_symbols:
            if sym.symbol_type not in by_type:
                by_type[sym.symbol_type] = []
            by_type[sym.symbol_type].append(sym)

        for sym_type, symbols in by_type.items():
            constraints.append(f"  {sym_type.upper()}ES:")
            for sym in symbols:
                constraints.append(f"    - {sym.name}")
                constraints.append(f"      Reason: {sym.reason}")
            constraints.append("")

        constraints.append("✅ You MAY:")
        constraints.append("  - ADD new methods to existing classes")
        constraints.append("  - ADD new helper functions")
        constraints.append("  - MODIFY method implementations (keep signatures)")
        constraints.append("  - ADD new classes")
        constraints.append("")
        constraints.append("❌ You MUST NOT:")
        constraints.append("  - DELETE or RENAME any protected symbol above")
        constraints.append("  - CHANGE the signature of protected methods")
        constraints.append("  - REMOVE imports that other files depend on")

        return "\n".join(constraints)

    def get_full_context(self) -> str:
        """Get a summary of the entire codebase structure for AI context."""
        context = []
        context.append("📁 CODEBASE STRUCTURE:")
        context.append("")

        for filename, exports in self.exports.items():
            context.append(f"  {filename}:")
            if exports:
                context.append(f"    Exports: {', '.join(sorted(exports)[:10])}")
                if len(exports) > 10:
                    context.append(f"    ... and {len(exports) - 10} more")
            context.append("")

        return "\n".join(context)


@dataclass
class ErrorClassification:
    """Classification of an error for retry logic"""

    is_retryable: bool
    backoff_seconds: float
    error_type: str


@dataclass
class AIEnhancementConfig:
    """Configuration for AI enhancement"""

    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    max_tokens: int = 8000
    temperature: float = 0.3
    timeout: int = 60
    enhance_workflows: bool = True
    enhance_test_data: bool = True
    enhance_validation: bool = True
    create_domain_flows: bool = True
    update_main_locust: bool = True


@dataclass
class EnhancementResult:
    """Result of AI enhancement"""

    success: bool
    enhanced_files: Dict[str, str]
    enhanced_directory_files: List[Dict[str, Any]]
    enhancements_applied: List[str]
    errors: List[str]
    processing_time: float


class EnhancementProcessor:
    """Handles individual enhancement operations"""

    def __init__(
        self,
        ai_config: Optional[AIEnhancementConfig],
        locust_generator: "HybridLocustGenerator",
        awareness: Optional[CodebaseAwareness] = None,
    ) -> None:
        self.ai_config = ai_config
        self.locust_generator = locust_generator
        self.awareness = awareness

    async def process_main_locust_enhancement(
        self,
        base_files: Dict[str, str],
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
    ) -> Tuple[Dict[str, str], List[str]]:
        """Process main locustfile enhancement"""
        enhanced_files = {}
        enhancements = []

        if self.ai_config and self.ai_config.update_main_locust:
            enhanced_content = await self.locust_generator._enhance_locustfile(
                base_files.get("locustfile.py", ""), endpoints, api_info
            )
            if enhanced_content:
                enhanced_files["locustfile.py"] = enhanced_content
                enhancements.append("main_locust_update")
        return enhanced_files, enhancements

    async def process_domain_flows_enhancement(
        self,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        custom_requirement: Optional[str] = "",
    ) -> Tuple[Dict[str, str], List[str]]:
        """Process domain flows enhancement"""
        enhanced_files = {}
        enhancements = []

        if self.ai_config and self.ai_config.create_domain_flows:
            domain_flows = await self.locust_generator._generate_domain_flows(
                endpoints, api_info, custom_requirement=custom_requirement
            )
            if domain_flows:
                enhanced_files["custom_flows.py"] = domain_flows
                enhancements.append("domain_flows")

        return enhanced_files, enhancements

    def _get_base_workflow_content(self, directory_files: List[Dict[str, Any]]) -> str:
        base_files = self.locust_generator.get_files_by_key(
            directory_files, base_workflow_path
        )
        if not base_files:
            return ""
        workflow_dict = base_files[0]
        return workflow_dict.get(base_workflow_path) or ""

    async def _process_workflow_item(
        self,
        file_dict: Dict[str, Any],
        base_files: Dict[str, str],
        base_workflow_content: str,
        grouped_endpoints: Dict[str, List[Endpoint]],
        db_type: str,
        template: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        return await self._enhance_single_workflow(
            file_dict,
            base_files,
            base_workflow_content,
            grouped_endpoints,
            db_type,
            template_path=template or workflow_jinja_path,
        )

    async def process_workflow_enhancements(
        self,
        base_files: Dict[str, str],
        directory_files: List[Dict[str, Any]],
        grouped_endpoints: Dict[str, List[Endpoint]],
        db_type: str = "",
        include_auth: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Process workflow enhancements with minimal duplication."""

        # Early exit if enhancements disabled
        if not (self.ai_config and self.ai_config.enhance_workflows):
            return [], []

        enhanced_directory_files: List[Dict[str, Any]] = []
        enhancements: List[str] = []

        # Extract base workflow content
        base_workflow_content = self._get_base_workflow_content(directory_files)

        # Handle special case: enhancing base workflow if auth included
        if include_auth and base_workflow_content:
            result = await self._process_workflow_item(
                file_dict={base_workflow_path: base_workflow_content},
                base_files=base_files,
                base_workflow_content=base_workflow_content,
                grouped_endpoints=grouped_endpoints,
                db_type=db_type,
                template="base_workflow.j2",
            )
            if result:
                base_workflow_content = result["files"].get(base_workflow_path, "")
                enhancements.extend(result["enhancements"])
                # Add enhanced base_workflow.py to output
                enhanced_directory_files.append(result["files"])
            else:
                # Enhancement failed, preserve original base_workflow.py
                enhanced_directory_files.append({base_workflow_path: base_workflow_content})
        elif base_workflow_content:
            # include_auth is False but base_workflow.py exists - preserve it
            enhanced_directory_files.append({base_workflow_path: base_workflow_content})

        # Enhance all other workflow files
        for workflow_item in directory_files:
            # workflow_item is a dict like {"filename.py": "content"}
            # Skip base_workflow.py as it's already handled above
            if base_workflow_path in workflow_item:
                continue

            result = await self._process_workflow_item(
                file_dict=workflow_item,
                base_files=base_files,
                base_workflow_content=base_workflow_content,
                grouped_endpoints=grouped_endpoints,
                db_type=db_type,
                template="",
            )
            if result:
                enhanced_directory_files.append(result["files"])
                enhancements.extend(result["enhancements"])

        return enhanced_directory_files, enhancements

    async def _enhance_single_workflow(
        self,
        workflow_item: Dict[str, Any],
        base_files: Dict[str, str],
        base_workflow_files: str,
        grouped_endpoints: Dict[str, List[Endpoint]],
        db_type: str = "",
        template_path: str = workflow_jinja_path,
    ) -> Dict[str, Any] | None:
        """Enhance a single workflow file"""
        for key, value in workflow_item.items():
            workflow_key = key.replace("_workflow.py", "")
            endpoints_for_workflow = grouped_endpoints.get(workflow_key, [])
            auth_endpoints = grouped_endpoints.get("Authentication", [])
            workflow_endpoints_dict = {workflow_key: endpoints_for_workflow}
            enhanced_workflow = await self.locust_generator._enhance_workflows(
                base_content=value,
                test_data_content=base_files.get(test_data_file_path, ""),
                base_workflow=base_workflow_files,
                grouped_enpoints=workflow_endpoints_dict,
                auth_endpoints=auth_endpoints,
                db_type=db_type,
                template_path=template_path,
            )
            if enhanced_workflow:
                return {
                    "files": {key: enhanced_workflow},
                    "enhancements": [f"enhanced_workflows_{key}"],
                }

        return None

    async def process_test_data_enhancement(
        self, base_files: Dict[str, str], endpoints: List[Endpoint], db_type: str = ""
    ) -> Tuple[Dict[str, str], List[str]]:
        """Process test data enhancement with validation and fallback"""
        enhanced_files = {}
        enhancements = []
        original_content = base_files.get(test_data_file_path, "")

        # Get constraints from codebase awareness
        constraints = ""
        if self.awareness:
            constraints = self.awareness.get_constraints_for_file(test_data_file_path)
            if constraints:
                logger.info(f"📋 Applying constraints to {test_data_file_path}")

        if self.ai_config and self.ai_config.enhance_test_data and original_content:
            enhanced_test_data = await self.locust_generator.enhance_test_data_file(
                original_content,
                endpoints,
                db_type,
                base_files.get(data_provider_path, ""),
                base_files.get("db_config.py", ""),
                data_provider_path,
                constraints=constraints,  # Pass constraints to AI
            )
            # Use safe enhancement with validation and fallback
            validated_content = self.locust_generator._safe_enhance_file(
                test_data_file_path, enhanced_test_data, original_content
            )
            enhanced_files[test_data_file_path] = validated_content
            # Only mark as enhanced if we actually used the AI output
            if validated_content == enhanced_test_data:
                enhancements.append("smart_test_data")
            else:
                enhancements.append("test_data_fallback_to_original")

        return enhanced_files, enhancements

    async def process_validation_enhancement(
        self, base_files: Dict[str, str], endpoints: List[Endpoint]
    ) -> Tuple[Dict[str, str], List[str]]:
        """Process validation enhancement with validation and fallback"""
        enhanced_files = {}
        enhancements = []
        original_content = base_files.get("utils.py", "")

        # Get constraints from codebase awareness
        constraints = ""
        if self.awareness:
            constraints = self.awareness.get_constraints_for_file("utils.py")
            if constraints:
                logger.info(f"📋 Applying constraints to utils.py")

        if self.ai_config and self.ai_config.enhance_validation and original_content:
            enhanced_validation = await self.locust_generator._enhance_validation(
                original_content, endpoints, constraints=constraints
            )
            # Use safe enhancement with validation and fallback
            validated_content = self.locust_generator._safe_enhance_file(
                "utils.py", enhanced_validation, original_content
            )
            enhanced_files["utils.py"] = validated_content
            # Only mark as enhanced if we actually used the AI output
            if validated_content == enhanced_validation:
                enhancements.append("advanced_validation")
            else:
                enhancements.append("utils_fallback_to_original")

        return enhanced_files, enhancements


class HybridLocustGenerator:
    """
    Hybrid generator that combines template-based reliability with AI creativity
    """

    def __init__(
        self,
        ai_client: AsyncTogether,
        ai_config: Optional[AIEnhancementConfig] = None,
        test_config: Optional[TestDataConfig] = None,
        prompt_dir: str = "prompt",
    ):
        self.ai_client = ai_client
        self.ai_config = ai_config or AIEnhancementConfig()
        self.template_generator = LocustTestGenerator(test_config)
        self.prompt_dir = self._find_project_root() / prompt_dir
        self._api_semaphore = asyncio.Semaphore(5)
        self._setup_jinja_env()
        self.MAX_RETRIES = 3
        self.RATE_LIMIT_BACKOFF = 10
        self.NON_RETRYABLE_CODES = [
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "authentication",
            "unauthorized",
            "invalid token",
        ]
        self.RATE_LIMIT_INDICATORS = ["429", "rate limit"]

    def _find_project_root(self) -> Path:
        """Find the project root by looking for setup.py, pyproject.toml, or .git"""
        current_path = Path(__file__).parent

        return current_path

    def _setup_jinja_env(self) -> None:
        """Setup Jinja2 environment with custom filters"""
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=False,
        )

    def _classify_error(self, error: Exception, attempt: int) -> ErrorClassification:
        """
        Classify an error to determine retry behavior.

        Args:
            error: The exception that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            ErrorClassification with retry decision and backoff time
        """
        error_str = str(error).lower()

        # Non-retryable errors (auth/permission)
        if any(code in error_str for code in self.NON_RETRYABLE_CODES):
            logger.error(f"Authentication error, not retrying: {error}")
            return ErrorClassification(
                is_retryable=False, backoff_seconds=0, error_type="auth"
            )

        # Rate limit errors (retryable with longer backoff)
        if any(indicator in error_str for indicator in self.RATE_LIMIT_INDICATORS):
            logger.warning(f"Rate limit hit on attempt {attempt + 1}")
            return ErrorClassification(
                is_retryable=True,
                backoff_seconds=self.RATE_LIMIT_BACKOFF,
                error_type="rate_limit",
            )

        # Other retryable errors (exponential backoff)
        logger.warning(
            f"Retryable error on attempt {attempt + 1}: {type(error).__name__}"
        )
        return ErrorClassification(
            is_retryable=True,
            backoff_seconds=2**attempt,  # Exponential: 1s, 2s, 4s
            error_type="retryable",
        )

    async def generate_from_endpoints(
        self,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        custom_requirement: Optional[str] = None,
        target_host: Optional[str] = None,
        include_auth: bool = True,
        db_type: str = "",
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Generate Locust tests using hybrid approach

        1. Generate reliable base structure with templates
        2. Enhance with AI for domain-specific improvements
        3. Validate and merge results
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: Generate reliable base structure
            logger.info("🔧 Generating base test structure with templates...")
            base_files, directory_files, grouped_enpoints = (
                self.template_generator.generate_from_endpoints(
                    endpoints,
                    api_info,
                    include_auth=include_auth,
                    target_host=target_host,
                    db_type=db_type,
                )
            )

            base_files = self.template_generator.fix_indent(base_files)
            # Step 2: Enhance with AI if available
            if self.ai_client and self._should_enhance(endpoints, api_info):
                logger.info("🤖 Enhancing tests with AI...")
                enhancement_result = await self._enhance_with_ai(
                    base_files,
                    endpoints,
                    api_info,
                    include_auth,
                    directory_files,
                    grouped_enpoints,
                    custom_requirement,
                    db_type,
                )
                if enhancement_result.success:
                    logger.info(
                        f"✅ AI enhancements applied: {', '.join(enhancement_result.enhancements_applied)}"
                    )
                    return (
                        enhancement_result.enhanced_files,
                        enhancement_result.enhanced_directory_files,
                    )
                else:
                    logger.warning(
                        f"⚠️ AI enhancement failed, using template base: {', '.join(enhancement_result.errors)}"
                    )
            else:
                logger.info("📋 Using template-based generation only")

            processing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"⏱️ Generation completed in {processing_time:.2f}s")

            return base_files, directory_files

        except Exception as e:
            logger.error(f"Hybrid generation failed: {e}")

            return {}, []

    def _should_enhance(
        self, endpoints: List[Endpoint], api_info: Dict[str, Any]
    ) -> bool:
        """Determine if AI enhancement is worthwhile"""
        # Enhance if we have enough endpoints or complex schemas
        complex_endpoints = [
            ep
            for ep in endpoints
            if ep.request_body or len(ep.parameters) > 3 or len(ep.responses) > 2
        ]

        return (
            len(endpoints) >= 3
            or len(complex_endpoints)  # Enough endpoints for meaningful enhancement
            >= 1
            or self._detect_domain_patterns(  # Has complex endpoints
                endpoints, api_info
            )  # Has recognizable domain patterns
        )

    def _detect_domain_patterns(
        self, endpoints: List[Endpoint], api_info: Dict[str, Any]
    ) -> bool:
        """Detect if API belongs to known domains that benefit from custom flows"""
        domain_keywords = {
            "ecommerce": ["product", "cart", "order", "payment", "checkout"],
            "user_management": ["user", "auth", "login", "register", "profile"],
            "content_management": ["post", "article", "comment", "media", "upload"],
            "financial": ["transaction", "account", "balance", "transfer"],
            "social": ["friend", "follow", "message", "notification", "feed"],
        }

        api_text = f"{api_info.get('title', '')} {api_info.get('description', '')}"
        endpoint_paths = " ".join([ep.path for ep in endpoints])
        combined_text = f"{api_text} {endpoint_paths}".lower()

        for domain, keywords in domain_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return True

        return False

    async def _enhance_locustfile(
        self, base_content: str, endpoints: List[Any], api_info: Dict[str, Any]
    ) -> Optional[str]:
        # Configuration

        try:
            template = self.jinja_env.get_template("locust.j2")

            # Prepare context for template
            context = {
                "base_content": base_content,
                "endpoints_for_prompt": self._format_endpoints_for_prompt(
                    endpoints[:5]
                ),
                "api_info": api_info,
            }
            # Render enhanced content
            prompt = template.render(**context)
            enhanced_content = await self._call_ai_service(prompt)
            return enhanced_content
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return base_content

    async def _enhance_with_ai(
        self,
        base_files: Dict[str, str],
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        include_auth: bool,
        directory_files: List[Dict[str, Any]],
        grouped_endpoints: Dict[str, List[Endpoint]],
        custom_requirement: Optional[str] = None,
        db_type: str = "",
    ) -> EnhancementResult:
        """Enhance base files with AI - Refactored for reduced cognitive complexity"""
        start_time = asyncio.get_event_loop().time()

        try:
            enhancement_result = await self._process_all_enhancements(
                base_files,
                endpoints,
                api_info,
                include_auth,
                directory_files,
                grouped_endpoints,
                custom_requirement,
                db_type,
            )

            processing_time = asyncio.get_event_loop().time() - start_time
            enhancement_result.processing_time = processing_time

            return enhancement_result

        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time

            return EnhancementResult(
                success=False,
                enhanced_files=base_files,
                enhancements_applied=[],
                enhanced_directory_files=[],
                errors=[str(e)],
                processing_time=processing_time,
            )

    async def _process_all_enhancements(
        self,
        base_files: Dict[str, str],
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        include_auth: bool,
        directory_files: List[Dict[str, Any]],
        grouped_endpoints: Dict[str, List[Endpoint]],
        custom_requirement: Optional[str] = None,
        db_type: str = "",
    ) -> EnhancementResult:
        """Process all enhancements using the enhancement processor"""
        # Analyze codebase to understand dependencies
        awareness = CodebaseAwareness()
        awareness.analyze_codebase(base_files, directory_files)

        # Log the protected symbols for debugging
        logger.info("📊 Codebase analysis complete:")
        for filename, protected in awareness.protected.items():
            if protected:
                logger.info(f"  {filename}: {len(protected)} protected symbols")

        processor = EnhancementProcessor(self.ai_config, self, awareness)

        enhanced_files = base_files.copy()
        enhanced_directory_files = []
        enhancements_applied: List[str] = []
        errors = []
        # Process each enhancement type
        enhancement_tasks = [
            processor.process_main_locust_enhancement(base_files, endpoints, api_info),
            processor.process_domain_flows_enhancement(
                endpoints, api_info, custom_requirement
            ),
            processor.process_test_data_enhancement(base_files, endpoints, db_type),
            processor.process_validation_enhancement(base_files, endpoints),
        ]

        # Execute file-based enhancements concurrently
        file_enhancement_results = await asyncio.gather(
            *enhancement_tasks, return_exceptions=True
        )

        # Process results from file-based enhancements
        for result in file_enhancement_results:
            if isinstance(result, BaseException):
                errors.append(str(result))
                continue

            files, enhancements = result
            enhanced_files.update(files)
            enhancements_applied.extend(enhancements)

        # Process workflow enhancements separately (more complex logic)
        try:
            (
                workflow_files,
                workflow_enhancements,
            ) = await processor.process_workflow_enhancements(
                base_files, directory_files, grouped_endpoints, db_type, include_auth
            )
            enhanced_directory_files.extend(workflow_files)
            enhancements_applied.extend(workflow_enhancements)
        except Exception as e:
            errors.append(f"Workflow enhancement error: {str(e)}")

        return EnhancementResult(
            success=len(errors) == 0,
            enhanced_files=enhanced_files,
            enhanced_directory_files=enhanced_directory_files,
            enhancements_applied=enhancements_applied,
            errors=errors,
            processing_time=0,  # Will be set by caller
        )

    async def _generate_domain_flows(
        self,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        custom_requirement: Optional[str] = "",
    ) -> Optional[str]:
        """Generate domain-specific user flows"""

        # Analyze endpoints to determine domain
        domain_analysis = self._analyze_api_domain(endpoints, api_info)
        try:
            template = self.jinja_env.get_template("domain.j2")
            # Render enhanced content
            prompt = template.render(
                domain_analysis=domain_analysis,
                custom_requirement=custom_requirement,
                endpoints=self._format_endpoints_for_prompt(endpoints),
            )

            enhanced_content = await self._call_ai_service(prompt)
            if enhanced_content:
                return enhanced_content
        except Exception as e:
            logger.warning(f"Domain flows generation failed: {e}")

        return ""

    def get_files_by_key(
        self, directory_files: List[Dict[str, Any]], target_key: str
    ) -> List[Dict[str, Any]]:
        """Return directory items that contain the specified key"""
        return [items for items in directory_files if target_key in items]

    async def _enhance_workflows(
        self,
        base_content: str,
        test_data_content: str,
        base_workflow: str,
        grouped_enpoints: Dict[str, List[Endpoint]],
        auth_endpoints: List[Endpoint],
        db_type: str = "",
        template_path: str = workflow_jinja_path,
    ) -> Optional[str]:
        try:
            template = self.jinja_env.get_template(template_path)

            # Format auth_endpoints as human-readable text to prevent
            # AI from copying Python object repr into generated code (Bug #2 fix)
            formatted_auth_endpoints = format_auth_endpoints_for_prompt(auth_endpoints)

            # Render enhanced content
            prompt = template.render(
                grouped_enpoints=grouped_enpoints,
                test_data_content=test_data_content,
                base_workflow=base_workflow,
                auth_endpoints=formatted_auth_endpoints,
                base_content=base_content,
                db_type=db_type,
            )

            enhanced_content = await self._call_ai_service(prompt)
            return enhanced_content
        except Exception as e:
            logger.warning(f"Workflow enhancement failed: {e}")

        return ""

    async def enhance_test_data_file(
        self,
        base_content: str,
        endpoints: List[Endpoint],
        db_type: str = "",
        data_provider: str = "",
        db_config: str = "",
        data_provider_path: str = "",
        constraints: str = "",
    ) -> Optional[str]:
        """Enhance test data generation with domain knowledge"""

        # Extract schema information
        schemas_info = self._extract_schema_patterns(endpoints)

        try:
            template = self.jinja_env.get_template("test_data.j2")

            # Prepare context for template
            context = {
                "base_content": base_content,
                "schemas_info": schemas_info,
                "endpoints": endpoints,
                "db_type": db_type,
                "data_provider_content": data_provider,
                "db_config": db_config,
                "data_provider_path": data_provider_path,
                "constraints": constraints,  # Pass constraints to template
            }

            # Render enhanced content
            prompt = template.render(**context)
            enhanced_content = await self._call_ai_service(prompt)
            if enhanced_content and self._validate_python_code(enhanced_content):
                return enhanced_content
        except Exception as e:
            logger.warning(f"Test data enhancement failed: {e}")

        return ""

    async def _enhance_validation(
        self, base_content: str, endpoints: List[Endpoint], constraints: str = ""
    ) -> Optional[str]:
        """Enhance response validation with endpoint-specific checks"""

        validation_patterns = self._extract_validation_patterns(endpoints)
        try:
            template = self.jinja_env.get_template("validation.j2")

            # Render enhanced content
            prompt = template.render(
                base_content=base_content,
                validation_patterns=validation_patterns,
                constraints=constraints,  # Pass constraints to template
            )
            enhanced_content = await self._call_ai_service(prompt)
            if enhanced_content:
                return enhanced_content
        except Exception as e:
            logger.warning(f"Validation enhancement failed: {e}")

        return ""

    def _build_messages(self, prompt: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert Python developer specializing in Locust load testing. "
                    "Follow the output format instructions EXACTLY as specified in the prompt. "
                    "CRITICAL RULES:\n"
                    "1. Return ONLY what is requested - no extra code, no explanations\n"
                    "2. Use the exact tag format specified (e.g., <new_methods> or <code>)\n"
                    "3. DO NOT include import statements unless explicitly requested\n"
                    "4. DO NOT include class definitions unless explicitly requested\n"
                    "5. DO NOT add code that already exists in the provided context\n"
                    "6. Keep output focused and minimal - quality over quantity"
                ),
            },
            {"role": "user", "content": prompt},
        ]

    async def _make_api_call(self, messages: list[dict]) -> Optional[str]:
        """Make API call - ONE job"""
        async with self._api_semaphore:
            api_call = self.ai_client.chat.completions.create(
                model=self.ai_config.model,
                messages=messages,
                max_tokens=self.ai_config.max_tokens,
                temperature=self.ai_config.temperature,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
            )

            # Wait for the API call with timeout
            response = await asyncio.wait_for(
                api_call,
                timeout=self.ai_config.timeout,
            )
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content.strip()
                # Clean up the response
                content = self._clean_ai_response(
                    self.extract_code_from_response(content)
                )
                return content

        return None

    async def _call_ai_service(self, prompt: str) -> Optional[str]:
        """Call AI service with retry logic and validation"""
        messages = self._build_messages(prompt)

        for attempt in range(self.MAX_RETRIES):  # Retry logic
            try:
                content = await self._make_api_call(messages)
                if content:
                    return content

            except asyncio.TimeoutError:
                logger.warning(f"AI service timeout on attempt {attempt + 1}")

            except Exception as e:
                classification = self._classify_error(e, attempt)  # Helper 3

                if not classification.is_retryable:
                    return ""

                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(classification.backoff_seconds)

                    continue

            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)

        return ""

    def extract_code_from_response(self, response_text: str) -> str:
        """
        Extract code from AI response, supporting multiple tag formats.

        Priority:
        1. <new_methods> tags (for methods-only output)
        2. <code> tags (for full code output)
        3. Fall back to full response if no tags found
        """
        # First try <new_methods> tags (preferred for enhancement prompts)
        new_methods_pattern = r"<new_methods>(.*?)</new_methods>"
        new_methods_matches = re.findall(new_methods_pattern, response_text, re.DOTALL)

        if new_methods_matches:
            content = max(new_methods_matches, key=len).strip()
            if content and len(content) > 10:
                logger.debug(f"Extracted {len(content)} chars from <new_methods> tags")
                return content

        # Then try <code> tags
        code_pattern = r"<code>(.*?)</code>"
        code_matches = re.findall(code_pattern, response_text, re.DOTALL)

        if code_matches:
            content = max(code_matches, key=len).strip()
            if content and len(content) > 10:
                logger.debug(f"Extracted {len(content)} chars from <code> tags")
                return content

        # No valid tags found
        # This happens when the LLM returns plain prose or another format
        # instead of the expected tagged blocks from our prompt templates.
        # We fall back to the entire response so downstream merging can still
        # attempt to salvage useful code fragments.
        logger.debug("No valid <new_methods> or <code> tags found, using full response")
        return response_text.strip()

    def _clean_ai_response(self, content: str) -> str:
        """
        Clean and validate AI response.

        With the new methods-only prompt format, we expect cleaner output.
        This function:
        - Removes markdown code blocks
        - Strips obvious non-code text
        - Removes unwanted class definitions that snuck through
        """
        if not content:
            return ""

        # Remove markdown code blocks if present
        if content.startswith("```python"):
            content = content[9:]
        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        # Remove any explanatory text before code starts
        lines = content.split("\n")
        start_idx = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Look for start of actual Python code
            if stripped.startswith(("def ", "class ", "#", '"""', "'''", "@")):
                start_idx = i
                break

        # Remove trailing non-code text
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line:
                # Stop if we find actual code
                if line.startswith(("return", "pass", "raise", ")", "]", "}")) or \
                   "=" in line or "(" in line or line.startswith("#"):
                    end_idx = i + 1
                    break
                # Skip obvious non-code endings
                if line.lower().startswith(("note:", "this ", "the ", "---")):
                    continue
                end_idx = i + 1
                break

        cleaned = "\n".join(lines[start_idx:end_idx])

        # Remove unwanted class definitions that AI might have included despite instructions
        cleaned = self._remove_unwanted_class_definitions(cleaned)

        return cleaned

    def _remove_unwanted_class_definitions(self, content: str) -> str:
        """
        Remove class definitions that shouldn't be in methods-only output.

        When using methods-only prompts, the AI shouldn't return class definitions.
        But if it does, this removes them while preserving methods.
        """
        if "class " not in content:
            return content

        # Classes that should NEVER appear in enhanced output (they exist elsewhere)
        forbidden_classes = {
            "TestDataGenerator",
            "LoadTestConfig",
            "ResponseValidator",
            "RequestLogger",
            "PerformanceMonitor",
            "DataManager",
        }

        lines = content.split("\n")
        result_lines = []
        skip_until_dedent = False
        class_indent = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check if this is a class definition we should remove
            if stripped.startswith("class "):
                class_match = re.match(r"class\s+(\w+)", stripped)
                if class_match:
                    class_name = class_match.group(1)
                    if class_name in forbidden_classes:
                        # Skip this entire class
                        skip_until_dedent = True
                        class_indent = len(line) - len(line.lstrip())
                        logger.info(f"Removing unwanted class '{class_name}' from AI output")
                        i += 1
                        continue

            # If we're skipping a class, check for end of class
            if skip_until_dedent:
                if stripped:
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= class_indent:
                        # We've exited the class
                        skip_until_dedent = False
                        # Check if this new line is also a forbidden class
                        if stripped.startswith("class "):
                            class_match = re.match(r"class\s+(\w+)", stripped)
                            if class_match and class_match.group(1) in forbidden_classes:
                                i += 1
                                continue
                        result_lines.append(line)
                i += 1
                continue

            result_lines.append(line)
            i += 1

        return "\n".join(result_lines)

    def _analyze_api_domain(
        self, endpoints: List[Endpoint], api_info: Dict[str, Any]
    ) -> str:
        """Analyze API to determine domain and patterns"""
        analysis = []

        # API info analysis
        analysis.append(f"API Title: {api_info.get('title', 'Unknown')}")
        analysis.append(f"Description: {api_info.get('description', 'No description')}")

        # Endpoint analysis
        methods = [ep.method for ep in endpoints]
        paths = [ep.path for ep in endpoints]

        analysis.append(f"Total Endpoints: {len(endpoints)}")
        analysis.append(f"HTTP Methods: {', '.join(set(methods))}")
        analysis.append(f"Common Path Patterns: {self._extract_path_patterns(paths)}")

        # Resource analysis
        resources = self._extract_resources_from_paths(paths)
        analysis.append(f"Main Resources: {', '.join(resources[:5])}")

        return "\n".join(analysis)

    def _format_endpoints_for_prompt(self, endpoints: List[Endpoint]) -> str:
        """Format endpoints for AI prompt"""
        formatted = []
        for ep in endpoints:
            params = f"({len(ep.parameters)} params)" if ep.parameters else ""
            body = "(with body)" if ep.request_body else ""
            formatted.append(
                f"- {ep.method} {ep.path} {params} {body} - {ep.summary or 'No summary'}"
            )

        return "\n".join(formatted)

    def _extract_schema_patterns(self, endpoints: List[Endpoint]) -> str:
        """Extract common schema patterns from endpoints"""
        patterns = []

        for ep in endpoints:
            if ep.request_body and ep.request_body.schema:
                schema = ep.request_body.schema
                if schema.get("properties"):
                    fields = list(schema["properties"].keys())
                    patterns.append(f"{ep.path} ({ep.method}): {', '.join(fields[:5])}")

        return "\n".join(patterns[:10])  # Limit for token efficiency

    def _extract_validation_patterns(self, endpoints: List[Endpoint]) -> str:
        """Extract validation patterns needed for endpoints"""
        patterns = []

        for ep in endpoints:
            for response in ep.responses:
                if response.status_code.startswith("2"):  # Success responses
                    pattern = f"{ep.method} {ep.path} -> {response.status_code}"
                    if response.schema:
                        pattern += " (schema validation needed)"
                    patterns.append(pattern)

        return "\n".join(patterns[:10])

    def _analyze_performance_patterns(self, endpoints: List[Endpoint]) -> str:
        """Analyze endpoints for performance testing patterns"""
        analysis = []

        # Categorize endpoints by performance characteristics
        read_heavy = [ep for ep in endpoints if ep.method == "GET"]
        write_heavy = [ep for ep in endpoints if ep.method in ["POST", "PUT", "PATCH"]]
        bulk_candidates = [
            ep
            for ep in endpoints
            if "bulk" in ep.path.lower() or "batch" in ep.path.lower()
        ]

        analysis.append(
            f"Read-heavy endpoints: {len(read_heavy)} (good for load testing)"
        )
        analysis.append(
            f"Write-heavy endpoints: {len(write_heavy)} (good for stress testing)"
        )
        analysis.append(
            f"Bulk operation endpoints: {len(bulk_candidates)} (good for volume testing)"
        )

        # Identify endpoints that might be resource intensive
        complex_endpoints = [
            ep
            for ep in endpoints
            if ep.request_body
            and ep.request_body.schema
            and len(ep.request_body.schema.get("properties", {})) > 5
        ]
        analysis.append(
            f"Complex endpoints: {len(complex_endpoints)} (monitor for performance)"
        )

        return "\n".join(analysis)

    def _extract_path_patterns(self, paths: List[str]) -> str:
        """Extract common patterns from API paths"""
        patterns = set()
        for path in paths:
            # Extract patterns like /api/v1/{resource}
            parts = path.split("/")
            if len(parts) > 2:
                pattern = "/".join(parts[:3])
                if "{" in pattern:
                    pattern = (
                        pattern.replace("{id}", "{id}")
                        .replace("{", "{")
                        .replace("}", "}")
                    )
                patterns.add(pattern)

        return ", ".join(list(patterns)[:5])

    def _extract_resources_from_paths(self, paths: List[str]) -> List[str]:
        """Extract resource names from API paths"""
        resources = set()
        for path in paths:
            parts = [p for p in path.split("/") if p and not p.startswith("{")]
            for part in parts:
                if len(part) > 2 and part.isalpha():  # Likely a resource name
                    resources.add(part)

        return sorted(resources)

    async def _create_test_files_safely(
        self,
        test_files: Dict[str, str],
        output_path: Path,
        max_file_size: int = 1024 * 1024,
    ) -> List[dict]:
        """Create test files safely with reduced complexity"""

        if not test_files:
            return []

        # Setup
        config = FileCreationConfig()
        config.MAX_FILE_SIZE = max_file_size
        creator = SafeFileCreator(config)
        temp_dir = output_path / f"temp_{uuid.uuid4().hex[:8]}"

        try:
            return await self._process_file_creation(
                creator, test_files, output_path, temp_dir
            )
        finally:
            await self._cleanup_temp_directory(temp_dir)

    async def _process_file_creation(
        self,
        creator: SafeFileCreator,
        test_files: Dict[str, str],
        output_path: Path,
        temp_dir: Path,
    ) -> List[dict]:
        """Process the file creation workflow"""

        # Ensure directories exist
        output_path.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Prepare files in temp directory
        prepared_files = await self._prepare_files_in_temp(
            creator, test_files, temp_dir
        )

        if not prepared_files:
            return []

        # Move files atomically to final location
        return await creator.move_files_atomically(prepared_files, output_path)

    async def _prepare_files_in_temp(
        self, creator: SafeFileCreator, test_files: Dict[str, str], temp_dir: Path
    ) -> List[dict]:
        """Prepare all files in temporary directory"""

        prepared_files = []

        for filename, content in test_files.items():
            file_result = await self._prepare_single_file(
                creator, filename, content, temp_dir
            )
            if file_result:
                prepared_files.append(file_result)

        return prepared_files

    async def _prepare_single_file(
        self, creator: SafeFileCreator, filename: str, content: str, temp_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Prepare a single file, return None if failed"""

        try:
            # Validate file
            is_valid, clean_filename, processed_content = creator.validate_file(
                filename, content
            )
            if not is_valid:
                return None

            # Create temp file
            file_info = await creator.create_temp_file(
                clean_filename, processed_content, temp_dir
            )
            logger.info(f"Prepared: {clean_filename} ({len(processed_content)} chars)")
            return file_info

        except Exception as e:
            logger.error(f"Failed to prepare file {filename}: {e}")
            return None

    async def _cleanup_temp_directory(self, temp_dir: Path) -> None:
        """Clean up temporary directory"""
        if temp_dir.exists():
            try:
                await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")

    def _validate_python_code(self, content: str) -> bool:
        """Validate Python code syntax"""
        try:
            compile(content, "<string>", "exec")
            return True
        except SyntaxError:
            return False

    def _validate_critical_elements(
        self, filename: str, enhanced_content: str, original_content: str
    ) -> Tuple[bool, str, List[str]]:
        """
        Validate that AI-enhanced content preserves critical classes and functions.

        Args:
            filename: Name of the file being validated
            enhanced_content: The AI-enhanced content
            original_content: The original template-generated content

        Returns:
            Tuple of (is_valid, content_to_use, list_of_missing_elements)
        """
        missing_elements = []

        # Check critical classes
        if filename in CRITICAL_CLASSES:
            for class_name in CRITICAL_CLASSES[filename]:
                # Look for class definition pattern
                class_pattern = rf"class\s+{class_name}\s*[:\(]"
                if not re.search(class_pattern, enhanced_content):
                    missing_elements.append(f"class {class_name}")
                    logger.warning(
                        f"AI corrupted {filename}: missing critical class '{class_name}'"
                    )

        # Check critical functions (only if they exist in original)
        if filename in CRITICAL_FUNCTIONS:
            for func_name in CRITICAL_FUNCTIONS[filename]:
                func_pattern = rf"def\s+{func_name}\s*\("
                # Only check if the function exists in original
                if re.search(func_pattern, original_content):
                    if not re.search(func_pattern, enhanced_content):
                        missing_elements.append(f"def {func_name}")
                        logger.warning(
                            f"AI corrupted {filename}: missing critical function '{func_name}'"
                        )

        if missing_elements:
            logger.error(
                f"AI enhancement corrupted {filename}, missing: {missing_elements}. "
                f"Falling back to original template code."
            )
            return False, original_content, missing_elements

        # Additional validation: enhanced content shouldn't be dramatically smaller
        original_lines = len(original_content.strip().split("\n"))
        enhanced_lines = len(enhanced_content.strip().split("\n"))

        # If enhanced is less than 30% of original size, it's likely corrupted
        if original_lines > 50 and enhanced_lines < original_lines * 0.3:
            logger.error(
                f"AI enhancement drastically reduced {filename} "
                f"({original_lines} -> {enhanced_lines} lines). "
                f"Falling back to original template code."
            )
            return False, original_content, ["content_too_small"]

        return True, enhanced_content, []

    def _safe_enhance_file(
        self,
        filename: str,
        enhanced_content: Optional[str],
        original_content: str,
    ) -> str:
        """
        Safely apply AI enhancement using additive-only merging.

        This method uses a fundamentally safer approach:
        1. ALWAYS keep the original code intact as the base
        2. Extract ONLY new methods from AI output
        3. Merge new methods into original (never replace)

        Args:
            filename: Name of the file
            enhanced_content: AI-generated content (may be corrupted)
            original_content: Original template-generated content (ALWAYS preserved)

        Returns:
            Original code with any valid AI additions safely merged
        """
        if not enhanced_content or not enhanced_content.strip():
            logger.info(f"No AI content for {filename}, using original")
            return original_content

        # Determine target class for this file
        target_class = None
        if filename == test_data_file_path:
            target_class = "TestDataGenerator"
        elif filename == "utils.py":
            target_class = "ResponseValidator"  # Primary class to extend

        # Use SafeCodeMerger to safely combine original + AI additions
        merged_content = SafeCodeMerger.safe_merge(
            original_code=original_content,
            ai_additions=enhanced_content,
            target_class=target_class
        )

        # Final validation: ensure critical elements still exist
        is_valid, final_content, missing = self._validate_critical_elements(
            filename, merged_content, original_content
        )

        if not is_valid:
            # This should never happen with SafeCodeMerger, but just in case
            logger.error(f"SafeCodeMerger produced invalid output for {filename}, using original")
            return original_content

        # Verify syntax of merged content
        if not self._validate_python_code(final_content):
            logger.error(f"Merged content has syntax errors for {filename}, using original")
            return original_content

        return final_content
