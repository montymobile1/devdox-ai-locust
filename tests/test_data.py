"""
Test Data Generator for Locust Performance Tests

Provides realistic test data generation for API endpoints.
"""

import random
import string
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from faker import Faker
import json

fake = Faker()


class TestDataGenerator:
    """Generates realistic test data for API testing"""

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
            Faker.seed(seed)

        self.generated_ids = set()
        self.user_sessions = {}
        self.cached_data = {}

    def generate_json_data(
        self, schema: Dict[str, Any], required_only: bool = False
    ) -> Dict[str, Any]:
        """
        Generate realistic JSON data based on JSON Schema

        Args:
            schema: JSON Schema dictionary
            required_only: If True, only generate required fields

        Returns:
            Dictionary with generated test data
        """
        if not isinstance(schema, dict):
            return {}

        schema_type = schema.get("type", "object")

        if schema_type == "object":
            return self._generate_object_data(schema, required_only)
        elif schema_type == "array":
            return self._generate_array_data(schema)
        elif schema_type == "string":
            return self._generate_string_value(schema)
        elif schema_type == "integer":
            return self._generate_integer_value(schema)
        elif schema_type == "number":
            return self._generate_number_value(schema)
        elif schema_type == "boolean":
            return self._generate_boolean_value(schema)
        else:
            return None

    def _generate_object_data(
        self, schema: Dict[str, Any], required_only: bool = False
    ) -> Dict[str, Any]:
        """Generate object data from schema properties"""
        result = {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for prop_name, prop_schema in properties.items():
            if required_only and prop_name not in required:
                continue

            if "$ref" in prop_schema:
                result[prop_name] = self._handle_reference(prop_schema["$ref"])

        return result

    def _generate_array_data(self, schema: Dict[str, Any]) -> List[Any]:
        """Generate array data from schema"""
        items_schema = schema.get("items", {})
        min_items = schema.get("minItems", 1)
        max_items = schema.get("maxItems", 3)

        array_length = random.randint(min_items, max_items)
        result = []

        for _ in range(array_length):
            if "$ref" in items_schema:
                result.append(self._handle_reference(items_schema["$ref"]))
            else:
                result.append(self.generate_json_data(items_schema))

        return result

    def _generate_integer_value(self, schema: Dict[str, Any]) -> int:
        """Generate integer value based on schema constraints"""
        minimum = schema.get("minimum", 0)
        maximum = schema.get("maximum", 1000)
        multiple_of = schema.get("multipleOf")

        value = random.randint(minimum, maximum)

        if multiple_of:
            value = (value // multiple_of) * multiple_of

        return value

    def _generate_number_value(self, schema: Dict[str, Any]) -> float:
        """Generate number/float value based on schema constraints"""
        minimum = schema.get("minimum", 0.0)
        maximum = schema.get("maximum", 1000.0)
        multiple_of = schema.get("multipleOf")

        value = random.uniform(minimum, maximum)

        if multiple_of:
            value = round((value / multiple_of)) * multiple_of
        else:
            value = round(value, 2)

        return value

    def _generate_boolean_value(self, schema: Dict[str, Any]) -> bool:
        """Generate boolean value"""
        return random.choice([True, False])

    def _handle_reference(self, ref: str) -> Any:
        """Handle $ref references in schema"""
        ref_name = ref.split("/")[-1]
        ref_lower = ref_name.lower()

        if "hosting" in ref_lower or "provider" in ref_lower:
            return random.choice(["github", "gitlab", "bitbucket"])
        elif "role" in ref_lower:
            return random.choice(["admin", "user", "viewer"])
        elif "status" in ref_lower:
            return random.choice(["active", "inactive", "pending"])
        elif "type" in ref_lower:
            return random.choice(["primary", "secondary", "tertiary"])
        else:
            return f"{ref_name.lower()}_value_{random.randint(1, 100)}"

    def _generate_by_name_pattern(self, prop_name: str) -> str:
        """Generate value based on property name patterns as fallback"""
        prop_name_lower = prop_name.lower()

        if "email" in prop_name_lower:
            return fake.email()
        elif any(keyword in prop_name_lower for keyword in ["name", "label"]):
            return fake.catch_phrase()
        elif any(keyword in prop_name_lower for keyword in ["token", "key"]):
            return "".join(random.choices(string.ascii_letters + string.digits, k=32))
        elif "id" in prop_name_lower:
            return str(uuid.uuid4())
        elif "url" in prop_name_lower:
            return fake.url()
        else:
            return self.random_string(10)

    def _generate_string_value(
        self, schema: Dict[str, Any], prop_name: str = ""
    ) -> str:
        """Generate string value based on schema constraints and property name"""
        min_length = schema.get("minLength", 1)
        max_length = schema.get("maxLength", 50)
        enum_values = schema.get("enum")
        prop_name_lower = prop_name.lower()

        if enum_values:
            return random.choice(enum_values)

        if any(keyword in prop_name_lower for keyword in ["email", "mail"]):
            return fake.email()

        if any(keyword in prop_name_lower for keyword in ["name", "label", "title"]):
            if "first" in prop_name_lower:
                return fake.first_name()
            elif "last" in prop_name_lower:
                return fake.last_name()
            elif "company" in prop_name_lower:
                return fake.company()
            else:
                return fake.catch_phrase()[:max_length]

        if any(
            keyword in prop_name_lower
            for keyword in ["token", "key", "secret", "password"]
        ):
            token_length = min(max_length, 32)
            return "".join(
                random.choices(string.ascii_letters + string.digits, k=token_length)
            )

        if any(keyword in prop_name_lower for keyword in ["url", "uri", "endpoint"]):
            return fake.url()

        if "phone" in prop_name_lower:
            return fake.phone_number()

        if "address" in prop_name_lower:
            return fake.address().replace("\\n", ", ")

        if any(
            keyword in prop_name_lower
            for keyword in ["description", "comment", "note", "text"]
        ):
            return fake.text(max_nb_chars=max_length)

        if any(keyword in prop_name_lower for keyword in ["id", "uuid"]):
            return str(uuid.uuid4())

        actual_length = min(max_length, max(min_length, random.randint(5, 20)))
        return "".join(
            random.choices(string.ascii_letters + string.digits, k=actual_length)
        )

    def generate_string(
        self, length: int = 10, pattern: str = None, default: str = None
    ) -> str:
        if default:
            return default

        if pattern:
            if "email" in pattern.lower():
                return fake.email()
            elif "name" in pattern.lower():
                return fake.name()
            elif "phone" in pattern.lower():
                return fake.phone_number()
            elif "address" in pattern.lower():
                return fake.address()
            elif "url" in pattern.lower():
                return fake.url()
            elif "uuid" in pattern.lower():
                return str(uuid.uuid4())

        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    def generate_integer(
        self, min_val: int = 1, max_val: int = 1000, default: int = None
    ) -> int:
        if default is not None:
            return default
        return random.randint(min_val, max_val)

    def generate_float(
        self, min_val: float = 0.0, max_val: float = 1000.0, default: float = None
    ) -> float:
        if default is not None:
            return default
        return round(random.uniform(min_val, max_val), 2)

    def generate_boolean(self, default: bool = None) -> bool:
        if default is not None:
            return default
        return random.choice([True, False])

    def generate_id(self, prefix: str = "", id_type: str = "uuid") -> str:
        if id_type == "uuid":
            new_id = str(uuid.uuid4())
        elif id_type == "incremental":
            new_id = f"{len(self.generated_ids) + 1}"
        else:
            new_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))

        if prefix:
            new_id = f"{prefix}_{new_id}"

        self.generated_ids.add(new_id)
        return new_id

    def generate_email(self) -> str:
        return fake.email()

    def random_string(self, length: int = 10) -> str:
        return "".join(random.choices(string.ascii_letters, k=length))

    def random_int(self, min_val: int = 0, max_val: int = 1000) -> int:
        return random.randint(min_val, max_val)

    def random_float(self, min_val: float = 0.0, max_val: float = 1000.0) -> float:
        return random.uniform(min_val, max_val)

    def random_bool(self) -> bool:
        return random.choice([True, False])

    def random_uuid(self) -> str:
        return str(uuid.uuid4())

    def random_date(self, start_days_ago: int = 365, end_days_ahead: int = 365) -> str:
        start_date = datetime.now() - timedelta(days=start_days_ago)
        end_date = datetime.now() + timedelta(days=end_days_ahead)
        return fake.date_between(start_date=start_date, end_date=end_date).isoformat()

    # New Methods
    def generate_affiliate_data(self) -> Dict[str, Any]:
        return {
            "id": self.generate_id(prefix="affiliate"),
            "name": fake.company(),
            "email": fake.email(),
        }

    def generate_user_credentials(self) -> Dict[str, Any]:
        return {
            "username": fake.user_name(),
            "password": self.generate_string(length=12),
            "email": fake.email(),
        }

    def generate_product_data(self) -> Dict[str, Any]:
        return {
            "id": self.generate_id(prefix="product"),
            "name": fake.product_name(),
            "price": self.generate_float(min_val=1.0, max_val=100.0),
        }

    def generate_realistic_id(self, entity_type: str) -> str:
        if entity_type == "user":
            return self.generate_id(prefix="user")
        elif entity_type == "product":
            return self.generate_id(prefix="product")
        elif entity_type == "order":
            return self.generate_id(prefix="order")
        else:
            return self.generate_id()

    def generate_affiliate_id(self) -> str:
        return self.generate_id(prefix="affiliate")

    def generate_partner_id(self) -> str:
        return self.generate_id(prefix="partner")

    def get_payload_template(self, endpoint_path: str, method: str) -> Dict[str, Any]:
        if endpoint_path == "/pet" and method == "POST":
            return {
                "id": self.generate_id(prefix="pet"),
                "name": fake.pet_name(),
                "category": fake.category(),
                "photoUrls": [fake.url()],
                "tags": [fake.tag()],
            }
        elif endpoint_path == "/store/order" and method == "POST":
            return {
                "id": self.generate_id(prefix="order"),
                "petId": self.generate_id(prefix="pet"),
                "quantity": self.generate_integer(min_val=1, max_val=10),
                "shipDate": self.random_date(),
                "status": fake.status(),
            }
        elif endpoint_path == "/user" and method == "POST":
            return {
                "id": self.generate_id(prefix="user"),
                "username": fake.user_name(),
                "firstName": fake.first_name(),
                "lastName": fake.last_name(),
                "email": fake.email(),
            }
        else:
            return {}

    def generate_login_payload(self) -> Dict[str, Any]:
        return {
            "username": fake.user_name(),
            "password": self.generate_string(length=12),
        }

    def generate_registration_payload(self) -> Dict[str, Any]:
        return {
            "username": fake.user_name(),
            "email": fake.email(),
            "password": self.generate_string(length=12),
        }

    def create_user_session(self, user_id: str) -> str:
        session_id = self.generate_id(prefix="session")
        self.user_sessions[session_id] = user_id
        return session_id

    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        user_id = self.user_sessions.get(session_id)
        if user_id:
            return {"user_id": user_id}
        else:
            return {}

    def link_related_entities(self, parent_id: str, child_type: str) -> str:
        child_id = self.generate_id(prefix=child_type)
        # Link child to parent
        return child_id

    def cache_generated_data(self, key: str, data: Any) -> None:
        self.cached_data[key] = data

    def get_cached_data(self, key: str) -> Any:
        return self.cached_data.get(key)

    def get_or_create_entity(self, entity_type: str, **kwargs) -> Any:
        cached_entity = self.get_cached_data(entity_type)
        if cached_entity:
            return cached_entity
        else:
            if entity_type == "user":
                entity = self.generate_user_credentials()
            elif entity_type == "product":
                entity = self.generate_product_data()
            else:
                entity = {}
            self.cache_generated_data(entity_type, entity)
            return entity

    def generate_api_key_data(self) -> Dict[str, Any]:
        return {
            "api_key": self.generate_string(length=32),
            "api_secret": self.generate_string(length=32),
        }

    def generate_webhook_payload(self) -> Dict[str, Any]:
        return {
            "event": fake.event(),
            "data": fake.data(),
        }

    def generate_pagination_data(self) -> Dict[str, Any]:
        return {
            "page": self.generate_integer(min_val=1, max_val=10),
            "size": self.generate_integer(min_val=1, max_val=100),
        }

    def generate_filter_data(self) -> Dict[str, Any]:
        return {
            "field": fake.field(),
            "value": fake.value(),
        }

    def generate_error_scenarios(self) -> List[Dict[str, Any]]:
        scenarios = []
        scenarios.append({"error": "invalid_request", "message": "Invalid request"})
        scenarios.append({"error": "unauthorized", "message": "Unauthorized"})
        scenarios.append({"error": "not_found", "message": "Not found"})
        return scenarios

    def validate_generated_data(self, data: Any, schema: Dict[str, Any]) -> bool:
        # Basic validation
        if not isinstance(data, dict):
            return False
        for field, field_schema in schema.items():
            if field not in data:
                return False
            if field_schema["type"] == "string" and not isinstance(data[field], str):
                return False
            if field_schema["type"] == "integer" and not isinstance(data[field], int):
                return False
            if field_schema["type"] == "boolean" and not isinstance(data[field], bool):
                return False
        return True


# Global instance for easy access
test_data_generator = TestDataGenerator()