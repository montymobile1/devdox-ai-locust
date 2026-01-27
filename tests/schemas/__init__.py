"""
Utils Package

Provides utility modules for the devdox_ai_locust package:
- llm_client: LLM API calls with rate limiting
- code_processor: Code extraction, fixing, and validation
- schema_utils: OpenAPI schema parsing utilities
- code_validator: Semantic code validation
- scenario_generator: Test scenario generation
"""

from devdox_ai_locust.utils.llm_client import (
    AIServiceError,
    LLMClient,
    RateLimitInfo,
    TimeEstimate,
)
from devdox_ai_locust.utils.code_processor import CodeProcessor
from devdox_ai_locust.utils.schema_utils import (
    escape_for_python_string,
    escape_for_raw_string,
    extract_all_properties,
    get_field_constraints,
    get_schema_type,
    is_required_field,
    resolve_ref_in_union,
    unwrap_nullable_schema,
)

__all__ = [
    # LLM Client
    "AIServiceError",
    "LLMClient",
    "RateLimitInfo",
    "TimeEstimate",
    # Code Processor
    "CodeProcessor",
    # Schema Utils
    "escape_for_python_string",
    "escape_for_raw_string",
    "extract_all_properties",
    "get_field_constraints",
    "get_schema_type",
    "is_required_field",
    "resolve_ref_in_union",
    "unwrap_nullable_schema",
]
# Test schemas package
