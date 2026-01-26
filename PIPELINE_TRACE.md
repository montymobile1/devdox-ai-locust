# DevDox AI Locust - Complete Pipeline Trace

This document traces the full code execution path from the moment an OpenAPI specification is provided as input, through every processing step, to the final generated output files.

---

## Table of Contents

1. [CLI Entry Point](#1-cli-entry-point)
2. [Schema Fetching](#2-schema-fetching)
3. [OpenAPI Parsing](#3-openapi-parsing)
4. [Endpoint Grouping & Configuration](#4-endpoint-grouping--configuration)
5. [ScenarioWorkflowGenerator Initialization](#5-scenarioworkflowgenerator-initialization)
6. [Static Base File Generation](#6-static-base-file-generation)
7. [Pre-LLM Template Generation (Fallbacks)](#7-pre-llm-template-generation-fallbacks)
8. [Per-Endpoint LLM Generation Loop](#8-per-endpoint-llm-generation-loop)
9. [Pre-Computation Phase (Per Scenario)](#9-pre-computation-phase-per-scenario)
10. [Prompt Template Rendering](#10-prompt-template-rendering)
11. [LLM Call & Response Handling](#11-llm-call--response-handling)
12. [Code Extraction & Post-Processing](#12-code-extraction--post-processing)
13. [Syntax Validation](#13-syntax-validation)
14. [Semantic Validation (CodeValidator)](#14-semantic-validation-codevalidator)
15. [Retry Loop with Fix Prompts](#15-retry-loop-with-fix-prompts)
16. [File Writing & Directory Structure](#16-file-writing--directory-structure)
17. [Orchestrator Generation](#17-orchestrator-generation)
18. [__init__.py Generation](#18-initpy-generation)
19. [Base File Writing](#19-base-file-writing)
20. [Static File Templates Detail](#20-static-file-templates-detail)

---

## 1. CLI Entry Point

**File:** `src/devdox_ai_locust/cli.py`

### Entry Function: `generate` (line 798)

The CLI is a Click-based command. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `swagger_url` | (required) | URL or file path to OpenAPI spec |
| `--output` | `"output"` | Output directory |
| `--users` | 10 | Locust simulated users |
| `--spawn-rate` | 2.0 | Users spawned per second |
| `--run-time` | `"5m"` | Test duration |
| `--host` | None (auto-detect) | Target host URL |
| `--auth/--no-auth` | True | Include auth handling |
| `--db-type` | `""` | Database type (`mongo`, `postgresql`, or empty) |
| `--custom-requirement` | None | Custom test generation instructions |
| `--timeout` | 120 | LLM API call timeout (seconds) |
| `--schema-timeout` | 30 | Schema fetch timeout (seconds) |
| `--max-llm-workers` | 1 | Concurrent LLM requests (max 10, fail-fast if exceeded) |
| `--debug` | False | Record all intermediate states |

### Execution Flow:

```
generate() → asyncio.run(_async_generate())
  → _initialize_config()          # Validate API key
  → DebugRecorder(output_dir)     # If --debug
  → _display_configuration()      # Rich panel output
  → _process_api_schema()         # Fetch + parse OpenAPI spec
  → _generate_and_create_tests()  # Main generation pipeline
  → _show_results()               # Display summary
```

---

## 2. Schema Fetching

**File:** `src/devdox_ai_locust/utils/swagger_utils.py`

### Function: `get_api_schema(source_request)`

**Input:** `SwaggerProcessingRequest` with `swagger_url` field.

**Logic:**
```
IF swagger_url starts with "http://" or "https://":
    → _fetch_from_url() using httpx async client
    → Response body as string
ELSE:
    → Read from local filesystem (Path)
    → File contents as string
```

**Output:** Raw schema content as `str` (JSON or YAML).

**Error handling:**
- `asyncio.TimeoutError` → exit with schema-timeout message
- Any other exception → exit with error message

---

## 3. OpenAPI Parsing

**File:** `src/devdox_ai_locust/utils/open_ai_parser.py`

### Step 3a: `parse_schema(schema_content)` (line 125)

```
TRY json.loads(schema_content)
EXCEPT JSONDecodeError:
    → yaml.safe_load(schema_content)

→ _validate_openapi_schema()
    → Check required fields: ["openapi", "info", "paths"]
    → Check version starts with "3."
    → IF missing or wrong version → raise ValueError

→ Store self.components = spec_data.get("components", {})
```

### Step 3b: `parse_endpoints()` (line 178)

Iterates all paths and HTTP methods:

```
FOR each path in spec_data["paths"]:
    path_parameters = path_item.get("parameters", [])  # Path-level params

    FOR each method in [get, post, put, patch, delete, head, options, trace]:
        IF method in path_item:
            operation = path_item[method]

            operation_id = operation.get("operationId")
            IF not operation_id:
                → _generate_operation_id(method, path)
                  # e.g., "get_users_userId"

            endpoint = Endpoint(
                path, method, operation_id,
                summary, description,
                parameters = _extract_parameters(operation, path_parameters),
                request_body = _extract_request_body(operation),
                responses = _extract_responses(operation),
                tags = operation.get("tags", []),
                security = operation.get("security"),
            )
```

### Step 3c: `_extract_parameters()` (line 236)

For each parameter (path-level + operation-level combined):

```
→ _resolve_reference(param)  # Resolve $ref if present
→ Extract: name, location (query/path/header/cookie), required, type
→ Handle array types: type becomes "array[itemType]"
→ Extract constraints: enum, format, pattern, minLength, maxLength, minimum, maximum
```

### Step 3d: `_extract_request_body()` (line 294)

```
IF no requestBody → return None

→ _resolve_reference(request_body_def)  # Resolve top-level $ref
→ Prioritize content types: application/json > x-www-form-urlencoded > multipart/form-data > first available
→ schema = media_type.get("schema", {})
→ _resolve_schema_deep(schema)  # Recursively resolve all nested $refs
→ Return RequestBody(content_type, schema, required, description, examples)
```

### Step 3e: `_extract_responses()` (line 348)

```
FOR each status_code, response_def in operation["responses"]:
    → _resolve_reference(response_def)  # Resolve top-level response $ref
    → Extract content type (prioritize JSON)
    → _resolve_schema_deep(schema)  # Recursively resolve all nested $refs in schema
    → Return Response(status_code, description, content_type, schema, headers)
```

### Step 3f: `_resolve_reference()` (line 397)

Resolves a single `$ref` pointer to its target (shallow, one level):

```
IF obj has no "$ref" → return obj as-is
IF $ref doesn't start with "#/" → log warning, return None (external refs unsupported)

→ Split ref path (e.g., "#/components/schemas/User" → ["components", "schemas", "User"])
→ Navigate spec_data following path parts
→ Return resolved object
```

### Step 3g: `_resolve_schema_deep()` (line 440)

Recursively resolves ALL `$ref` references in a schema tree. Uses ancestry-based
circular reference detection to prevent infinite loops:

```
IF schema has "$ref":
    IF ref already in ancestry set → CIRCULAR: return one level of resolved fields (no further recursion)
    ELSE → resolve ref, add to ancestry, continue resolving the target

THEN recursively resolve:
    → properties (each property schema)
    → additionalProperties
    → items (array item schema)
    → allOf / oneOf / anyOf (each list item)
```

### Step 3h: `get_schema_info()` (line 520+)

Returns dict with:
- `title`: from info.title
- `version`: from info.version
- `description`: from info.description
- `base_url`: from servers[0].url (defaults to "http://localhost")
- `security_schemes`: from components.securitySchemes

---

## 4. Endpoint Grouping & Configuration

**File:** `src/devdox_ai_locust/cli.py`, `_generate_scenario_based_tests()` (line 312)

### Grouping by Tag:

```python
FOR each endpoint in endpoints:
    tag = endpoint.tags[0] IF endpoint.tags ELSE "default"
    grouped_endpoints[tag].append(endpoint)
```

### Auth Endpoint Detection:

```python
auth_endpoints = [ep for ep in endpoints if any(
    keyword in ep.path.lower()
    for keyword in ["auth", "login", "token", "session"]
)]
```

---

## 5. ScenarioWorkflowGenerator Initialization

**File:** `src/devdox_ai_locust/utils/scenario_generator.py` (line 134)

```python
ScenarioWorkflowGenerator(
    prompt_dir = Path(__file__).parent / "prompt",
    ai_client = AsyncTogether(api_key=api_key),
    ai_config = AIEnhancementConfig(timeout=timeout),
    max_concurrency = max_llm_workers,  # CLI --max-llm-workers (default=1, max=10)
    debug_recorder = debug_recorder,
)
```

Initializes:
- `_api_semaphore = asyncio.Semaphore(max_concurrency)`
- `_fallback_registry = FallbackHttpResponseRegistry()`
- `_code_validator = CodeValidator()`
- `prompt_env = Environment(loader=FileSystemLoader(prompt_dir))`

### Scenario Types (3 per endpoint):

| Type | File | Template |
|------|------|----------|
| POSITIVE | `positive_workflow.py` | `workflow_positive.j2` |
| NEGATIVE | `negative_workflow.py` | `workflow_negative.j2` |
| SECURITY | `security_workflow.py` | `workflow_security.j2` |

---

## 6. Static Base File Generation

**File:** `src/devdox_ai_locust/locust_generator.py`

### Function: `generate_from_endpoints()` (line 162)

Called at `cli.py:368`. Generates ALL static files using Jinja2 templates:

```python
generated_files = {
    "locustfile.py":     _generate_main_locustfile(),
    "base_workflow.py":  generate_base_common_file(),
    "test_data.py":      _generate_test_data_file(db_type),
    "config.py":         _generate_config_file(),
    "utils.py":          _generate_utils_file(),
    "custom_flows.py":   _generate_custom_flows_file(),
    "requirements.txt":  _generate_requirements_file(),
    "README.md":         _generate_readme_file(),
    ".env.example":      _generate_env_example(),
}

IF db_type != "":
    generated_files["db_config.py"] = _generate_db_file(db_type, "db_config.py.j2")
    generated_files["data_provider.py"] = _generate_db_file(db_type, "data_provider.py.j2")
```

### Post-processing: `fix_indent()` (line 136)

All generated Python files are formatted with Black:
```
FOR each file in base_files:
    TRY: formatted = black.format_str(content, mode=black.Mode())
    EXCEPT InvalidInput: keep original (not valid Python, e.g., .env)
```

---

## 7. Pre-LLM Template Generation (Fallbacks)

**File:** `src/devdox_ai_locust/cli.py`, `generate_pre_llm_workflow()` (line 426)

For every endpoint + scenario type combination, a simple fallback template is generated:

```python
FOR each endpoint:
    FOR each scenario_type in [positive, negative, security]:
        pre_llm_templates[(id(endpoint), scenario_type)] = generate_pre_llm_workflow()
```

The fallback template contains:
- Basic imports (locust task, BaseWorkflow)
- A class inheriting from BaseWorkflow
- A single task method generated by `_generate_task_method(endpoint)`

**Purpose:** If LLM generation fails for an endpoint, these fallback templates are written instead, ensuring the test suite is always runnable.

---

## 8. Per-Endpoint LLM Generation Loop

**File:** `src/devdox_ai_locust/cli.py`, `process_and_save_endpoint()` (line 495)

All endpoints are processed in parallel via `asyncio.gather()`:

```python
tasks = [process_and_save_endpoint(ep) for ep in endpoints]
await asyncio.gather(*tasks)
```

For each endpoint:

```
1. Determine tag directory name (sanitize_dir_name)
2. Create endpoint directory: workflows/{tag}/{operation_id}/
3. Call scenario_gen.generate_endpoint_workflows()
4. For each generated scenario:
    → Write to: workflows/{tag}/{operation_id}/{scenario_type}_workflow.py
5. Track success/failure counts

ON EXCEPTION:
    → Use pre_llm_templates as fallback
    → Write all 3 fallback files
    → Track as failed endpoint
```

### `generate_endpoint_workflows()` (scenario_generator.py:299)

Generates all 3 scenarios in parallel for one endpoint:

```python
llm_tasks = [
    _generate_llm_scenario(ScenarioType.POSITIVE, ...),
    _generate_llm_scenario(ScenarioType.NEGATIVE, ...),
    _generate_llm_scenario(ScenarioType.SECURITY, ...),
]
llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)
```

**Error handling:**
- If some succeed and some fail → return partial results (successful ones)
- If ALL fail → raise the first error (triggers fallback)

---

## 9. Pre-Computation Phase (Per Scenario)

**File:** `src/devdox_ai_locust/utils/scenario_generator.py`, `_generate_llm_scenario()` (line 620)

Before calling the LLM, several values are pre-computed to reduce LLM hallucination:

### 9a. Status Code Pre-computation

**Method:** `_precompute_scenario_status_codes()` (line 1884)

```
1. Extract all status codes from endpoint.responses (with descriptions)
   → _extract_status_codes_with_descriptions()

2. Filter by scenario type:
   - POSITIVE: only 2xx codes
   - NEGATIVE: only 4xx codes
   - SECURITY: all codes < 500

3. IF filter produces results → use them (spec is source of truth)

4. IF spec has responses but no matching codes:
   - POSITIVE: return [] (signals skip - no success case exists)
   - NEGATIVE/SECURITY: fall through to fallback

5. IF spec has NO responses at all → use FallbackHttpResponseRegistry:
   → _get_fallback_codes(method, scenario_type)
   → Returns method-appropriate default codes
```

**Skip logic for POSITIVE:**
```python
IF scenario_type == POSITIVE AND all_status_codes AND NOT expected_status_codes:
    → Skip positive generation (endpoint has no 2xx responses defined)
    → Return None
```

### 9b. Injection Points Pre-computation (SECURITY only)

**Method:** `_precompute_injection_points()` (line 1970)

```
1. Scan request body for string fields → body_fields list
2. Scan parameters for string query params → query_params list

IF no body_fields AND no query_params:
    → Return None (skip security generation - no valid injection targets)
ELSE:
    → Format as prompt text listing injection targets
```

### 9c. Negative Scenarios Pre-computation (NEGATIVE only)

**Method:** `_precompute_negative_scenarios()` (line 2026)

Scans endpoint schema and determines testable scenarios:

```
IF path_params exist:
    → NON_EXISTENT_ID scenario (integer: 999999999, string: "nonexistent-id-12345")

IF required body fields:
    → MISSING_REQUIRED scenario

IF typed body fields (integer/number/boolean/array):
    → WRONG_TYPE scenario (send string where number expected)

IF enum fields:
    → INVALID_ENUM scenario (send "INVALID_VALUE_XYZ")

IF pattern fields:
    → INVALID_PATTERN scenario (send "!!!invalid!!!")

IF numeric constrained fields:
    → BOUNDARY scenario (send min-1 or max+1)

IF none of above but query params:
    → INVALID_QUERY scenario (send wrong type/very long string)

IF absolutely nothing testable:
    → Return "" (skip negative generation)
```

### 9d. Positive Fields Pre-computation (POSITIVE only)

**Method:** `_precompute_positive_fields()` (line 2177)

Pre-computes exact generation instructions for each request body field:

```
IF no request_body → return ""

STEP 1: allOf merging
    IF schema has allOf:
        → Merge all allOf items' properties together
        → Merge all allOf items' required lists
        → Combine with any direct schema properties

STEP 2: Discriminated Unions (oneOf/anyOf at top level)
    IF schema has oneOf/anyOf AND no direct properties:
        → List all variants with their required fields
        → Instruct LLM to "Pick ONE variant and include ALL its required fields"
        → Return early

STEP 3: For each field in merged properties:
    Determine instruction based on:

    1. field has enum → random.choice([values])  (NO truncation, full enum list)
    2. field has pattern → generate_string(pattern="...")
    3. format == "date" → random_date()
    4. format == "date-time" → datetime.now().isoformat()
    5. format == "email" → generate_email()
    6. format == "uuid" → random_uuid()
    7. format == "uri"/"url" → literal string
    8. format == "ipv4" → "192.168.1.1"
    9. format == "ipv6" → "::1"
    10. format == "hostname" → "test.example.com"
    11. format == "time" → "12:30:00"
    12. type == "string" (no format) → generate_string(length=min(maxLength, 50))
    13. type == "integer" → generate_integer(min_val, max_val, exclusive, multiple_of)
    14. type == "number" → generate_float(min_val, max_val, exclusive)
    15. type == "boolean" → generate_boolean()
    16. type == "object" → _precompute_object_instruction() (recursive, with circular detection)
    17. type == "array" → list comprehension with correct item generator
    18. fallback → generate_string(length=10)
```

**`_precompute_object_instruction()` (recursive):**
```
Generates a Python dict literal string for nested object schemas.
Uses identity-based ancestry tracking (id(schema)) to detect circular references.

FOR each property in object schema:
    → Determine generator (same logic as top-level fields)
    → IF type == "object" with sub-properties → RECURSE
    → IF type == "array" with object items → recurse into items schema
    → IF circular reference detected (same schema id in ancestry) → return "{}"
```

### 9e. Setup Endpoints Discovery

**Method:** `_find_related_create_endpoints()` (line 1496)

```
IF endpoint is POST → skip (doesn't need setup)
IF endpoint has path params OR is not POST:
    → Search for related POST endpoints

Scoring algorithm:
    Factor 1 (highest, 100+): Exact parent path match
        /api/v1/users → POST /api/v1/users creates user_id for /users/{user_id}/posts

    Factor 2 (medium, 20+): Shared path prefix (min 3 segments)
        /api/v1/comprehensive/... namespace matching

    Factor 3 (low, 10): Same tag
        Tiebreaker when paths are similar

→ Sort by score descending
→ Format with full schema details for prompt
```

### 9f. Endpoint Details Stripping (NEGATIVE only)

**Method:** `_strip_success_responses()` (line 1048)

```
FOR negative tests:
    Remove all 2xx response lines from endpoint description
    → Prevents LLM from copying success codes into negative expected_status
```

---

## 10. Prompt Template Rendering

**File:** `src/devdox_ai_locust/prompt/*.j2`

### Template Context Variables:

```python
template_context = {
    "endpoint":                formatted endpoint details (full schema),
    "auth_endpoints":          formatted auth endpoint list,
    "base_workflow":           complete base_workflow.py content,
    "test_data_content":       complete test_data.py content,
    "class_name":              PascalCase class name,
    "operation_id":            sanitized operation ID,
    "method":                  HTTP method,
    "path":                    endpoint path,
    "endpoint_expected_status": pre-computed status codes list,
    "expected_status_info":    formatted status code descriptions,
    "injection_points":        (security only) valid injection targets,
    "negative_scenarios":      (negative only) testable scenarios,
    "positive_fields":         (positive only) field generation instructions,
    "setup_endpoints":         formatted related CREATE endpoints,
    "custom_requirement":      user custom requirements,
    "db_type":                 database type,
}
```

### Template: `workflow_positive.j2`

Key sections:
1. Custom requirements (if provided)
2. Endpoint context (full schema)
3. Setup endpoints (if applicable)
4. Authentication handling (if auth_endpoints)
5. Allowed imports list
6. Test data generator API reference (with complete signatures)
7. MongoDB integration (if db_type == "mongo")
8. HTTP request pattern (`make_request()` only)
9. Expected status codes
10. Output requirements (ASCII only, raw strings for regex)
11. Chain-of-thought instructions (5 steps)
12. Weight guidance table
13. Error handling patterns
14. Constraints section
15. Critical mistakes to avoid
16. Output format specification (`<analysis>` + `<code>` tags)

### Template: `workflow_negative.j2`

Key differences from positive:
- Expected status codes filtered to 4xx only
- Pre-computed negative scenarios with explicit test instructions
- Constraints about NEVER expecting 2xx
- Guidance for missing path params (use invalid value, not empty)

### Template: `workflow_security.j2`

Key differences:
- Pre-computed injection points (body fields + query params)
- Security payload examples (XSS, SQL injection, path traversal)
- Constraint: NEVER inject into path parameters
- Expected status codes include all < 500

---

## 11. LLM Call & Response Handling

**File:** `src/devdox_ai_locust/utils/scenario_generator.py`, `_call_ai_service()` (line 2441)

### System Prompt:
```
"You are an expert Python developer specializing in Locust load testing.
Generate clean, production-ready code for a SINGLE endpoint.
Keep the code focused and concise.
Return code in <code></code> tags. Do not truncate."
```

### API Call:
```python
async with self._api_semaphore:  # Bounded concurrency
    response = await asyncio.wait_for(
        ai_client.chat.completions.create(
            model = ai_config.model,
            messages = [system_prompt, user_prompt],
            max_tokens = ai_config.max_tokens,
            temperature = ai_config.temperature,
        ),
        timeout = ai_config.timeout,
    )
```

### Retry Logic (3 attempts):
```
FOR attempt in [0, 1, 2]:
    TRY:
        → Make API call with semaphore
        → Update rate limit from response headers
        → Return raw response content (stripped)
    EXCEPT TimeoutError:
        → Log, sleep 2^attempt seconds
    EXCEPT any Exception:
        → Log, sleep 2^attempt seconds

IF all 3 attempts fail:
    → raise AIServiceError

NOTE: _call_ai_service returns RAW content. Code extraction (_extract_code)
is performed by the caller (_generate_llm_scenario / orchestrator).
```

### Rate Limit Adaptation:
```python
def _update_concurrency(rpm):
    optimal = min(int(rpm * 0.8 / 20), max_concurrency)
    new_concurrency = max(2, optimal)
    IF changed:
        _api_semaphore = asyncio.Semaphore(new_concurrency)
```

---

## 12. Code Extraction & Post-Processing

**File:** `src/devdox_ai_locust/utils/scenario_generator.py`

### Step 12a: `_extract_code()` (line 2493)

```
1. Strip <analysis>...</analysis> sections (chain-of-thought)
2. Find <code>...</code> tags (case-insensitive, handles attributes)
   IF multiple <code> blocks → take the longest
   IF no closing tag → take everything after opening tag
   IF no <code> tag at all → use entire response
3. Strip markdown code fences (```python, ```)
4. Strip remaining HTML code tags
5. Remove garbage lines:
   - Lines starting with '!'
   - 'DO NOT EDIT' / 'generated by' headers
   - 'Note:' / 'Since ' / 'This endpoint' / 'This is' / 'We '
   - 'STEP N:' / 'Method:' / 'Path:' / 'Required:' / 'Optional:'
```

### Step 12b: `_sanitize_unicode()` (line 2592)

```
FOR each line:
    Keep only characters with ord(c) < 128 (ASCII only)
    → Removes Chinese, Arabic, emoji, etc.
```

### Step 12c: `_fix_class_name()` (line 2559)

```
Expected: {ClassName}{ScenarioType}Workflow (e.g., GetUsersPositiveWorkflow)

Find class definition matching: class XXX(BaseWorkflow):
IF actual name != expected name:
    → regex replace class definition to use expected name
```

### Step 12d: `_fix_bytes_literals()` (line 2609)

```
Find bytes literals: b'...' or b"..."
IF content has non-ASCII characters:
    → Convert to: '...'.encode('utf-8')
```

### Step 12e: `_fix_regex_strings()` (line 2654)

```
Tokenize the code using Python's tokenizer
FOR each STRING token:
    IF not already raw string (r"..."):
    IF not bytes/f-string:
    IF contains problematic escapes (\d, \w, \s, \+, etc.):
        → Convert to raw string: r"..."
```

---

## 13. Syntax Validation

**File:** `src/devdox_ai_locust/utils/scenario_generator.py`, `_validate_python_code()` (line 2849)

```python
TRY:
    compile(content, "<string>", "exec")
EXCEPT SyntaxError as e:
    → return (False, "Line N: error_msg - text")

# Check imports (warnings only, does not fail)
→ _check_imports(content)
    → AST walk for Import/ImportFrom nodes
    → Log warning if module not in ALLOWED_IMPORTS set
    → DOES NOT fail validation

→ return (True, "")
```

**Allowed imports:**
```
random, logging, datetime, time, json, re, uuid, string,
locust, workflows.base_workflow, workflows, base_workflow,
test_data, mongo_data_provider
```

---

## 14. Semantic Validation (CodeValidator)

**File:** `src/devdox_ai_locust/utils/code_validator.py`

After syntax passes, the code is checked for semantic correctness:

```python
semantic_result = code_validator.validate(
    code = content,
    scenario_type = "positive"/"negative"/"security",
    endpoint_path = endpoint.path,
    all_endpoint_paths = [all paths from spec],
    request_body_schema = endpoint.request_body.schema,
)
```

### Check 1: Template Boilerplate (all scenarios)

Matches patterns like:
- `# Check if request succeeded (result is dict or None)`
- `# Success - result contains JSON response data`
- `# Example: item_id = result.get`
- etc.

**Severity:** error

### Check 2: Placeholder Comments (all scenarios)

Matches patterns like:
- `# Add other required ... fields`
- `# TODO: `
- `# Fill in remaining`
- `# Complete this`

**Severity:** error

### Check 3: Empty Path Segments (all scenarios)

Detects `//` in URL strings (excluding `https://`):
```
make_request("PATCH", "/api/items//verify", ...)
                              ^^^ empty segment
```

**Severity:** error

### Check 4: Security Path Injection (security only)

Detects payload variables in URL path f-strings:
```python
make_request("GET", f"/items/{payload}/details", ...)
```

**Severity:** error

### Check 5: Success Codes in Negative (negative only)

Detects 2xx codes in `expected_status` for the endpoint under test:
```python
make_request("PUT", "/items/1", expected_status=[200], ...)  # BAD in negative
```

**Exception:** Setup calls to DIFFERENT endpoints are allowed to use 2xx:
```python
# This is fine - creating test data on a different endpoint
make_request("POST", "/items", expected_status=[201], ...)
```

**Path comparison:** Segment-by-segment comparison via `_paths_match()`. Treats `{param}`
segments as wildcards. Different segment counts = no match (prevents `/items/{id}` from
falsely matching `/items/{id}/details`).

**Severity:** error

### Check 6: Hallucinated Endpoints (all scenarios, if all_endpoint_paths provided)

For every `make_request` call, checks if the URL path exists in the OpenAPI spec:
```
FOR each make_request call:
    Extract URL path (supports f-string paths with {variable} expressions)
    IF not in all_endpoint_paths:
        IF not matching any spec path via _paths_match() segment comparison:
            → violation
```

**Path matching:** Segment-by-segment comparison. `{param}` segments in either path
match any segment. Both paths must have the same number of segments.

**Severity:** error

### Check 7: Schema Compliance (positive only, if request_body_schema provided)

Parses the generated code's AST and finds dict literals assigned to body variables:

**7a. Enum ignored:**
```python
# BAD: Using generate_string() for an enum field
"status": test_data_generator.generate_string()  # Should be random.choice(["active", "inactive"])
```

**7b. Wrong format generator:**
```python
# BAD: Using generate_string() for a date field
"created_at": test_data_generator.generate_string()  # Should be random_date()
```

**7c. Mixed array types:**
```python
# BAD: String array with mixed types
"tags": ["foo", 123, True]  # All elements must be strings
```

**Severity:** error

---

## 15. Retry Loop with Fix Prompts

**File:** `src/devdox_ai_locust/utils/scenario_generator.py` (line 812)

```
max_validation_retries = 2

FOR attempt in [0, 1]:
    IF attempt > 0:
        IF last failure was semantic:
            → Use workflow_semantic_fix.j2 template
        ELSE (syntax error):
            → Use workflow_fix.j2 template

    → Call LLM (_call_ai_service)
    → Extract code
    → Post-process (unicode, class name, bytes, regex)
    → Syntax validation

    IF syntax fails:
        → Save error, continue to next attempt

    IF syntax passes:
        → Semantic validation
        IF semantic passes:
            → Return code (SUCCESS)
        ELSE:
            → Save semantic error, continue to next attempt

IF all attempts exhausted:
    → raise CodeValidationError
```

### Fix Prompt (`workflow_fix.j2`):

Provides:
- The failed code
- The exact error message
- Instructions to fix and return complete code

### Semantic Fix Prompt (`workflow_semantic_fix.j2`):

Provides:
- The failed code
- Semantic violation details
- The correct expected_status codes
- The endpoint path and method
- Instructions to fix ALL violations

---

## 16. File Writing & Directory Structure

**File:** `src/devdox_ai_locust/cli.py`

### Directory Layout:

```
output/
├── locustfile.py              # Main entry point
├── test_data.py               # Test data generators
├── config.py                  # Configuration
├── utils.py                   # Utility functions
├── custom_flows.py            # Custom flow extension point
├── requirements.txt           # Python dependencies
├── README.md                  # Usage instructions
├── .env.example               # Environment variables template
├── workflows/
│   ├── __init__.py            # Imports all tag modules
│   ├── base_workflow.py       # Base class for all workflows
│   ├── {tag_name}/
│   │   ├── __init__.py        # Imports all endpoint workflows + orchestrator
│   │   ├── orchestrator_workflow.py  # Tag-level orchestrator
│   │   ├── {operation_id}/
│   │   │   ├── positive_workflow.py   # LLM-generated
│   │   │   ├── negative_workflow.py   # LLM-generated
│   │   │   └── security_workflow.py   # LLM-generated
│   │   ├── {operation_id_2}/
│   │   │   ├── ...
```

### Naming Conventions:

- **Tag directory:** `sanitize_dir_name(tag)` → lowercase, underscores, alphanumeric only
- **Operation directory:** `get_endpoint_dir_name(endpoint)` → sanitized operation_id, lowercase
- **Class names:** `to_class_name(name)` → PascalCase

---

## 17. Orchestrator Generation

**File:** `src/devdox_ai_locust/utils/scenario_generator.py`, `generate_tag_orchestrator()` (line 377)

One orchestrator per tag, sequencing all endpoints with data flow.

### Template: `workflow_orchestrator.j2`

Context:
- All endpoints in the tag (grouped by HTTP method)
- Request body schemas for POST/PUT/PATCH endpoints
- Response schemas for 2xx responses (for understanding ID fields)
- Class name derived from tag name

### Validation:
Same retry loop as per-endpoint (2 attempts):
- Syntax validation
- Class name fix (`_fix_orchestrator_class_name`)
- No semantic validation for orchestrators

### Generated Output:
A `SequentialTaskSet` class that:
- Creates resources via POST
- Reads via GET
- Updates via PUT/PATCH
- Deletes via DELETE
- Handles data flow between steps (IDs from create → subsequent operations)

---

## 18. __init__.py Generation

**File:** `src/devdox_ai_locust/cli.py` (line 664)

### workflows/__init__.py:
```python
from .{tag_dir_name_1} import *
from .{tag_dir_name_2} import *
...
```

### workflows/{tag}/__init__.py:
```python
"""Auto-generated workflow exports"""
# Only imports workflows that actually exist (handles partial failures)
from .{op_id}.positive_workflow import {ClassName}PositiveWorkflow
from .{op_id}.negative_workflow import {ClassName}NegativeWorkflow
from .{op_id}.security_workflow import {ClassName}SecurityWorkflow
from .orchestrator_workflow import {TagName}Orchestrator
```

**Robustness:** Only imports files that exist on disk (checks with `workflow_file.exists()`).

---

## 19. Base File Writing

**File:** `src/devdox_ai_locust/cli.py` (line 704)

```python
FOR each filename, content in base_files:
    IF filename == "base_workflow.py":
        → Write to workflows/ directory
    ELSE:
        → Write to output/ directory (root)
```

---

## 20. Static File Templates Detail

**File:** `src/devdox_ai_locust/templates/*.j2`

### 20a. `base_workflow.py.j2`

Generates the `BaseWorkflow` class that all LLM-generated workflows inherit from.

Key methods:
- `on_start()` / `on_stop()`: Lifecycle hooks
- `make_request(method, path, expected_status, **kwargs)`: Central HTTP method
  - Uses `self.client.request()` with `catch_response=True`
  - Validates status code against `expected_status` list
  - Returns parsed JSON (dict) or None on failure
  - Logs failures via Locust's response.failure()
- `login_and_get_token(path, credentials)`: Auth helper
- `logout(path)`: Auth cleanup
- `handle_401_retry()`: Re-auth on token expiry
- `_store_response_data(key, data)`: Cross-test data sharing

Template variables:
- `api_info`: API metadata
- `total_endpoints`: Endpoint count (for logging)

### 20b. `test_data.py.j2`

Generates the `TestDataGenerator` class.

Key methods:
- `generate_string(length, pattern, default)`: String generation with regex support
- `generate_integer(min_val, max_val, default, exclusive, multiple_of)`: Integer with constraints
- `generate_float(min_val, max_val, default, exclusive)`: Float with bounds
- `generate_boolean(default)`: Random boolean
- `generate_email()`: Faker email
- `random_uuid()`: UUID v4
- `random_date()`: ISO format date
- `generate_json_data(schema)`: Schema-based generation
- `_generate_from_pattern(pattern)`: Regex pattern matching

Template variables:
- `data_provider_content`: If set, imports mongo_data_provider

### 20c. `locust.py.j2`

Main locustfile entry point.

Structure:
- Imports all workflows via `from workflows import *`
- Defines `APIUser(HttpUser)` class
- Sets `tasks` dynamically from imported workflow classes
- Configures wait_time, host from environment

### 20d. `config.py.j2`

Environment-based configuration.

Variables:
- `API_BASE_URL`, `API_VERSION`
- Locust settings (users, spawn rate, run time, host)
- Test data settings (seed, timeout, retries)

### 20e. `utils.py.j2`

Utility functions:
- Response validation helpers
- Timing/performance helpers
- Data extraction utilities

### 20f. `user_classes.py.j2`

Generated user class definitions for Locust.

### 20g. `custom_flows.py.j2`

Extension point for user-defined custom test flows.

### 20h. `endpoint_template.py.j2`

Template for individual endpoint workflow files (pre-LLM fallback path).

### 20i. `fallback_locust.py.j2`

Minimal fallback locustfile when main generation fails completely.

---

## Appendix: Data Flow Diagram

```
OpenAPI Spec (JSON/YAML)
    │
    ▼
┌─────────────────────┐
│   Schema Fetching    │  swagger_utils.get_api_schema()
│   (URL or File)      │
└─────────┬───────────┘
          │ raw string
          ▼
┌─────────────────────┐
│   OpenAPI Parsing    │  OpenAPIParser.parse_schema() + parse_endpoints()
│   ($ref resolution)  │
└─────────┬───────────┘
          │ List[Endpoint]
          ▼
┌─────────────────────┐
│  Group by Tag +      │  cli._generate_scenario_based_tests()
│  Init Generator      │
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌────────┐  ┌──────────────────────────────────────────┐
│ Static │  │    Per-Endpoint (parallel)                │
│ Files  │  │                                          │
│ (Jinja)│  │  FOR each endpoint:                      │
└────────┘  │    ┌────────────────────────────────┐    │
            │    │ Pre-compute:                    │    │
            │    │  - Status codes                 │    │
            │    │  - Injection points (security)  │    │
            │    │  - Negative scenarios           │    │
            │    │  - Positive fields              │    │
            │    │  - Setup endpoints              │    │
            │    └──────────────┬─────────────────┘    │
            │                   │                       │
            │    ┌──────────────▼─────────────────┐    │
            │    │ Render prompt template (.j2)    │    │
            │    └──────────────┬─────────────────┘    │
            │                   │                       │
            │    ┌──────────────▼─────────────────┐    │
            │    │ LLM Call (Together AI)          │    │
            │    │ (with semaphore + retry)        │    │
            │    └──────────────┬─────────────────┘    │
            │                   │                       │
            │    ┌──────────────▼─────────────────┐    │
            │    │ Post-process:                   │    │
            │    │  - Extract <code>               │    │
            │    │  - Sanitize unicode             │    │
            │    │  - Fix class name               │    │
            │    │  - Fix bytes literals           │    │
            │    │  - Fix regex strings            │    │
            │    └──────────────┬─────────────────┘    │
            │                   │                       │
            │    ┌──────────────▼─────────────────┐    │
            │    │ Validate:                       │    │
            │    │  1. Syntax (compile)            │    │
            │    │  2. Semantic (CodeValidator)    │    │
            │    │                                 │    │
            │    │ IF fails + retries remain:      │    │
            │    │   → Fix prompt → retry LLM     │    │
            │    └──────────────┬─────────────────┘    │
            │                   │                       │
            │    ┌──────────────▼─────────────────┐    │
            │    │ Write to disk:                  │    │
            │    │  workflows/{tag}/{op}/          │    │
            │    │    {scenario}_workflow.py        │    │
            │    └────────────────────────────────┘    │
            └──────────────────────────────────────────┘
                        │
                        ▼
            ┌──────────────────────┐
            │ Orchestrator per tag  │
            │ (LLM + validation)   │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ __init__.py files    │
            │ (import wiring)      │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Base files to disk   │
            │ (locustfile, config, │
            │  test_data, etc.)    │
            └──────────────────────┘
```

---

## Appendix: FallbackHttpResponseRegistry

**File:** `src/devdox_ai_locust/utils/http_fallback_presets.py`

Used when the OpenAPI spec defines NO responses for an endpoint.

| Method | 2xx Codes | 4xx Codes |
|--------|-----------|-----------|
| GET | 200, 204 | 400, 401, 403, 404, 422 |
| POST | 200, 201, 204 | 400, 401, 403, 409, 415, 422 |
| PUT | 200, 201, 204 | 400, 401, 403, 409, 422 |
| PATCH | 200, 204 | 400, 401, 403, 409, 415, 422 |
| DELETE | 200, 204 | 400, 401, 403, 404, 422 |

All methods also include 5xx: 500, 502, 503, 504 (merged automatically).

When `exclude_auth=True`: 401 and 403 are removed from fallback codes.

---

## Appendix: Endpoint Detail Formatting

**Method:** `_format_single_endpoint()` (line 1093)

The formatted endpoint string sent to the LLM includes:

```
Operation: GET /api/v1/items/{item_id}
Operation ID: get_api_v1_items_item_id
Summary: Get item by ID
Description: ...

Parameters:
  - item_id [path]: string (required)
      description: The item identifier
  - include_deleted [query]: boolean (optional)
      default: false

  *** COOKIE VALUES MUST BE STRINGS ***     (if cookie params present)
  *** HEADER VALUES MUST BE STRINGS ***     (if header params present)

Request Body:
  Content-Type: application/json
  Required: true
  Schema:
    Required fields: ["name", "email"]
    Properties (use these EXACT field names):
      - name: string (REQUIRED)
          constraints: minLength=1, maxLength=100
      - email: string [email] (REQUIRED)
      - age: integer (optional)
          constraints: min=0, max=150
      - tags: array (optional)
          array items type: string
      - status: string (optional)
          allowed values: ['active', 'inactive', 'pending']

  *** FILE UPLOAD ENDPOINT ***              (if multipart/form-data)

Responses:
  - 200: Success
    Response Schema:
      - id: string
      - name: string
      - created_at: string [date-time]
  - 404: Not Found
  - 422: Validation Error
```
