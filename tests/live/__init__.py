"""
Live integration tests for DevDox AI Locust.

These tests are SKIPPED by default to prevent running in CI/CD.

To run live tests:
    pytest tests/live --run-live --api-key YOUR_KEY --swagger-url YOUR_URL

Optional arguments:
    --output-dir DIR    Custom output directory (default: temp)
    --keep-output       Keep generated files after tests
    --mongodb-uri URI   MongoDB URI for database tests
    --postgresql-uri URI PostgreSQL URI for database tests

Environment variables (alternative to command-line args):
    TOGETHER_API_KEY    Together AI API key
    SWAGGER_URL         Swagger/OpenAPI URL
    MONGODB_URI         MongoDB connection URI
    POSTGRESQL_URI      PostgreSQL connection URI
"""
