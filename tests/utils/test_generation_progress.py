"""
Tests for generation_progress module
"""

from rich.console import Console

from devdox_ai_locust.utils.generation_progress import (
    EndpointAnalysis,
    FailureInfo,
    GenerationProgress,
    InjectionAnalysis,
    OrchestratorAnalysis,
    OrchestratorEndpointInfo,
    SchemaAnalysis,
    ScenarioResult,
    SetupAnalysis,
)


class TestFailureInfo:
    """Test FailureInfo dataclass."""

    def test_required_fields(self):
        """Test creation with required fields only."""
        info = FailureInfo(endpoint="GET /users", scenario="positive", error="timeout")
        assert info.endpoint == "GET /users"
        assert info.scenario == "positive"
        assert info.error == "timeout"

    def test_optional_fields_default_none(self):
        """Test that optional fields default to None."""
        info = FailureInfo(endpoint="x", scenario="y", error="z")
        assert info.line_number is None
        assert info.code_snippet is None
        assert info.saved_path is None


class TestSchemaAnalysis:
    """Test SchemaAnalysis dataclass."""

    def test_defaults(self):
        """Test default values."""
        schema = SchemaAnalysis()
        assert schema.schema_type == "object"
        assert schema.discriminator is None
        assert schema.variants == []
        assert schema.total_fields == 0
        assert schema.required_fields == 0

    def test_custom_values(self):
        """Test with custom values."""
        schema = SchemaAnalysis(
            schema_type="discriminated_union",
            discriminator="type",
            variants=["cat", "dog"],
            total_fields=5,
            required_fields=2,
        )
        assert schema.schema_type == "discriminated_union"
        assert len(schema.variants) == 2


class TestSetupAnalysis:
    """Test SetupAnalysis dataclass."""

    def test_defaults(self):
        """Test default values."""
        setup = SetupAnalysis()
        assert setup.needs_setup is False
        assert setup.parent_resources == []
        assert setup.setup_endpoints_found == 0

    def test_with_setup(self):
        """Test with setup required."""
        setup = SetupAnalysis(
            needs_setup=True,
            parent_resources=["organization"],
            setup_endpoints_found=2,
            setup_endpoints=["POST /orgs", "GET /orgs/{id}"],
        )
        assert setup.needs_setup is True
        assert len(setup.setup_endpoints) == 2


class TestEndpointAnalysis:
    """Test EndpointAnalysis dataclass."""

    def test_creation(self):
        """Test basic creation with method and path."""
        ea = EndpointAnalysis(method="GET", path="/users")
        assert ea.method == "GET"
        assert ea.path == "/users"
        assert ea.operation_id == ""
        assert isinstance(ea.schema, SchemaAnalysis)
        assert isinstance(ea.setup, SetupAnalysis)
        assert isinstance(ea.injection, InjectionAnalysis)
        assert ea.scenarios == {}


class TestOrchestratorAnalysis:
    """Test OrchestratorAnalysis dataclass."""

    def test_creation(self):
        """Test creation with tag name."""
        oa = OrchestratorAnalysis(tag_name="users")
        assert oa.tag_name == "users"
        assert oa.total_endpoints == 0
        assert oa.has_create is False
        assert oa.crud_lifecycle_possible is False
        assert oa.endpoints == []
        assert oa.warnings == []

    def test_endpoint_info(self):
        """Test OrchestratorEndpointInfo."""
        ep = OrchestratorEndpointInfo(method="POST", path="/users")
        assert ep.has_positive is False
        assert ep.has_negative is False
        assert ep.has_security is False


class TestGenerationProgressContextManager:
    """Test GenerationProgress context manager behavior."""

    def test_enter_returns_self(self):
        """Test that __enter__ returns self."""
        gp = GenerationProgress(total=5, num_workers=2, console=Console(quiet=True))
        result = gp.__enter__()
        assert result is gp
        gp.__exit__(None, None, None)

    def test_context_manager_protocol(self):
        """Test using as context manager does not crash."""
        with GenerationProgress(
            total=3, num_workers=1, console=Console(quiet=True)
        ) as gp:
            assert gp.total == 3
            assert gp.completed == 0


class TestGenerationProgressRecording:
    """Test record_success and record_failure tracking."""

    def _make_progress(self):
        return GenerationProgress(total=5, num_workers=1, console=Console(quiet=True))

    def test_endpoint_done_increments_completed(self):
        """Test that endpoint_done increments completed counter."""
        gp = self._make_progress()
        gp.endpoint_done("GET /users")
        assert gp.completed == 1

    def test_endpoint_failed_increments_failed(self):
        """Test that endpoint_failed increments failed counter."""
        gp = self._make_progress()
        try:
            raise ValueError("test error")
        except ValueError as e:
            gp.endpoint_failed("GET /users", e)
        assert gp.failed == 1
        assert len(gp._failures) == 1

    def test_endpoint_skipped_increments_skipped(self):
        """Test that endpoint_skipped increments skipped counter."""
        gp = self._make_progress()
        gp.endpoint_skipped("GET /users", "no schema")
        assert gp.skipped == 1

    def test_scenario_failed_records_failure_info(self):
        """Test that scenario_failed stores FailureInfo."""
        gp = self._make_progress()
        gp.scenario_failed("GET /users", "positive", "validation error", line_number=10)
        assert len(gp._failures) == 1
        assert gp._failures[0].line_number == 10

    def test_scenario_skipped_increments_skipped(self):
        """Test that scenario_skipped increments skipped counter."""
        gp = self._make_progress()
        gp.scenario_skipped("GET /users", "security", "no injectable fields")
        assert gp.skipped == 1


class TestGenerationProgressDisplay:
    """Test that display/summary methods do not crash."""

    def test_stop_does_not_crash(self):
        """Test that stop (summary) runs without error."""
        gp = GenerationProgress(total=2, num_workers=1, console=Console(quiet=True))
        gp.start()
        gp.endpoint_done("GET /users")
        gp.stop()

    def test_stop_with_failures_does_not_crash(self):
        """Test that stop with failures runs without error."""
        gp = GenerationProgress(total=2, num_workers=1, console=Console(quiet=True))
        gp.start()
        gp.scenario_failed(
            "GET /users",
            "positive",
            "bad code",
            line_number=5,
            code="line1\nline2\nline3\nline4\nline5\nline6",
            saved_path="/tmp/fail.py",
        )
        gp.stop()

    def test_verbose_endpoint_does_not_crash(self):
        """Test verbose endpoint output runs without error."""
        gp = GenerationProgress(
            total=2, num_workers=1, console=Console(quiet=True), verbose=True
        )
        gp.start()
        analysis = EndpointAnalysis(
            method="POST",
            path="/users",
            operation_id="createUser",
            responses_defined=[201, 400],
            schema=SchemaAnalysis(total_fields=5, required_fields=3, patterns_found=1),
            setup=SetupAnalysis(needs_setup=True, setup_endpoints_found=1),
            injection=InjectionAnalysis(total_injectable=2, high_risk_fields=["name"]),
            positive_fields_precomputed=3,
            negative_scenarios_precomputed=2,
            negative_scenario_types=["missing_required", "invalid_type"],
            warnings=["No 404 response defined"],
        )
        analysis.scenarios["positive"] = ScenarioResult(
            scenario_type="positive",
            status="success",
            time_seconds=1.5,
            tokens_used=500,
        )
        gp.set_endpoint_analysis("POST /users", analysis)
        gp.endpoint_done("POST /users")
        gp.stop()

    def test_milestone_check_does_not_crash(self):
        """Test milestone printing at 25% intervals."""
        gp = GenerationProgress(total=4, num_workers=1, console=Console(quiet=True))
        gp.start()
        for i in range(4):
            gp.endpoint_done(f"EP{i}")
        gp.stop()

    def test_orchestrator_done_does_not_crash(self):
        """Test orchestrator completion output."""
        gp = GenerationProgress(total=1, num_workers=1, console=Console(quiet=True))
        gp.orchestrator_done("users")
        assert gp.orchestrator_completed == 1
