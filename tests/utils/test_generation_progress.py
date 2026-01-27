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


class TestGenerationProgressOrchestrator:
    """Test orchestrator progress methods."""

    def _make_progress(self, verbose=False):
        return GenerationProgress(
            total=5, num_workers=1, console=Console(quiet=True), verbose=verbose
        )

    def test_orchestrator_failed(self):
        gp = self._make_progress()
        try:
            raise RuntimeError("orch fail")
        except RuntimeError as e:
            gp.orchestrator_failed("users", e)
        assert gp._orchestrator_failed_count == 1

    def test_orchestrator_skipped(self):
        gp = self._make_progress()
        gp.orchestrator_skipped("users", "no valid endpoints")
        assert gp._orchestrator_skipped_count == 1

    def test_set_orchestrator_analysis(self):
        gp = self._make_progress()
        analysis = OrchestratorAnalysis(tag_name="users")
        gp.set_orchestrator_analysis("users", analysis)
        assert "users" in gp._orchestrator_analyses

    def test_orchestrator_done_verbose(self):
        gp = self._make_progress(verbose=True)
        analysis = OrchestratorAnalysis(
            tag_name="users",
            class_name="UsersOrchestrator",
            total_endpoints=3,
            valid_endpoints=2,
            endpoints=[
                OrchestratorEndpointInfo(
                    method="GET",
                    path="/users",
                    has_positive=True,
                    has_negative=True,
                    has_security=False,
                ),
                OrchestratorEndpointInfo(
                    method="POST",
                    path="/users",
                    has_positive=True,
                    has_negative=False,
                    has_security=True,
                ),
            ],
            has_create=True,
            has_read=True,
            has_update=False,
            has_delete=False,
            crud_lifecycle_possible=False,
            auth_endpoints_found=1,
            auth_tests_possible=True,
            state_dependent_tests=["409_conflict"],
            concurrent_tests_possible=True,
            resource_limit_tests=True,
            prompt_tokens=100,
            completion_tokens=200,
            time_seconds=2.5,
            retries=1,
            warnings=["Some warning"],
        )
        gp.set_orchestrator_analysis("users", analysis)
        gp.orchestrator_done("users")
        assert gp.orchestrator_completed == 1
        assert "users" not in gp._orchestrator_analyses  # cleaned up

    def test_orchestrator_failed_cleans_up(self):
        gp = self._make_progress()
        analysis = OrchestratorAnalysis(tag_name="users")
        gp.set_orchestrator_analysis("users", analysis)
        try:
            raise RuntimeError("fail")
        except RuntimeError as e:
            gp.orchestrator_failed("users", e)
        assert "users" not in gp._orchestrator_analyses


class TestGenerationProgressScenarios:
    """Test scenario-level progress methods."""

    def _make_progress(self):
        return GenerationProgress(total=5, num_workers=1, console=Console(quiet=True))

    def test_endpoint_start_noop(self):
        gp = self._make_progress()
        gp.endpoint_start("GET /users")  # should not crash

    def test_scenario_start_noop(self):
        gp = self._make_progress()
        gp.scenario_start("GET /users", "positive")

    def test_scenario_done_noop(self):
        gp = self._make_progress()
        gp.scenario_done("GET /users", "positive")

    def test_scenario_detail_noop(self):
        gp = self._make_progress()
        gp.scenario_detail("GET /users", "positive", "some detail")

    def test_scenario_retry_not_final(self):
        gp = self._make_progress()
        gp.scenario_retry("GET /users", "positive", 0, 3, "error msg")

    def test_scenario_retry_final(self):
        gp = self._make_progress()
        gp.scenario_retry("GET /users", "positive", 2, 3, "error msg" * 50)

    def test_record_scenario_result(self):
        gp = self._make_progress()
        analysis = EndpointAnalysis(method="GET", path="/users")
        gp.set_endpoint_analysis("GET /users", analysis)
        result = ScenarioResult(scenario_type="positive", status="success")
        gp.record_scenario_result("GET /users", "positive", result)
        assert "positive" in analysis.scenarios

    def test_record_scenario_result_no_analysis(self):
        gp = self._make_progress()
        result = ScenarioResult(scenario_type="positive", status="success")
        gp.record_scenario_result("MISSING /endpoint", "positive", result)
        # Should not crash

    def test_scenario_skipped_with_analysis(self):
        gp = self._make_progress()
        analysis = EndpointAnalysis(method="GET", path="/users")
        gp.set_endpoint_analysis("GET /users", analysis)
        gp.scenario_skipped("GET /users", "security", "no injectable fields")
        assert gp.skipped == 1
        assert "security" in analysis.scenarios
        assert analysis.scenarios["security"].status == "skipped"

    def test_scenario_skipped_without_analysis(self):
        gp = self._make_progress()
        gp.scenario_skipped("MISSING /ep", "security", "reason")
        assert gp.skipped == 1


class TestGenerationProgressEndpointFailure:
    """Test endpoint failure with various error types."""

    def test_endpoint_failed_with_code_and_error_attrs(self):
        gp = GenerationProgress(total=5, num_workers=1, console=Console(quiet=True))

        class CodeError(Exception):
            def __init__(self):
                super().__init__("validation failed")
                self.code = "line1\nline2\nline3\nline4\nline5"
                self.error = "Syntax error on line 3"

        try:
            raise CodeError()
        except CodeError as e:
            gp.endpoint_failed("POST /users", e)

        assert gp.failed == 1
        assert len(gp._failures) == 1
        assert gp._failures[0].line_number == 3

    def test_endpoint_failed_no_line_number(self):
        gp = GenerationProgress(total=5, num_workers=1, console=Console(quiet=True))
        try:
            raise ValueError("generic error no line")
        except ValueError as e:
            gp.endpoint_failed("GET /items", e)
        assert gp.failed == 1
        assert gp._failures[0].line_number is None

    def test_endpoint_skipped_no_reason(self):
        gp = GenerationProgress(total=5, num_workers=1, console=Console(quiet=True))
        gp.endpoint_skipped("GET /users")
        assert gp.skipped == 1


class TestGenerationProgressVerboseEndpoint:
    """Test verbose endpoint printing branches."""

    def test_verbose_endpoint_no_analysis(self):
        """Verbose mode but no analysis set - falls through to normal."""
        gp = GenerationProgress(
            total=2, num_workers=1, console=Console(quiet=True), verbose=True
        )
        gp.endpoint_done("GET /users")
        assert gp.completed == 1

    def test_verbose_endpoint_with_all_scenarios(self):
        """Verbose mode with all scenario results."""
        gp = GenerationProgress(
            total=2, num_workers=1, console=Console(quiet=True), verbose=True
        )
        analysis = EndpointAnalysis(
            method="POST",
            path="/users",
            schema=SchemaAnalysis(
                schema_type="discriminated_union",
                discriminator="type",
                variants=["cat", "dog"],
                total_fields=5,
                required_fields=2,
                enums_found=1,
                formats_found=1,
            ),
            setup=SetupAnalysis(needs_setup=False, setup_endpoints_found=0),
            injection=InjectionAnalysis(
                total_injectable=3,
                high_risk_fields=["name", "email", "query", "path", "header", "extra"],
                injection_locations=["body", "query"],
            ),
            positive_fields_precomputed=5,
            negative_scenarios_precomputed=3,
            negative_scenario_types=[
                "missing_req",
                "invalid_type",
                "overflow",
                "xss",
                "sqli",
                "extra_type",
            ],
        )
        analysis.scenarios["positive"] = ScenarioResult(
            scenario_type="positive",
            status="success",
            time_seconds=1.2,
            tokens_used=500,
            fields_used=3,
            fields_total=5,
            scenarios_generated=2,
            retries=1,
            syntax_fixes=["fix1"],
        )
        analysis.scenarios["negative"] = ScenarioResult(
            scenario_type="negative",
            status="failed",
        )
        analysis.scenarios["security"] = ScenarioResult(
            scenario_type="security",
            status="skipped",
            skip_reason="no injectable",
        )
        gp.set_endpoint_analysis("POST /users", analysis)
        gp.endpoint_done("POST /users")
        assert gp.completed == 1

    def test_verbose_endpoint_no_precomputed(self):
        """Verbose mode with no precomputed data (skips that section)."""
        gp = GenerationProgress(
            total=1, num_workers=1, console=Console(quiet=True), verbose=True
        )
        analysis = EndpointAnalysis(method="GET", path="/items")
        gp.set_endpoint_analysis("GET /items", analysis)
        gp.endpoint_done("GET /items")

    def test_verbose_endpoint_no_injection(self):
        """Verbose mode with no injectable fields (skips injection section)."""
        gp = GenerationProgress(
            total=1, num_workers=1, console=Console(quiet=True), verbose=True
        )
        analysis = EndpointAnalysis(
            method="GET",
            path="/items",
            injection=InjectionAnalysis(total_injectable=0),
        )
        gp.set_endpoint_analysis("GET /items", analysis)
        gp.endpoint_done("GET /items")


class TestGenerationProgressMilestones:
    """Test milestone checking edge cases."""

    def test_milestone_zero_total(self):
        gp = GenerationProgress(total=0, num_workers=1, console=Console(quiet=True))
        gp._check_milestone()  # should not crash or divide by zero

    def test_format_time_with_minutes(self):
        import time

        gp = GenerationProgress(total=1, num_workers=1, console=Console(quiet=True))
        gp.start_time = time.time() - 125  # 2m 5s ago
        result = gp._format_time()
        assert "m" in result

    def test_format_time_seconds_only(self):
        import time

        gp = GenerationProgress(total=1, num_workers=1, console=Console(quiet=True))
        gp.start_time = time.time() - 30
        result = gp._format_time()
        assert "s" in result
        assert "m" not in result


class TestGenerationProgressStopWithFailureDetails:
    """Test stop with code context in failures."""

    def test_stop_failure_with_code_no_line_match(self):
        gp = GenerationProgress(total=1, num_workers=1, console=Console(quiet=True))
        gp.start()
        gp.scenario_failed(
            "GET /users",
            "positive",
            "bad code",
            line_number=None,
            code=None,
            saved_path=None,
        )
        gp.stop()

    def test_stop_no_failures(self):
        gp = GenerationProgress(total=1, num_workers=1, console=Console(quiet=True))
        gp.start()
        gp.endpoint_done("GET /users")
        gp.stop()


class TestInjectionAnalysis:
    """Test InjectionAnalysis dataclass."""

    def test_defaults(self):
        inj = InjectionAnalysis()
        assert inj.total_injectable == 0
        assert inj.high_risk_fields == []
        assert inj.skipped_fields == []
        assert inj.injection_locations == []


class TestScenarioResult:
    """Test ScenarioResult dataclass."""

    def test_defaults(self):
        result = ScenarioResult(scenario_type="positive", status="success")
        assert result.time_seconds == 0.0
        assert result.tokens_used == 0
        assert result.retries == 0
        assert result.syntax_fixes == []

    def test_custom_values(self):
        result = ScenarioResult(
            scenario_type="negative",
            status="failed",
            time_seconds=2.5,
            tokens_used=1000,
            retries=3,
        )
        assert result.time_seconds == 2.5
        assert result.retries == 3
