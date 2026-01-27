"""
Simple Progress Display for Workflow Generation

Clean, scrolling terminal output that:
- Shows each endpoint as it completes
- Displays errors/warnings with full context
- Doesn't hide or swallow any information
- No fancy live updates - just informative output

Verbose mode adds detailed analysis metadata for each endpoint.
"""

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console


@dataclass
class FailureInfo:
    """Detailed information about a failure."""

    endpoint: str
    scenario: str
    error: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    saved_path: Optional[str] = None


@dataclass
class SchemaAnalysis:
    """Analysis of endpoint schema."""

    schema_type: str = "object"  # object, discriminated_union, array, primitive
    discriminator: Optional[str] = None
    variants: List[str] = field(default_factory=list)
    total_fields: int = 0
    required_fields: int = 0
    patterns_found: int = 0
    enums_found: int = 0
    formats_found: int = 0
    arrays_with_constraints: int = 0


@dataclass
class SetupAnalysis:
    """Analysis of setup requirements."""

    needs_setup: bool = False
    parent_resources: List[str] = field(default_factory=list)
    setup_endpoints_found: int = 0
    setup_endpoints: List[str] = field(default_factory=list)


@dataclass
class InjectionAnalysis:
    """Analysis of security injection points."""

    total_injectable: int = 0
    high_risk_fields: List[str] = field(default_factory=list)
    skipped_fields: List[str] = field(default_factory=list)
    injection_locations: List[str] = field(
        default_factory=list
    )  # body, query, path, header


@dataclass
class ScenarioResult:
    """Result of generating a scenario."""

    scenario_type: str  # positive, negative, security
    status: str  # success, failed, skipped
    skip_reason: Optional[str] = None
    time_seconds: float = 0.0
    tokens_used: int = 0
    fields_used: int = 0
    fields_total: int = 0
    generators_followed: int = 0
    generators_total: int = 0
    scenarios_generated: int = 0
    scenarios_total: int = 0
    retries: int = 0
    syntax_fixes: List[str] = field(default_factory=list)


@dataclass
class EndpointAnalysis:
    """Complete analysis for an endpoint."""

    method: str
    path: str
    operation_id: str = ""

    # OpenAPI Analysis
    responses_defined: List[int] = field(default_factory=list)
    source_of_truth: str = "spec"  # spec, fallback
    content_type: str = "application/json"

    # Schema Analysis
    schema: SchemaAnalysis = field(default_factory=SchemaAnalysis)

    # Constraint Detection
    strings_with_pattern: int = 0
    numbers_with_bounds: int = 0
    fields_with_format: int = 0

    # Setup Analysis
    setup: SetupAnalysis = field(default_factory=SetupAnalysis)

    # Injection Analysis
    injection: InjectionAnalysis = field(default_factory=InjectionAnalysis)

    # Pre-computation
    positive_fields_precomputed: int = 0
    negative_scenarios_precomputed: int = 0
    negative_scenario_types: List[str] = field(default_factory=list)

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Results
    scenarios: Dict[str, ScenarioResult] = field(default_factory=dict)


@dataclass
class OrchestratorEndpointInfo:
    """Information about an endpoint in the orchestrator."""

    method: str
    path: str
    operation_id: str = ""
    has_positive: bool = False
    has_negative: bool = False
    has_security: bool = False


@dataclass
class OrchestratorAnalysis:
    """Complete analysis for an orchestrator."""

    tag_name: str
    class_name: str = ""

    # Endpoint composition
    total_endpoints: int = 0
    valid_endpoints: int = 0  # Successfully generated
    endpoints: List[OrchestratorEndpointInfo] = field(default_factory=list)

    # CRUD Detection
    has_create: bool = False
    has_read: bool = False
    has_update: bool = False
    has_delete: bool = False
    crud_lifecycle_possible: bool = False

    # Auth Detection
    auth_endpoints_found: int = 0
    auth_tests_possible: bool = False

    # Orchestration Capabilities
    state_dependent_tests: List[str] = field(
        default_factory=list
    )  # 409, double-delete, etc.
    concurrent_tests_possible: bool = False
    resource_limit_tests: bool = False

    # Generation Stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_seconds: float = 0.0
    retries: int = 0

    # Warnings
    warnings: List[str] = field(default_factory=list)


class GenerationProgress:
    """
    Simple, informative progress display.

    Prints to terminal as things happen - no fancy live updates.
    Focuses on being verbose and informative, especially for errors.

    Verbose mode shows detailed analysis for each endpoint.
    """

    def __init__(
        self,
        total: int,
        num_workers: int,
        console: Optional[Console] = None,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        self.total = total
        self.num_workers = num_workers
        self.console = console or Console()
        self.output_dir = output_dir
        self.verbose = verbose

        # Counters
        self.completed = 0
        self.failed = 0
        self.skipped = 0

        # Timing
        self.start_time = time.time()

        # Track failures for summary
        self._failures: List[FailureInfo] = []

        # Milestone tracking (25%, 50%, 75%, 100%)
        self._printed_milestones: set = set()

        # Verbose mode: track endpoint analyses by endpoint_info key (thread-safe for parallel processing)
        self._endpoint_analyses: Dict[str, EndpointAnalysis] = {}

        # Verbose mode: track orchestrator analyses by tag_name
        self._orchestrator_analyses: Dict[str, OrchestratorAnalysis] = {}

        # Orchestrator counters
        self.orchestrator_completed = 0
        self._orchestrator_failed_count = 0
        self._orchestrator_skipped_count = 0

    def start(self) -> None:
        """Print start message."""
        self.start_time = time.time()
        self.console.print(
            f"\n[bold]→ Generating workflows[/bold] "
            f"({self.num_workers} concurrent, {self.total} endpoints)"
        )

    def stop(self) -> None:
        """Print final summary."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        self.console.print(
            f"\n[bold green]✓ Generation complete[/bold green] in {time_str}"
        )
        self.console.print(
            f"  [green]{self.completed} succeeded[/green], "
            f"[red]{self.failed} failed[/red], "
            f"[dim]{self.skipped} skipped[/dim]"
        )

        # Show detailed failure summary
        if self._failures:
            self.console.print(
                f"\n[bold red]═══ FAILURES ({len(self._failures)}) ═══[/bold red]\n"
            )

            for i, failure in enumerate(self._failures, 1):
                self.console.print(f"[bold red]{i}. {failure.endpoint}[/bold red]")
                self.console.print(f"   Scenario: {failure.scenario}")
                self.console.print(f"   Error: {failure.error}")

                # Show code context if available
                if failure.code_snippet and failure.line_number:
                    self.console.print(
                        f"   [dim]Code context (line {failure.line_number}):[/dim]"
                    )
                    lines = failure.code_snippet.split("\n")
                    start = max(0, failure.line_number - 3)
                    end = min(len(lines), failure.line_number + 2)

                    for j, line in enumerate(lines[start:end], start + 1):
                        if j == failure.line_number:
                            self.console.print(f"   [red]→ {j:4d} │ {line}[/red]")
                        else:
                            self.console.print(f"     {j:4d} │ {line}", style="dim")

                # Show saved path
                if failure.saved_path:
                    self.console.print(f"   [dim]Saved to: {failure.saved_path}[/dim]")

                self.console.print()  # Blank line between failures

    def _format_time(self) -> str:
        """Format elapsed time."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    def _check_milestone(self) -> None:
        """Print milestone progress at 25% intervals."""
        total_processed = self.completed + self.failed
        if self.total <= 0:
            return

        percent = (total_processed * 100) // self.total
        for milestone in [25, 50, 75, 100]:
            if percent >= milestone and milestone not in self._printed_milestones:
                self._printed_milestones.add(milestone)
                self.console.print(
                    f"  [bold]━ {milestone}%[/bold] ({total_processed}/{self.total}) "
                    f"— {self.completed} done, {self.failed} failed, {self.skipped} skipped "
                    f"[dim]({self._format_time()})[/dim]"
                )

    def endpoint_start(self, endpoint_info: str) -> None:
        """Called when starting to process an endpoint."""
        # Don't print anything - reduces noise
        pass

    def set_endpoint_analysis(
        self, endpoint_info: str, analysis: EndpointAnalysis
    ) -> None:
        """Set the analysis data for an endpoint (verbose mode)."""
        # Store by endpoint_info key to avoid race conditions with parallel processing
        self._endpoint_analyses[endpoint_info] = analysis

    def record_scenario_result(
        self, endpoint_info: str, scenario: str, result: ScenarioResult
    ) -> None:
        """Record the result of a scenario generation."""
        # Look up by endpoint_info to handle parallel processing correctly
        analysis = self._endpoint_analyses.get(endpoint_info)
        if analysis:
            analysis.scenarios[scenario] = result

    def scenario_start(self, endpoint_info: str, scenario: str) -> None:
        """Called when starting a specific scenario."""
        # Don't print anything - reduces noise
        pass

    def scenario_done(self, endpoint_info: str, scenario: str) -> None:
        """Called when a scenario completes successfully."""
        # Don't print individual scenario success - too noisy
        pass

    def scenario_skipped(
        self, endpoint_info: str, scenario: str, reason: str = ""
    ) -> None:
        """Called when a scenario is skipped."""
        self.skipped += 1
        # Record in analysis if available - look up by endpoint_info for thread safety
        analysis = self._endpoint_analyses.get(endpoint_info)
        if analysis:
            analysis.scenarios[scenario] = ScenarioResult(
                scenario_type=scenario,
                status="skipped",
                skip_reason=reason,
            )

    def scenario_detail(self, endpoint_info: str, scenario: str, detail: str) -> None:
        """Called to add detail about a scenario."""
        pass

    def scenario_retry(
        self,
        endpoint_info: str,
        scenario: str,
        attempt: int,
        max_attempts: int,
        error: str,
    ) -> None:
        """Called when a scenario is being retried."""
        # Print retry warnings - these are important
        if attempt >= max_attempts - 1:
            # Final retry failed
            short_error = error[:150] if len(error) > 150 else error
            self.console.print(
                f"  [yellow]⚠ RETRY FAILED[/yellow] {endpoint_info} → {scenario}"
            )
            self.console.print(f"    {short_error}", style="dim")

    def scenario_failed(
        self,
        endpoint_info: str,
        scenario: str,
        error: str,
        line_number: Optional[int] = None,
        code: Optional[str] = None,
        saved_path: Optional[str] = None,
    ) -> None:
        """Called when a scenario fails with detailed context."""
        self._failures.append(
            FailureInfo(
                endpoint=endpoint_info,
                scenario=scenario,
                error=error,
                line_number=line_number,
                code_snippet=code,
                saved_path=saved_path,
            )
        )

    def endpoint_done(self, endpoint_info: str, scenarios_generated: int = 0) -> None:
        """Called when an endpoint finishes processing."""
        self.completed += 1

        # Look up analysis by endpoint_info for thread-safe parallel processing
        analysis = self._endpoint_analyses.get(endpoint_info)
        if self.verbose and analysis:
            # Verbose mode: show full analysis
            self._print_verbose_endpoint(endpoint_info, analysis)
        else:
            # Normal mode: just show success
            self.console.print(f"  [green]✓[/green] {endpoint_info}")

        # Clean up this endpoint's analysis
        if endpoint_info in self._endpoint_analyses:
            del self._endpoint_analyses[endpoint_info]
        self._check_milestone()

    def _print_verbose_endpoint(
        self, endpoint_info: str, analysis: EndpointAnalysis
    ) -> None:
        """Print detailed verbose output for an endpoint."""
        c = self.console
        c.print(f"\n[bold]→ {analysis.method} {analysis.path}[/bold]")

        self._print_endpoint_openapi(c, analysis)
        self._print_endpoint_schema(c, analysis.schema)
        self._print_endpoint_setup(c, analysis.setup)
        self._print_endpoint_injection(c, analysis.injection)
        self._print_endpoint_precomputed(c, analysis)
        self._print_warnings(c, analysis.warnings)

        # Scenario Results
        c.print("  [dim]│[/dim]")
        for scenario_name in ["positive", "negative", "security"]:
            result = analysis.scenarios.get(scenario_name)
            if result:
                self._print_scenario_result(scenario_name, result)
            else:
                c.print(f"  [dim]├─[/dim] {scenario_name}  [dim]○ not generated[/dim]")

        c.print()  # Blank line after endpoint

    def _print_endpoint_openapi(self, c: Console, analysis: EndpointAnalysis) -> None:
        """Print OpenAPI analysis section."""
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [cyan]── OpenAPI Analysis ──[/cyan]")
        responses_str = (
            ", ".join(str(r) for r in analysis.responses_defined)
            if analysis.responses_defined
            else "none"
        )
        c.print(f"  [dim]│[/dim] responses_defined: {responses_str}")
        c.print(f"  [dim]│[/dim] source_of_truth: {analysis.source_of_truth}")
        c.print(f"  [dim]│[/dim] content_type: {analysis.content_type}")

    def _print_endpoint_schema(self, c: Console, schema: SchemaAnalysis) -> None:
        """Print schema analysis section."""
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [cyan]── Schema Analysis ──[/cyan]")
        c.print(f"  [dim]│[/dim] schema_type: {schema.schema_type}")
        if schema.discriminator:
            c.print(f"  [dim]│[/dim] discriminator: {schema.discriminator}")
            if schema.variants:
                c.print(f"  [dim]│[/dim] variants: {', '.join(schema.variants)}")
        c.print(
            f"  [dim]│[/dim] total_fields: {schema.total_fields}, required: {schema.required_fields}"
        )
        if schema.patterns_found or schema.enums_found or schema.formats_found:
            constraints = []
            if schema.patterns_found:
                constraints.append(f"patterns={schema.patterns_found}")
            if schema.enums_found:
                constraints.append(f"enums={schema.enums_found}")
            if schema.formats_found:
                constraints.append(f"formats={schema.formats_found}")
            c.print(f"  [dim]│[/dim] constraints: {', '.join(constraints)}")

    def _print_endpoint_setup(self, c: Console, setup: SetupAnalysis) -> None:
        """Print setup analysis section if relevant."""
        if not setup.needs_setup and setup.setup_endpoints_found <= 0:
            return
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [cyan]── Setup Analysis ──[/cyan]")
        c.print(f"  [dim]│[/dim] needs_setup: {setup.needs_setup}")
        if setup.parent_resources:
            c.print(
                f"  [dim]│[/dim] parent_resources: {', '.join(setup.parent_resources)}"
            )
        c.print(f"  [dim]│[/dim] setup_endpoints_found: {setup.setup_endpoints_found}")

    def _print_endpoint_injection(self, c: Console, inj: InjectionAnalysis) -> None:
        """Print injection analysis section if relevant."""
        if inj.total_injectable <= 0:
            return
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [cyan]── Injection Analysis ──[/cyan]")
        c.print(f"  [dim]│[/dim] injectable_fields: {inj.total_injectable}")
        if inj.high_risk_fields:
            c.print(f"  [dim]│[/dim] high_risk: {', '.join(inj.high_risk_fields[:5])}")
        if inj.injection_locations:
            c.print(f"  [dim]│[/dim] locations: {', '.join(inj.injection_locations)}")

    def _print_endpoint_precomputed(
        self, c: Console, analysis: EndpointAnalysis
    ) -> None:
        """Print pre-computation section if relevant."""
        if (
            analysis.positive_fields_precomputed <= 0
            and analysis.negative_scenarios_precomputed <= 0
        ):
            return
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [cyan]── Pre-computed ──[/cyan]")
        if analysis.positive_fields_precomputed > 0:
            c.print(
                f"  [dim]│[/dim] positive_fields: {analysis.positive_fields_precomputed} generators ready"
            )
        if analysis.negative_scenarios_precomputed > 0:
            c.print(
                f"  [dim]│[/dim] negative_scenarios: {analysis.negative_scenarios_precomputed} identified"
            )
            if analysis.negative_scenario_types:
                for scenario_type in analysis.negative_scenario_types[:5]:
                    c.print(f"  [dim]│[/dim]   • {scenario_type}")

    def _print_warnings(self, c: Console, warnings: List[str]) -> None:
        """Print warnings section if any."""
        if not warnings:
            return
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [yellow]⚠ warnings:[/yellow]")
        for warning in warnings[:3]:
            c.print(f"  [dim]│[/dim]   • {warning}")

    def _print_scenario_result(self, name: str, result: ScenarioResult) -> None:
        """Print a single scenario result."""
        c = self.console

        if result.status == "success":
            status_icon = "[green]✓[/green]"
            details = []
            if result.time_seconds > 0:
                details.append(f"{result.time_seconds:.1f}s")
            if result.tokens_used > 0:
                details.append(f"{result.tokens_used} tokens")
            if result.fields_used > 0 and result.fields_total > 0:
                details.append(f"fields={result.fields_used}/{result.fields_total}")
            if result.scenarios_generated > 0:
                details.append(f"scenarios={result.scenarios_generated}")
            if result.retries > 0:
                details.append(f"[yellow]retries={result.retries}[/yellow]")
            if result.syntax_fixes:
                details.append(f"[yellow]fixes={len(result.syntax_fixes)}[/yellow]")
            detail_str = f"  {', '.join(details)}" if details else ""
            c.print(f"  [dim]├─[/dim] {name}  {status_icon}{detail_str}")
        elif result.status == "skipped":
            reason = f": {result.skip_reason}" if result.skip_reason else ""
            c.print(f"  [dim]├─[/dim] {name}  [dim]⊘ skipped{reason}[/dim]")
        else:
            c.print(f"  [dim]├─[/dim] {name}  [red]✗ failed[/red]")

    def endpoint_failed(self, endpoint_info: str, error: Exception) -> None:
        """Called when an endpoint fails completely."""
        self.failed += 1

        # Extract error details
        error_str = str(error)
        line_number = None
        code_snippet = None
        saved_path = None

        # Try to get details from CodeValidationError
        if hasattr(error, "code"):
            code_snippet = error.code
        if hasattr(error, "error"):
            error_str = error.error

        # Parse line number from error message
        import re

        line_match = re.search(r"line\s*(\d+)", error_str, re.IGNORECASE)
        if line_match:
            line_number = int(line_match.group(1))

        # Print failure with FULL error and traceback - no truncation
        self.console.print(f"  [red]✗[/red] {endpoint_info}")
        self.console.print(f"    [red]Error:[/red] {error_str}")

        # Print full traceback
        tb_str = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        self.console.print("    [dim]Traceback:[/dim]")
        for line in tb_str.split("\n"):
            if line.strip():
                self.console.print(f"    [dim]{line}[/dim]")

        if line_number and code_snippet:
            lines = code_snippet.split("\n")
            if 0 < line_number <= len(lines):
                self.console.print(
                    f"    [dim]Line {line_number}:[/dim] {lines[line_number-1].strip()}"
                )

        # Store for summary
        self._failures.append(
            FailureInfo(
                endpoint=endpoint_info,
                scenario="generation",
                error=error_str,
                line_number=line_number,
                code_snippet=code_snippet,
                saved_path=saved_path,
            )
        )

        self._check_milestone()

    def endpoint_skipped(self, endpoint_info: str, reason: str = "") -> None:
        """Called when an endpoint is skipped."""
        self.skipped += 1
        if reason:
            self.console.print(f"  [dim]○ {endpoint_info} (skipped: {reason})[/dim]")
        self._check_milestone()

    # =========================================================================
    # Orchestrator Progress Methods
    # =========================================================================

    def set_orchestrator_analysis(
        self, tag_name: str, analysis: OrchestratorAnalysis
    ) -> None:
        """Set the analysis data for an orchestrator (verbose mode)."""
        self._orchestrator_analyses[tag_name] = analysis

    def orchestrator_done(self, tag_name: str) -> None:
        """Called when an orchestrator finishes processing."""
        self.orchestrator_completed += 1

        analysis = self._orchestrator_analyses.get(tag_name)
        if self.verbose and analysis:
            self._print_verbose_orchestrator(tag_name, analysis)
        else:
            self.console.print(
                f"  [green]✓[/green] {tag_name}/orchestrator_workflow.py"
            )

        # Clean up
        if tag_name in self._orchestrator_analyses:
            del self._orchestrator_analyses[tag_name]

    def orchestrator_failed(self, tag_name: str, error: Exception) -> None:
        """Called when an orchestrator fails."""
        self._orchestrator_failed_count += 1

        error_str = str(error)
        self.console.print(
            f"  [yellow]⚠[/yellow] {tag_name}/orchestrator_workflow.py failed"
        )
        self.console.print(f"    [red]Error:[/red] {error_str}")

        # Print full traceback
        tb_str = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        self.console.print("    [dim]Traceback:[/dim]")
        for line in tb_str.split("\n"):
            if line.strip():
                self.console.print(f"    [dim]{line}[/dim]")

        # Clean up
        if tag_name in self._orchestrator_analyses:
            del self._orchestrator_analyses[tag_name]

    def orchestrator_skipped(self, tag_name: str, reason: str = "") -> None:
        """Called when an orchestrator is skipped."""
        self._orchestrator_skipped_count += 1
        self.console.print(
            f"  [yellow]⚠[/yellow] {tag_name}/orchestrator_workflow.py skipped ({reason})"
        )

    def _print_verbose_orchestrator(
        self, tag_name: str, analysis: OrchestratorAnalysis
    ) -> None:
        """Print detailed verbose output for an orchestrator."""
        c = self.console
        c.print(f"\n[bold]→ Orchestrator: {tag_name}[/bold]")

        self._print_orchestrator_info(c, analysis)
        self._print_orchestrator_endpoints(c, analysis.endpoints)
        self._print_orchestrator_crud(c, analysis)
        self._print_orchestrator_auth(c, analysis)
        self._print_orchestrator_capabilities(c, analysis)
        self._print_orchestrator_stats(c, analysis)
        self._print_warnings(c, analysis.warnings)

        c.print("  [dim]│[/dim]")
        c.print(f"  [green]✓[/green] {tag_name}/orchestrator_workflow.py generated")
        c.print()  # Blank line after orchestrator

    def _print_orchestrator_info(
        self, c: Console, analysis: OrchestratorAnalysis
    ) -> None:
        """Print orchestrator basic info."""
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [cyan]── Orchestrator Info ──[/cyan]")
        c.print(f"  [dim]│[/dim] class_name: {analysis.class_name}")
        c.print(
            f"  [dim]│[/dim] endpoints: {analysis.valid_endpoints}/{analysis.total_endpoints} (valid/total)"
        )

    def _print_orchestrator_endpoints(
        self, c: Console, endpoints: List[OrchestratorEndpointInfo]
    ) -> None:
        """Print endpoint composition section."""
        if not endpoints:
            return
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [cyan]── Endpoint Composition ──[/cyan]")
        for ep in endpoints[:5]:
            scenarios = []
            if ep.has_positive:
                scenarios.append("pos")
            if ep.has_negative:
                scenarios.append("neg")
            if ep.has_security:
                scenarios.append("sec")
            scenarios_str = f"[{', '.join(scenarios)}]" if scenarios else "[none]"
            c.print(f"  [dim]│[/dim]   {ep.method} {ep.path} {scenarios_str}")
        if len(endpoints) > 5:
            c.print(f"  [dim]│[/dim]   ... and {len(endpoints) - 5} more")

    def _print_orchestrator_crud(
        self, c: Console, analysis: OrchestratorAnalysis
    ) -> None:
        """Print CRUD detection section."""
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [cyan]── CRUD Detection ──[/cyan]")
        crud_ops = []
        if analysis.has_create:
            crud_ops.append("Create")
        if analysis.has_read:
            crud_ops.append("Read")
        if analysis.has_update:
            crud_ops.append("Update")
        if analysis.has_delete:
            crud_ops.append("Delete")
        crud_str = ", ".join(crud_ops) if crud_ops else "none"
        c.print(f"  [dim]│[/dim] operations: {crud_str}")
        c.print(
            f"  [dim]│[/dim] crud_lifecycle_possible: {analysis.crud_lifecycle_possible}"
        )

    def _print_orchestrator_auth(
        self, c: Console, analysis: OrchestratorAnalysis
    ) -> None:
        """Print auth detection section if relevant."""
        if analysis.auth_endpoints_found <= 0 and not analysis.auth_tests_possible:
            return
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [cyan]── Auth Detection ──[/cyan]")
        c.print(f"  [dim]│[/dim] auth_endpoints_found: {analysis.auth_endpoints_found}")
        c.print(f"  [dim]│[/dim] auth_tests_possible: {analysis.auth_tests_possible}")

    def _print_orchestrator_capabilities(
        self, c: Console, analysis: OrchestratorAnalysis
    ) -> None:
        """Print orchestration capabilities section."""
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [cyan]── Orchestration Capabilities ──[/cyan]")
        if analysis.state_dependent_tests:
            c.print(
                f"  [dim]│[/dim] state_dependent_tests: {', '.join(analysis.state_dependent_tests)}"
            )
        else:
            c.print("  [dim]│[/dim] state_dependent_tests: none identified")
        c.print(
            f"  [dim]│[/dim] concurrent_tests_possible: {analysis.concurrent_tests_possible}"
        )
        c.print(f"  [dim]│[/dim] resource_limit_tests: {analysis.resource_limit_tests}")

    def _print_orchestrator_stats(
        self, c: Console, analysis: OrchestratorAnalysis
    ) -> None:
        """Print generation stats section if relevant."""
        if analysis.time_seconds <= 0 and analysis.prompt_tokens <= 0:
            return
        c.print("  [dim]│[/dim]")
        c.print("  [dim]│[/dim] [cyan]── Generation Stats ──[/cyan]")
        if analysis.time_seconds > 0:
            c.print(f"  [dim]│[/dim] time: {analysis.time_seconds:.1f}s")
        if analysis.prompt_tokens > 0 or analysis.completion_tokens > 0:
            c.print(
                f"  [dim]│[/dim] tokens: {analysis.prompt_tokens} prompt, {analysis.completion_tokens} completion"
            )
        if analysis.retries > 0:
            c.print(f"  [dim]│[/dim] [yellow]retries: {analysis.retries}[/yellow]")

    def __enter__(self) -> "GenerationProgress":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
