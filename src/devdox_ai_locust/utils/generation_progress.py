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
from typing import Any, Dict, List, Optional, Set
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
    injection_locations: List[str] = field(default_factory=list)  # body, query, path, header


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


class GenerationProgress:
    """
    Simple, informative progress display.

    Prints to terminal as things happen - no fancy live updates.
    Focuses on being verbose and informative, especially for errors.

    Verbose mode shows detailed analysis for each endpoint.
    """

    def __init__(self, total: int, num_workers: int, console: Optional[Console] = None,
                 output_dir: Optional[Path] = None, verbose: bool = False):
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
            self.console.print(f"\n[bold red]═══ FAILURES ({len(self._failures)}) ═══[/bold red]\n")

            for i, failure in enumerate(self._failures, 1):
                self.console.print(f"[bold red]{i}. {failure.endpoint}[/bold red]")
                self.console.print(f"   Scenario: {failure.scenario}")
                self.console.print(f"   Error: {failure.error}")

                # Show code context if available
                if failure.code_snippet and failure.line_number:
                    self.console.print(f"   [dim]Code context (line {failure.line_number}):[/dim]")
                    lines = failure.code_snippet.split('\n')
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

    def set_endpoint_analysis(self, endpoint_info: str, analysis: EndpointAnalysis) -> None:
        """Set the analysis data for an endpoint (verbose mode)."""
        # Store by endpoint_info key to avoid race conditions with parallel processing
        self._endpoint_analyses[endpoint_info] = analysis

    def record_scenario_result(self, endpoint_info: str, scenario: str, result: ScenarioResult) -> None:
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

    def scenario_skipped(self, endpoint_info: str, scenario: str, reason: str = "") -> None:
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

    def scenario_retry(self, endpoint_info: str, scenario: str, attempt: int, max_attempts: int, error: str) -> None:
        """Called when a scenario is being retried."""
        # Print retry warnings - these are important
        if attempt >= max_attempts - 1:
            # Final retry failed
            short_error = error[:150] if len(error) > 150 else error
            self.console.print(
                f"  [yellow]⚠ RETRY FAILED[/yellow] {endpoint_info} → {scenario}"
            )
            self.console.print(f"    {short_error}", style="dim")

    def scenario_failed(self, endpoint_info: str, scenario: str, error: str,
                        line_number: Optional[int] = None, code: Optional[str] = None,
                        saved_path: Optional[str] = None) -> None:
        """Called when a scenario fails with detailed context."""
        self._failures.append(FailureInfo(
            endpoint=endpoint_info,
            scenario=scenario,
            error=error,
            line_number=line_number,
            code_snippet=code,
            saved_path=saved_path,
        ))

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

    def _print_verbose_endpoint(self, endpoint_info: str, analysis: EndpointAnalysis) -> None:
        """Print detailed verbose output for an endpoint."""
        c = self.console

        # Header with full path
        c.print(f"\n[bold]→ {analysis.method} {analysis.path}[/bold]")

        # OpenAPI Analysis
        c.print(f"  [dim]│[/dim]")
        c.print(f"  [dim]│[/dim] [cyan]── OpenAPI Analysis ──[/cyan]")
        responses_str = ", ".join(str(r) for r in analysis.responses_defined) if analysis.responses_defined else "none"
        c.print(f"  [dim]│[/dim] responses_defined: {responses_str}")
        c.print(f"  [dim]│[/dim] source_of_truth: {analysis.source_of_truth}")
        c.print(f"  [dim]│[/dim] content_type: {analysis.content_type}")

        # Schema Analysis
        schema = analysis.schema
        c.print(f"  [dim]│[/dim]")
        c.print(f"  [dim]│[/dim] [cyan]── Schema Analysis ──[/cyan]")
        c.print(f"  [dim]│[/dim] schema_type: {schema.schema_type}")
        if schema.discriminator:
            c.print(f"  [dim]│[/dim] discriminator: {schema.discriminator}")
            if schema.variants:
                c.print(f"  [dim]│[/dim] variants: {', '.join(schema.variants)}")
        c.print(f"  [dim]│[/dim] total_fields: {schema.total_fields}, required: {schema.required_fields}")
        if schema.patterns_found or schema.enums_found or schema.formats_found:
            constraints = []
            if schema.patterns_found:
                constraints.append(f"patterns={schema.patterns_found}")
            if schema.enums_found:
                constraints.append(f"enums={schema.enums_found}")
            if schema.formats_found:
                constraints.append(f"formats={schema.formats_found}")
            c.print(f"  [dim]│[/dim] constraints: {', '.join(constraints)}")

        # Setup Analysis (only if relevant)
        setup = analysis.setup
        if setup.needs_setup or setup.setup_endpoints_found > 0:
            c.print(f"  [dim]│[/dim]")
            c.print(f"  [dim]│[/dim] [cyan]── Setup Analysis ──[/cyan]")
            c.print(f"  [dim]│[/dim] needs_setup: {setup.needs_setup}")
            if setup.parent_resources:
                c.print(f"  [dim]│[/dim] parent_resources: {', '.join(setup.parent_resources)}")
            c.print(f"  [dim]│[/dim] setup_endpoints_found: {setup.setup_endpoints_found}")

        # Injection Analysis (only if has injectable fields)
        inj = analysis.injection
        if inj.total_injectable > 0:
            c.print(f"  [dim]│[/dim]")
            c.print(f"  [dim]│[/dim] [cyan]── Injection Analysis ──[/cyan]")
            c.print(f"  [dim]│[/dim] injectable_fields: {inj.total_injectable}")
            if inj.high_risk_fields:
                c.print(f"  [dim]│[/dim] high_risk: {', '.join(inj.high_risk_fields[:5])}")
            if inj.injection_locations:
                c.print(f"  [dim]│[/dim] locations: {', '.join(inj.injection_locations)}")

        # Pre-computation
        if analysis.positive_fields_precomputed > 0 or analysis.negative_scenarios_precomputed > 0:
            c.print(f"  [dim]│[/dim]")
            c.print(f"  [dim]│[/dim] [cyan]── Pre-computed ──[/cyan]")
            if analysis.positive_fields_precomputed > 0:
                c.print(f"  [dim]│[/dim] positive_fields: {analysis.positive_fields_precomputed} generators ready")
            if analysis.negative_scenarios_precomputed > 0:
                c.print(f"  [dim]│[/dim] negative_scenarios: {analysis.negative_scenarios_precomputed} identified")
                if analysis.negative_scenario_types:
                    for scenario_type in analysis.negative_scenario_types[:5]:
                        c.print(f"  [dim]│[/dim]   • {scenario_type}")

        # Warnings
        if analysis.warnings:
            c.print(f"  [dim]│[/dim]")
            c.print(f"  [dim]│[/dim] [yellow]⚠ warnings:[/yellow]")
            for warning in analysis.warnings[:3]:
                c.print(f"  [dim]│[/dim]   • {warning}")

        # Scenario Results
        c.print(f"  [dim]│[/dim]")
        for scenario_name in ["positive", "negative", "security"]:
            result = analysis.scenarios.get(scenario_name)
            if result:
                self._print_scenario_result(scenario_name, result)
            else:
                c.print(f"  [dim]├─[/dim] {scenario_name}  [dim]○ not generated[/dim]")

        c.print()  # Blank line after endpoint

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
        if hasattr(error, 'code'):
            code_snippet = error.code
        if hasattr(error, 'error'):
            error_str = error.error

        # Parse line number from error message
        import re
        line_match = re.search(r'line\s*(\d+)', error_str, re.IGNORECASE)
        if line_match:
            line_number = int(line_match.group(1))

        # Print failure with FULL error and traceback - no truncation
        self.console.print(f"  [red]✗[/red] {endpoint_info}")
        self.console.print(f"    [red]Error:[/red] {error_str}")

        # Print full traceback
        tb_str = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        self.console.print(f"    [dim]Traceback:[/dim]")
        for line in tb_str.split('\n'):
            if line.strip():
                self.console.print(f"    [dim]{line}[/dim]")

        if line_number and code_snippet:
            lines = code_snippet.split('\n')
            if 0 < line_number <= len(lines):
                self.console.print(f"    [dim]Line {line_number}:[/dim] {lines[line_number-1].strip()}")

        # Store for summary
        self._failures.append(FailureInfo(
            endpoint=endpoint_info,
            scenario="generation",
            error=error_str,
            line_number=line_number,
            code_snippet=code_snippet,
            saved_path=saved_path,
        ))

        self._check_milestone()

    def endpoint_skipped(self, endpoint_info: str, reason: str = "") -> None:
        """Called when an endpoint is skipped."""
        self.skipped += 1
        if reason:
            self.console.print(f"  [dim]○ {endpoint_info} (skipped: {reason})[/dim]")
        self._check_milestone()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
