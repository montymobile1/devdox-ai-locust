"""
Simple Progress Display for Workflow Generation

Clean, scrolling terminal output that:
- Shows each endpoint as it completes
- Displays errors/warnings with full context
- Doesn't hide or swallow any information
- No fancy live updates - just informative output
"""

import time
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


class GenerationProgress:
    """
    Simple, informative progress display.

    Prints to terminal as things happen - no fancy live updates.
    Focuses on being verbose and informative, especially for errors.
    """

    def __init__(self, total: int, num_workers: int, console: Optional[Console] = None,
                 output_dir: Optional[Path] = None):
        self.total = total
        self.num_workers = num_workers
        self.console = console or Console()
        self.output_dir = output_dir

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
        # Print success
        self.console.print(f"  [green]✓[/green] {endpoint_info}")
        self._check_milestone()

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

        # Print failure immediately with context
        self.console.print(f"  [red]✗[/red] {endpoint_info}")
        self.console.print(f"    [red]Error:[/red] {error_str[:200]}")

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
