"""
Rich-based Progress Display for Workflow Generation

Claude Code CLI-style output with:
- Live updating table for concurrent workers
- Clear status indicators (✓ ✗ ⟳ ·)
- Detailed error context with line numbers
- Non-overwhelming, informative reporting
"""

import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax


@dataclass
class WorkerState:
    """State of a single worker."""
    endpoint: str = ""
    scenario: str = ""
    status: str = "idle"  # idle, working, done, failed


@dataclass
class EndpointResult:
    """Result for a completed endpoint."""
    endpoint: str
    scenarios: Dict[str, str] = field(default_factory=dict)  # scenario -> status
    error_details: Optional[str] = None
    error_line: Optional[int] = None
    error_code_snippet: Optional[str] = None
    saved_path: Optional[str] = None


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
    Rich-based progress display with live updating table.

    Features:
    - Live table showing active workers
    - Scrolling log of completed items
    - Detailed error context
    - Concurrent-safe updates
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

        # Worker states (thread-safe)
        self._lock = threading.Lock()
        self._workers: Dict[int, WorkerState] = {
            i: WorkerState() for i in range(num_workers)
        }
        self._worker_assignment: Dict[str, int] = {}  # endpoint -> worker_id

        # Recent completions for scrolling log
        self._recent: List[EndpointResult] = []
        self._max_recent = 5

        # Failures for summary
        self._failures: List[FailureInfo] = []

        # Live display
        self._live: Optional[Live] = None

    def _get_worker_id(self, endpoint_info: str) -> int:
        """Get or assign a worker ID for an endpoint."""
        with self._lock:
            if endpoint_info in self._worker_assignment:
                return self._worker_assignment[endpoint_info]

            # Find idle worker
            for wid, state in self._workers.items():
                if state.status == "idle":
                    self._worker_assignment[endpoint_info] = wid
                    return wid

            # All busy, use round-robin
            wid = len(self._worker_assignment) % self.num_workers
            self._worker_assignment[endpoint_info] = wid
            return wid

    def _release_worker(self, endpoint_info: str) -> None:
        """Release a worker when endpoint is done."""
        with self._lock:
            if endpoint_info in self._worker_assignment:
                wid = self._worker_assignment[endpoint_info]
                self._workers[wid] = WorkerState()
                del self._worker_assignment[endpoint_info]

    def _build_display(self) -> Group:
        """Build the live display content."""
        elements = []

        # Active workers table
        workers_table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 1),
            expand=True,
        )
        workers_table.add_column("#", width=3, justify="right")
        workers_table.add_column("Status", width=8)
        workers_table.add_column("Endpoint", ratio=3)
        workers_table.add_column("Scenario", width=12)

        with self._lock:
            for wid, state in sorted(self._workers.items()):
                if state.status == "idle":
                    workers_table.add_row(
                        str(wid + 1),
                        "[dim]·[/dim]",
                        "[dim]idle[/dim]",
                        ""
                    )
                elif state.status == "working":
                    workers_table.add_row(
                        str(wid + 1),
                        "[yellow]⟳[/yellow]",
                        state.endpoint[:50] + "..." if len(state.endpoint) > 50 else state.endpoint,
                        state.scenario
                    )
                elif state.status == "done":
                    workers_table.add_row(
                        str(wid + 1),
                        "[green]✓[/green]",
                        state.endpoint[:50] + "..." if len(state.endpoint) > 50 else state.endpoint,
                        state.scenario
                    )
                elif state.status == "failed":
                    workers_table.add_row(
                        str(wid + 1),
                        "[red]✗[/red]",
                        state.endpoint[:50] + "..." if len(state.endpoint) > 50 else state.endpoint,
                        state.scenario
                    )

        # Progress bar
        total_processed = self.completed + self.failed
        if self.total > 0:
            percent = (total_processed * 100) // self.total
            bar_width = 40
            filled = (total_processed * bar_width) // self.total
            bar = "━" * filled + "─" * (bar_width - filled)
            progress_text = Text()
            progress_text.append(f"  {bar} ", style="bold")
            progress_text.append(f"{percent}%", style="bold cyan")
            progress_text.append(f"  {total_processed}/{self.total}  ")
            progress_text.append(f"{self.completed} ✓  ", style="green")
            progress_text.append(f"{self.failed} ✗  ", style="red")
            progress_text.append(f"{self.skipped} skipped", style="dim")
        else:
            progress_text = Text("  No endpoints to process")

        # Elapsed time
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
        progress_text.append(f"  [dim]({time_str})[/dim]")

        # Combine into panel
        elements.append(Panel(
            Group(workers_table, Text(""), progress_text),
            title="[bold]Generating Workflows[/bold]",
            border_style="blue",
        ))

        # Recent completions log
        if self._recent:
            recent_text = Text()
            for result in self._recent[-self._max_recent:]:
                # Endpoint line
                if any(s == "failed" for s in result.scenarios.values()):
                    recent_text.append("  ✗ ", style="red")
                else:
                    recent_text.append("  ✓ ", style="green")
                recent_text.append(result.endpoint + "\n")

                # Scenario details
                for scenario, status in result.scenarios.items():
                    if status == "done":
                        recent_text.append(f"    ├─ ✓ {scenario}\n", style="green")
                    elif status == "failed":
                        recent_text.append(f"    ├─ ✗ {scenario}\n", style="red")
                        if result.error_details:
                            # Show error context
                            error_short = result.error_details[:100]
                            if len(result.error_details) > 100:
                                error_short += "..."
                            recent_text.append(f"    │     {error_short}\n", style="dim red")
                    elif status == "skipped":
                        recent_text.append(f"    ├─ · {scenario} [dim](skipped)[/dim]\n")

            elements.append(Text("\nRecent:", style="bold"))
            elements.append(recent_text)

        return Group(*elements)

    def start(self) -> None:
        """Start the live display."""
        self.start_time = time.time()
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display and show summary."""
        if self._live:
            self._live.stop()
            self._live = None

        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        # Summary panel
        summary = Text()
        summary.append(f"\n  Completed: ", style="bold")
        summary.append(f"{self.completed} ✓\n", style="green")
        summary.append(f"  Failed:    ", style="bold")
        summary.append(f"{self.failed} ✗\n", style="red")
        summary.append(f"  Skipped:   ", style="bold")
        summary.append(f"{self.skipped}\n", style="dim")
        summary.append(f"\n  Duration:  {time_str}\n")

        self.console.print(Panel(
            summary,
            title="[bold green]Generation Complete[/bold green]",
            border_style="green",
        ))

        # Show failures with context
        if self._failures:
            self.console.print(f"\n[bold red]Failures ({len(self._failures)}):[/bold red]\n")
            for i, failure in enumerate(self._failures, 1):
                self.console.print(f"[bold]{i}. {failure.endpoint}[/bold] → {failure.scenario}")

                # Error message
                self.console.print(f"   [red]{failure.error}[/red]")

                # Code snippet if available
                if failure.code_snippet and failure.line_number:
                    self.console.print(f"   [dim]Line {failure.line_number}:[/dim]")
                    # Show a few lines of context
                    lines = failure.code_snippet.split('\n')
                    start = max(0, failure.line_number - 2)
                    end = min(len(lines), failure.line_number + 2)
                    for j, line in enumerate(lines[start:end], start + 1):
                        if j == failure.line_number:
                            self.console.print(f"   [red]→ {j:3d} │ {line}[/red]")
                        else:
                            self.console.print(f"     {j:3d} │ {line}", style="dim")

                # Saved path
                if failure.saved_path:
                    self.console.print(f"   [dim]Saved: {failure.saved_path}[/dim]")

                self.console.print()

    def _update_display(self) -> None:
        """Update the live display."""
        if self._live:
            self._live.update(self._build_display())

    def endpoint_start(self, endpoint_info: str) -> None:
        """Called when starting to process an endpoint."""
        wid = self._get_worker_id(endpoint_info)
        with self._lock:
            self._workers[wid] = WorkerState(
                endpoint=endpoint_info,
                scenario="",
                status="working"
            )
        self._update_display()

    def scenario_start(self, endpoint_info: str, scenario: str) -> None:
        """Called when starting a specific scenario."""
        wid = self._get_worker_id(endpoint_info)
        with self._lock:
            self._workers[wid].scenario = scenario
            self._workers[wid].status = "working"
        self._update_display()

    def scenario_done(self, endpoint_info: str, scenario: str) -> None:
        """Called when a scenario completes successfully."""
        wid = self._get_worker_id(endpoint_info)
        with self._lock:
            self._workers[wid].status = "done"
        self._update_display()

    def scenario_skipped(self, endpoint_info: str, scenario: str, reason: str = "") -> None:
        """Called when a scenario is skipped."""
        self.skipped += 1
        self._update_display()

    def scenario_detail(self, endpoint_info: str, scenario: str, detail: str) -> None:
        """Called to add detail about a scenario."""
        pass  # Details shown in recent log

    def scenario_retry(self, endpoint_info: str, scenario: str, attempt: int, max_attempts: int, error: str) -> None:
        """Called when a scenario is being retried."""
        # Only log on final failure
        if attempt >= max_attempts - 1:
            wid = self._get_worker_id(endpoint_info)
            with self._lock:
                self._workers[wid].status = "failed"
            self._update_display()

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

        wid = self._get_worker_id(endpoint_info)
        with self._lock:
            self._workers[wid].status = "failed"
        self._update_display()

    def endpoint_done(self, endpoint_info: str, scenarios_generated: int = 0) -> None:
        """Called when an endpoint finishes processing."""
        self.completed += 1

        # Add to recent
        with self._lock:
            wid = self._worker_assignment.get(endpoint_info, 0)
            state = self._workers.get(wid, WorkerState())
            result = EndpointResult(
                endpoint=endpoint_info,
                scenarios={"completed": "done"} if scenarios_generated > 0 else {},
            )
            self._recent.append(result)
            if len(self._recent) > self._max_recent * 2:
                self._recent = self._recent[-self._max_recent:]

        self._release_worker(endpoint_info)
        self._update_display()

    def endpoint_failed(self, endpoint_info: str, error: Exception) -> None:
        """Called when an endpoint fails completely."""
        self.failed += 1

        # Extract error details
        error_str = str(error)
        line_number = None
        code_snippet = None

        # Try to get code from CodeValidationError
        if hasattr(error, 'code'):
            code_snippet = error.code
        if hasattr(error, 'error'):
            error_str = error.error

        # Parse line number from error message
        import re
        line_match = re.search(r'line\s*(\d+)', error_str, re.IGNORECASE)
        if line_match:
            line_number = int(line_match.group(1))

        self._failures.append(FailureInfo(
            endpoint=endpoint_info,
            scenario="generation",
            error=error_str,
            line_number=line_number,
            code_snippet=code_snippet,
        ))

        self._release_worker(endpoint_info)
        self._update_display()

    def endpoint_skipped(self, endpoint_info: str, reason: str = "") -> None:
        """Called when an endpoint is skipped."""
        self.skipped += 1
        self._release_worker(endpoint_info)
        self._update_display()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
