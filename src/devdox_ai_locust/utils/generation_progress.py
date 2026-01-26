"""
Live-Refreshing Progress Display for Workflow Generation

Uses Rich Live to show active workers in a refreshing panel.
Only errors and the final summary persist in terminal history.
Success items disappear from the panel once done.
"""

import time
import threading
import traceback as tb_module
from typing import Dict, Optional
from rich.console import Console
from rich.live import Live
from rich.table import Table


class GenerationProgress:
    """
    Live-refreshing progress display for workflow generation.

    Active workers are shown in a refreshing panel.
    Only failures and the final summary persist in terminal history.
    """

    def __init__(self, total: int, num_workers: int, console: Optional[Console] = None):
        self.total = total
        self.num_workers = num_workers
        self.console = console or Console()

        # Counters
        self.completed = 0
        self.failed = 0
        self.skipped = 0

        # Timing
        self.start_time = time.time()

        # Active worker states: {endpoint_info: {"scenario": str, "detail": str}}
        self._workers: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()

        # Rich Live display
        self._live: Optional[Live] = None

    def start(self) -> None:
        """Start the live display."""
        self.start_time = time.time()
        self.console.print(
            f"\n[bold]→ Generating workflows[/bold] "
            f"({self.num_workers} concurrent, {self.total} endpoints)"
        )
        self._live = Live(
            self._render_panel(),
            console=self.console,
            refresh_per_second=4,
            transient=True,  # Remove panel when stopped
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display and print final summary."""
        if self._live:
            self._live.stop()
            self._live = None

        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        self.console.print(
            f"\n[bold green]✓ Generation complete[/bold green] in {time_str} — "
            f"[green]{self.completed} endpoints done[/green], "
            f"[red]{self.failed} failed[/red]"
            + (f", [dim]{self.skipped} scenarios skipped[/dim]" if self.skipped else "")
        )

    def endpoint_start(self, endpoint_info: str) -> None:
        """Register an endpoint as actively processing."""
        with self._lock:
            self._workers[endpoint_info] = {"scenario": "", "detail": "starting..."}
        self._refresh()

    def scenario_start(self, endpoint_info: str, scenario: str) -> None:
        """Update worker state: scenario started."""
        with self._lock:
            if endpoint_info in self._workers:
                self._workers[endpoint_info] = {"scenario": scenario, "detail": ""}
        self._refresh()

    def scenario_done(self, endpoint_info: str, scenario: str) -> None:
        """Update worker state: scenario completed."""
        with self._lock:
            if endpoint_info in self._workers:
                self._workers[endpoint_info] = {"scenario": scenario, "detail": "done"}
        self._refresh()

    def scenario_skipped(self, endpoint_info: str, scenario: str, reason: str = "") -> None:
        """Update worker state: scenario skipped."""
        self.skipped += 1
        with self._lock:
            if endpoint_info in self._workers:
                self._workers[endpoint_info] = {"scenario": scenario, "detail": f"skipped"}
        self._refresh()

    def scenario_detail(self, endpoint_info: str, scenario: str, detail: str) -> None:
        """Update worker state with a sub-step detail."""
        with self._lock:
            if endpoint_info in self._workers:
                self._workers[endpoint_info] = {"scenario": scenario, "detail": detail}
        self._refresh()

    def scenario_retry(self, endpoint_info: str, scenario: str, attempt: int, max_attempts: int, error: str) -> None:
        """Persist a retry warning above the live panel."""
        short_error = error[:150] if len(error) > 150 else error
        self._persist(
            f"  [yellow]⚠[/yellow] {endpoint_info} → {scenario} retry {attempt}/{max_attempts}: "
            f"[dim]{short_error}[/dim]"
        )
        with self._lock:
            if endpoint_info in self._workers:
                self._workers[endpoint_info] = {"scenario": scenario, "detail": f"retry {attempt}/{max_attempts}"}
        self._refresh()

    def endpoint_done(self, endpoint_info: str, scenarios_generated: int = 0) -> None:
        """Remove endpoint from active workers (success disappears)."""
        self.completed += 1
        with self._lock:
            self._workers.pop(endpoint_info, None)
        self._refresh()

    def endpoint_failed(self, endpoint_info: str, error: Exception) -> None:
        """Persist failure above the live panel, then remove from active workers."""
        self.failed += 1
        # Persist the failure with traceback
        self._persist(f"  [red]✗[/red] {endpoint_info} [red]FAILED[/red]")
        exc_lines = tb_module.format_exception(type(error), error, error.__traceback__)
        for line in exc_lines:
            for subline in line.rstrip().split("\n"):
                self._persist(f"    [red]{subline}[/red]")

        with self._lock:
            self._workers.pop(endpoint_info, None)
        self._refresh()

    def endpoint_skipped(self, endpoint_info: str, reason: str = "") -> None:
        """Remove endpoint from active workers (skipped)."""
        self.skipped += 1
        with self._lock:
            self._workers.pop(endpoint_info, None)
        self._refresh()

    def _persist(self, markup: str) -> None:
        """Print a line above the Live panel so it stays in terminal history."""
        if self._live:
            self._live.console.print(markup, highlight=False)
        else:
            self.console.print(markup, highlight=False)

    def _refresh(self) -> None:
        """Update the live panel with current state."""
        if self._live:
            self._live.update(self._render_panel())

    def _render_panel(self) -> Table:
        """Render the live panel content showing active workers + progress bar."""
        total_processed = self.completed + self.failed
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
        percent = (total_processed * 100 // self.total) if self.total > 0 else 0

        # Build a simple table
        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", width=3)
        table.add_column(min_width=50)
        table.add_column(style="dim", justify="right")

        # Progress header row
        bar_width = 30
        filled = int(bar_width * percent / 100) if self.total > 0 else 0
        bar = "━" * filled + "╺" + "─" * (bar_width - filled - 1)
        status_text = (
            f"[bold]{percent}%[/bold] ({total_processed}/{self.total}) "
            f"— {self.completed} done, {self.failed} failed, {self.skipped} skipped"
        )
        table.add_row("", f"[cyan]{bar}[/cyan] {status_text}", f"[dim]{time_str}[/dim]")

        # Active workers
        with self._lock:
            workers_snapshot = dict(self._workers)

        for endpoint, state in list(workers_snapshot.items())[:self.num_workers]:
            scenario = state.get("scenario", "")
            detail = state.get("detail", "")

            endpoint_display = endpoint[:55] + "..." if len(endpoint) > 55 else endpoint
            if scenario and detail:
                info = f"[dim]{scenario}: {detail}[/dim]"
            elif scenario:
                info = f"[dim]{scenario}[/dim]"
            else:
                info = f"[dim]{detail}[/dim]"

            table.add_row("[cyan]●[/cyan]", endpoint_display, info)

        return table

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
