"""
Persistent Progress Display for Workflow Generation

Provides Claude Code CLI-inspired persistent scrolling output where every
event stays in the terminal history for auditing. Nothing is erased or
refreshed - all output accumulates naturally.

Style:
  ● current/active items
  ✓ completed items
  ✗ failed items
  ⊘ skipped items
  ⚠ warnings/retries
"""

import time
import traceback as tb_module
from typing import Optional
from rich.console import Console


class GenerationProgress:
    """
    Persistent scrolling progress display for workflow generation.

    All output is printed line-by-line and stays in terminal history.
    No live-refresh, no erasing - everything is auditable.
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

        # Milestone tracking (print at every 25%)
        self._printed_milestones: set = set()

    def start(self) -> None:
        """Print the start banner."""
        self.start_time = time.time()
        self.console.print(
            f"\n[bold]→ Generating workflows[/bold] "
            f"({self.num_workers} concurrent, {self.total} endpoints)"
        )

    def stop(self) -> None:
        """Print the completion summary."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        self.console.print()
        self.console.print(
            f"[bold green]✓ Generation complete[/bold green] in {time_str} — "
            f"[green]{self.completed} endpoints done[/green], "
            f"[red]{self.failed} failed[/red]"
            + (f", [dim]{self.skipped} scenarios skipped[/dim]" if self.skipped else "")
        )

    def endpoint_start(self, endpoint_info: str) -> None:
        """Log that an endpoint started processing."""
        self.console.print(f"  [cyan]●[/cyan] {endpoint_info}")

    def scenario_start(self, endpoint_info: str, scenario: str) -> None:
        """Log that a specific scenario started for an endpoint."""
        self.console.print(f"    [dim]├─[/dim] {scenario}...", highlight=False)

    def scenario_done(self, endpoint_info: str, scenario: str) -> None:
        """Log that a scenario completed successfully."""
        self.console.print(f"    [dim]├─[/dim] [green]✓[/green] {scenario} done")

    def scenario_skipped(self, endpoint_info: str, scenario: str, reason: str = "") -> None:
        """Log that a scenario was skipped."""
        self.skipped += 1
        reason_text = f" ({reason})" if reason else ""
        self.console.print(f"    [dim]├─[/dim] [dim]⊘ {scenario} skipped{reason_text}[/dim]")

    def scenario_detail(self, endpoint_info: str, scenario: str, detail: str) -> None:
        """Log a sub-step detail within a scenario (pre-computation, LLM call, validation)."""
        self.console.print(f"    [dim]│    {scenario}: {detail}[/dim]", highlight=False)

    def scenario_retry(self, endpoint_info: str, scenario: str, attempt: int, max_attempts: int, error: str) -> None:
        """Log a retry attempt for a scenario."""
        self.console.print(
            f"    [dim]├─[/dim] [yellow]⚠[/yellow] {scenario} retry {attempt}/{max_attempts}: "
            f"{error}"
        )

    def endpoint_done(self, endpoint_info: str, scenarios_generated: int = 0) -> None:
        """Log that an endpoint completed successfully."""
        self.completed += 1
        self.console.print(
            f"  [green]✓[/green] {endpoint_info} "
            f"[dim]({scenarios_generated} scenarios)[/dim]"
        )
        self._check_milestone()

    def endpoint_failed(self, endpoint_info: str, error: Exception) -> None:
        """Log that an endpoint failed with full traceback."""
        self.failed += 1
        self.console.print(f"  [red]✗[/red] {endpoint_info} [red]FAILED[/red]")
        # Full traceback - never hide exceptions
        exc_lines = tb_module.format_exception(type(error), error, error.__traceback__)
        for line in exc_lines:
            for subline in line.rstrip().split("\n"):
                self.console.print(f"    [red]{subline}[/red]", highlight=False)
        self._check_milestone()

    def endpoint_skipped(self, endpoint_info: str, reason: str = "") -> None:
        """Log that an entire endpoint was skipped."""
        self.skipped += 1
        reason_text = f" ({reason})" if reason else ""
        self.console.print(f"  [dim]⊘ {endpoint_info} skipped{reason_text}[/dim]")
        self._check_milestone()

    def _check_milestone(self) -> None:
        """Print progress milestone at 25% intervals."""
        total_processed = self.completed + self.failed
        if self.total <= 0:
            return

        percent = (total_processed * 100) // self.total
        for milestone in [25, 50, 75, 100]:
            if percent >= milestone and milestone not in self._printed_milestones:
                self._printed_milestones.add(milestone)
                elapsed = time.time() - self.start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
                self.console.print(
                    f"\n  [bold]━ {milestone}% ({total_processed}/{self.total})[/bold] "
                    f"— {self.completed} done, {self.failed} failed, {self.skipped} skipped "
                    f"[dim]({time_str})[/dim]\n"
                )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
