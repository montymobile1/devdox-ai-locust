"""
Simple Progress Display for Workflow Generation

Minimal, non-intrusive output:
- Milestone progress at 25% intervals
- Errors/retries only when they matter
- Final summary
"""

import time
import traceback as tb_module
from typing import Optional
from rich.console import Console


class GenerationProgress:
    """
    Simple progress display. Only prints milestones and errors.
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

        # Milestone tracking
        self._printed_milestones: set = set()

    def start(self) -> None:
        self.start_time = time.time()
        self.console.print(
            f"\n[bold]→ Generating workflows[/bold] "
            f"({self.num_workers} concurrent, {self.total} endpoints)"
        )

    def stop(self) -> None:
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        self.console.print(
            f"\n[bold green]✓ Generation complete[/bold green] in {time_str} — "
            f"[green]{self.completed} done[/green], "
            f"[red]{self.failed} failed[/red], "
            f"[dim]{self.skipped} skipped[/dim]"
        )

    def endpoint_start(self, endpoint_info: str) -> None:
        pass

    def scenario_start(self, endpoint_info: str, scenario: str) -> None:
        pass

    def scenario_done(self, endpoint_info: str, scenario: str) -> None:
        pass

    def scenario_skipped(self, endpoint_info: str, scenario: str, reason: str = "") -> None:
        self.skipped += 1

    def scenario_detail(self, endpoint_info: str, scenario: str, detail: str) -> None:
        pass

    def scenario_retry(self, endpoint_info: str, scenario: str, attempt: int, max_attempts: int, error: str) -> None:
        # Only show final retry failure (attempt == max_attempts)
        if attempt >= max_attempts - 1:
            short_error = error[:120] if len(error) > 120 else error
            self.console.print(
                f"  [yellow]⚠[/yellow] {endpoint_info} → {scenario}: {short_error}",
                highlight=False,
            )

    def endpoint_done(self, endpoint_info: str, scenarios_generated: int = 0) -> None:
        self.completed += 1
        self._check_milestone()

    def endpoint_failed(self, endpoint_info: str, error: Exception) -> None:
        self.failed += 1
        self.console.print(f"  [red]✗[/red] {endpoint_info} [red]FAILED[/red]")
        exc_lines = tb_module.format_exception(type(error), error, error.__traceback__)
        for line in exc_lines:
            for subline in line.rstrip().split("\n"):
                self.console.print(f"    [red]{subline}[/red]", highlight=False)
        self._check_milestone()

    def endpoint_skipped(self, endpoint_info: str, reason: str = "") -> None:
        self.skipped += 1
        self._check_milestone()

    def _check_milestone(self) -> None:
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
                    f"  [bold]━ {milestone}%[/bold] ({total_processed}/{self.total}) "
                    f"— {self.completed} done, {self.failed} failed, {self.skipped} skipped "
                    f"[dim]({time_str})[/dim]"
                )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
