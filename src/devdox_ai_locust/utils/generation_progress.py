"""
Rich Live Progress Display for Workflow Generation

Provides a Claude Code CLI-inspired live display showing:
- Overall progress bar with counts
- Currently active workers with spinners
- Recent completions/skips/failures
- Elapsed time
"""

import time
from collections import deque
from typing import Dict, Optional, Set
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.table import Table
from rich.text import Text


class GenerationProgress:
    """
    Live progress display for concurrent workflow generation.

    Inspired by Claude Code CLI's task list approach:
    - Shows active workers with spinners
    - Shows recent completions with status icons
    - Overall progress bar with elapsed time
    """

    def __init__(self, total: int, num_workers: int, console: Optional[Console] = None):
        self.total = total
        self.num_workers = num_workers
        self.console = console or Console()

        # Counters
        self.completed = 0
        self.failed = 0
        self.skipped = 0

        # Active workers: {endpoint_info: scenario_info}
        self.active: Dict[str, str] = {}

        # Recent events (scrolling log) - keep last 8
        self.recent: deque = deque(maxlen=8)

        # Timing
        self.start_time = time.time()

        # Rich progress bar
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating workflows"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("[green]{task.fields[success]}[/green]"),
            TextColumn("[red]{task.fields[fail]}[/red]"),
            TextColumn("[dim]{task.fields[skip]}[/dim]"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self._task_id = self._progress.add_task(
            "generate",
            total=total,
            success="",
            fail="",
            skip="",
        )

        # Live display
        self._live: Optional[Live] = None

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def worker_start(self, endpoint_info: str, scenario: str = "") -> None:
        """Mark a worker as active."""
        self.active[endpoint_info] = scenario
        self._refresh()

    def worker_scenario(self, endpoint_info: str, scenario: str) -> None:
        """Update which scenario a worker is processing."""
        if endpoint_info in self.active:
            self.active[endpoint_info] = scenario
            self._refresh()

    def worker_done(self, endpoint_info: str, success: bool = True, skipped_scenarios: int = 0) -> None:
        """Mark a worker as complete."""
        self.active.pop(endpoint_info, None)

        if success:
            self.completed += 1
            icon = "[green]done[/green]"
            self.recent.append(f"  [green]\\[done][/green] {endpoint_info}")
        else:
            self.failed += 1
            icon = "[red]fail[/red]"
            self.recent.append(f"  [red]\\[fail][/red] {endpoint_info}")

        self.skipped += skipped_scenarios

        # Update progress bar
        total_processed = self.completed + self.failed
        self._progress.update(
            self._task_id,
            completed=total_processed,
            success=f"done:{self.completed}" if self.completed else "",
            fail=f"fail:{self.failed}" if self.failed else "",
            skip=f"skip:{self.skipped}" if self.skipped else "",
        )
        self._refresh()

    def worker_skip(self, endpoint_info: str, reason: str = "") -> None:
        """Mark an endpoint as skipped entirely."""
        self.active.pop(endpoint_info, None)
        self.skipped += 1
        short_reason = reason[:40] if reason else "no testable scenarios"
        self.recent.append(f"  [dim]\\[skip][/dim] {endpoint_info} ({short_reason})")

        total_processed = self.completed + self.failed
        self._progress.update(
            self._task_id,
            completed=total_processed,
            skip=f"skip:{self.skipped}" if self.skipped else "",
        )
        self._refresh()

    def log_retry(self, endpoint_info: str, scenario: str, attempt: int, reason: str) -> None:
        """Log a retry event."""
        short_reason = reason[:60] if reason else ""
        self.recent.append(
            f"  [yellow]\\[retry][/yellow] {endpoint_info} {scenario} "
            f"(attempt {attempt}: {short_reason})"
        )
        self._refresh()

    def _build_display(self) -> Group:
        """Build the complete display renderable."""
        parts = []

        # Progress bar
        parts.append(self._progress)

        # Active workers
        if self.active:
            active_table = Table.grid(padding=(0, 1))
            active_table.add_column(style="cyan", width=4)
            active_table.add_column(min_width=40)
            active_table.add_column(style="dim")

            for endpoint, scenario in list(self.active.items())[:self.num_workers]:
                truncated = endpoint if len(endpoint) <= 45 else endpoint[:42] + "..."
                scenario_text = scenario if scenario else "starting..."
                active_table.add_row("  ->", truncated, scenario_text)

            parts.append(active_table)

        # Recent events
        if self.recent:
            recent_text = Text()
            for event in self.recent:
                recent_text.append_text(Text.from_markup(event + "\n"))
            parts.append(recent_text)

        return Group(*parts)

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_display())

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
