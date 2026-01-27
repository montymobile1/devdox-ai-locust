"""
Locust Log Analyzer

Reads a Locust .log file and produces a reduced .log file containing only
unique errors and exceptions (deduplicated), with their request/response
context preserved. Similar issues across different APIs are grouped together.

Efficient streaming approach for handling millions of lines.

Usage:
    python -m devdox_ai_locust.log_analyzer /path/to/output_locust_run.log
    # or
    devdox_ai_locust_analyze /path/to/output_locust_run.log
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, TextIO

from devdox_ai_locust.utils.constants import (
    ADDR_RE as _ADDR_RE,
    API_PATH_RE as _API_PATH_RE,
    CONTEXT_BUFFER_SIZE as _CONTEXT_BUFFER_SIZE,
    CONTEXT_PATTERN as _CONTEXT_PATTERN,
    ERROR_PATTERN as _ERROR_PATTERN,
    LONG_VALUE_RE as _LONG_VALUE_RE,
    NUMERIC_PATH_RE as _NUMERIC_PATH_RE,
    TIMESTAMP_RE as _TIMESTAMP_RE,
    TRACEBACK_START as _TRACEBACK_START,
    UUID_RE as _UUID_RE,
)


def _normalize_error(line: str) -> str:
    """Normalize an error line for deduplication (grouping key only)."""
    line = _TIMESTAMP_RE.sub("", line)
    line = _UUID_RE.sub("<UUID>", line)
    line = _NUMERIC_PATH_RE.sub("/<ID>", line)
    line = _ADDR_RE.sub("<ADDR>", line)
    # Remove specific API paths to group same error across different endpoints
    line = re.sub(r"(GET|POST|PUT|PATCH|DELETE)\s+/\S+", r"\1 <PATH>", line)
    return line.strip()


def _normalize_exception(line: str) -> str:
    """Normalize an exception line for deduplication (grouping key only)."""
    line = _UUID_RE.sub("<UUID>", line)
    line = _NUMERIC_PATH_RE.sub("/<ID>", line)
    line = _LONG_VALUE_RE.sub("'<LONG_VALUE>'", line)
    return line.strip()


def _extract_api_path(lines: List[str]) -> str:
    """Extract the API path from context/error lines."""
    for line in lines:
        m = _API_PATH_RE.search(line)
        if m:
            return f"{m.group(1)} {m.group(2)}"
    return ""


def analyze_log(log_path: str, output_path: Optional[str] = None) -> Dict[str, int]:
    """
    Stream-parse a Locust .log file and write a reduced .log with unique errors.

    Preserves original error lines as-is, includes request/response context,
    and groups similar errors from different APIs.

    Args:
        log_path: Path to the input .log file
        output_path: Path for the reduced output file. Defaults to <name>_reduced.log

    Returns:
        Dict with stats: total_lines, unique_errors, unique_exceptions, duplicates_removed
    """
    src = Path(log_path)
    if not src.exists():
        print(f"Error: File not found: {log_path}")
        sys.exit(1)

    if output_path is None:
        output_path = str(src.with_suffix("")) + "_reduced.log"

    # Track signatures
    error_signatures: Counter = Counter()
    exception_signatures: Counter = Counter()

    # First-seen samples: signature -> (context_lines + error_line)
    error_samples: Dict[str, List[str]] = {}
    exception_samples: Dict[str, List[str]] = {}

    # All API paths that hit each signature
    error_apis: Dict[str, set] = defaultdict(set)
    exception_apis: Dict[str, set] = defaultdict(set)

    total_lines = 0
    duplicates_removed = 0

    # Rolling context buffer (recent lines before an error)
    context_buffer: List[str] = []

    with open(src, "r", encoding="utf-8", errors="replace") as f:
        line_iter = _line_iterator(f)
        for line in line_iter:
            total_lines += 1

            # Detect traceback blocks
            if _TRACEBACK_START.match(line):
                tb_dup = _collect_traceback(
                    line,
                    line_iter,
                    context_buffer,
                    exception_signatures,
                    exception_samples,
                    exception_apis,
                )
                total_lines += tb_dup[0]
                duplicates_removed += tb_dup[1]
                context_buffer.clear()
                continue

            # Detect error lines
            if _ERROR_PATTERN.search(line):
                duplicates_removed += _record_error(
                    line,
                    context_buffer,
                    error_signatures,
                    error_samples,
                    error_apis,
                )
                context_buffer.clear()
                continue

            # Maintain rolling context buffer
            context_buffer.append(line)
            if len(context_buffer) > _CONTEXT_BUFFER_SIZE:
                context_buffer.pop(0)

    # Write reduced log
    _write_reduced_log(
        output_path,
        error_signatures,
        error_samples,
        error_apis,
        exception_signatures,
        exception_samples,
        exception_apis,
        total_lines,
        duplicates_removed,
    )

    return {
        "total_lines": total_lines,
        "unique_errors": len(error_signatures),
        "unique_exceptions": len(exception_signatures),
        "duplicates_removed": duplicates_removed,
    }


def _relevant_context(context_buffer: List[str]) -> List[str]:
    """Filter context buffer to lines relevant to errors."""
    return [line for line in context_buffer if _CONTEXT_PATTERN.search(line)]


def _collect_traceback(
    first_line: str,
    line_iter: Iterator[str],
    context_buffer: List[str],
    exception_signatures: Counter,
    exception_samples: Dict[str, List[str]],
    exception_apis: Dict[str, set],
) -> tuple:
    """Collect a traceback block and record the exception. Returns (extra_lines, duplicates)."""
    extra_lines = 0
    duplicates = 0
    tb_lines = [first_line]
    for next_line in line_iter:
        extra_lines += 1
        if next_line.startswith((" ", "\t")) or not next_line.strip():
            tb_lines.append(next_line)
        else:
            tb_lines.append(next_line)
            sig = _normalize_exception(next_line)
            exception_signatures[sig] += 1
            api = _extract_api_path(context_buffer + tb_lines)
            if api:
                exception_apis[sig].add(api)
            if sig not in exception_samples:
                exception_samples[sig] = _relevant_context(context_buffer) + tb_lines
            else:
                duplicates = len(tb_lines)
            break
    return extra_lines, duplicates


def _record_error(
    line: str,
    context_buffer: List[str],
    error_signatures: Counter,
    error_samples: Dict[str, List[str]],
    error_apis: Dict[str, set],
) -> int:
    """Record an error line. Returns number of duplicates (0 or 1)."""
    sig = _normalize_error(line)
    error_signatures[sig] += 1
    api = _extract_api_path(context_buffer + [line])
    if api:
        error_apis[sig].add(api)
    if sig not in error_samples:
        error_samples[sig] = _relevant_context(context_buffer) + [line]
        return 0
    return 1


def _line_iterator(f: TextIO) -> Iterator[str]:
    """Yield lines from file, stripping newlines."""
    for line in f:
        yield line.rstrip("\n\r")


def _write_reduced_log(
    output_path: str,
    error_signatures: "Counter[str]",
    error_samples: Dict[str, List[str]],
    error_apis: Dict[str, set],
    exception_signatures: "Counter[str]",
    exception_samples: Dict[str, List[str]],
    exception_apis: Dict[str, set],
    total_lines: int,
    duplicates_removed: int,
) -> None:
    """Write the reduced log file with unique errors/exceptions and their context."""
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("# Reduced Locust Log\n")
        out.write(
            f"# Original: {total_lines} lines | Duplicates removed: {duplicates_removed}\n"
        )
        out.write(
            f"# Unique errors: {len(error_signatures)} | Unique exceptions: {len(exception_signatures)}\n"
        )
        out.write(f"#{'=' * 69}\n\n")

        # Exceptions (most important)
        if exception_signatures:
            out.write(
                f"# {'=' * 30} EXCEPTIONS ({len(exception_signatures)}) {'=' * 30}\n\n"
            )
            for sig, count in exception_signatures.most_common():
                sample_lines = exception_samples.get(sig, [])
                apis = exception_apis.get(sig, set())

                out.write(f"# [{count}x]")
                if apis:
                    out.write(f" Affected APIs: {', '.join(sorted(apis))}")
                out.write("\n")

                for sample_line in sample_lines:
                    out.write(f"{sample_line}\n")
                out.write("\n")

        # Error lines
        if error_signatures:
            out.write(f"# {'=' * 30} ERRORS ({len(error_signatures)}) {'=' * 30}\n\n")
            for sig, count in error_signatures.most_common():
                sample_lines = error_samples.get(sig, [])
                apis = error_apis.get(sig, set())

                out.write(f"# [{count}x]")
                if apis:
                    out.write(f" Affected APIs: {', '.join(sorted(apis))}")
                out.write("\n")

                for sample_line in sample_lines:
                    out.write(f"{sample_line}\n")
                out.write("\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: devdox_ai_locust_analyze <log_file_path> [output_path]")
        print(
            "       python -m devdox_ai_locust.log_analyzer <log_file_path> [output_path]"
        )
        sys.exit(1)

    log_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    stats = analyze_log(log_path, output_path)

    src = Path(log_path)
    out_name = output_path or (str(src.with_suffix("")) + "_reduced.log")
    print(f"Original:  {stats['total_lines']} lines")
    print(
        f"Unique:    {stats['unique_errors']} errors, {stats['unique_exceptions']} exceptions"
    )
    print(f"Removed:   {stats['duplicates_removed']} duplicate lines")
    print(f"Output:    {out_name}")


if __name__ == "__main__":
    main()
