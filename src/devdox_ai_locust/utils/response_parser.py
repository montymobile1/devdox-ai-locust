"""
Shared utilities for parsing and cleaning AI responses.

Provides reusable functions for extracting code from tagged AI responses,
cleaning markdown artifacts, and validating Python syntax.
"""

import re
import logging

logger = logging.getLogger(__name__)


def extract_code_from_response(response_text: str) -> str:
    """Extract code from ``<code>...</code>`` tags in an AI response.

    If no tags are found, or the content inside tags is too short
    (<= 10 chars), the full response is returned instead.
    """
    pattern = r"<code>(.*?)</code>"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if not matches:
        logger.warning("No <code> tags found, using full response")
        return response_text.strip()

    content = max(matches, key=len).strip()

    if not content or len(content) <= 10:
        logger.warning(
            f"Code in tags too short ({len(content)} chars), using full response"
        )
        return response_text.strip()

    logger.debug(f"Extracted {len(content)} chars from <code> tags")
    return str(content)


def clean_response(content: str) -> str:
    """Clean and normalise an AI response.

    Strips markdown code fences and removes leading/trailing
    explanatory text that is not valid Python.
    """
    # Remove markdown code blocks if present
    if content.startswith("```python") and content.endswith("```"):
        content = content[9:-3].strip()
    elif content.startswith("```") and content.endswith("```"):
        content = content[3:-3].strip()

    lines = content.split("\n")
    start_idx = 0
    end_idx = len(lines)

    # Find actual Python code start
    for i, line in enumerate(lines):
        if line.strip().startswith(
            ("import ", "from ", "class ", "def ", '"""', "'''")
        ):
            start_idx = i
            break

    # Find actual Python code end (remove trailing explanations)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if (
            line
            and not line.startswith("#")
            and not line.lower().startswith(("note:", "this", "the "))
        ):
            end_idx = i + 1
            break

    return "\n".join(lines[start_idx:end_idx])


def validate_python_code(code: str) -> bool:
    """Check if *code* is syntactically valid Python."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False
