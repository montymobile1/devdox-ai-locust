"""Targeted unit tests for :mod:`devdox_ai_locust.modular_generator`."""

import textwrap

from devdox_ai_locust.modular_generator import ModularGenerator


def _new_generator():
    """Create a generator instance without running __init__."""

    return object.__new__(ModularGenerator)


class TestModularGeneratorHelpers:
    def test_strip_xml_tags_removes_markup_and_fences(self):
        generator = _new_generator()
        raw = textwrap.dedent(
            """
            <methods>
            ```python
            def hello():
                pass
            ```
            </methods>
            """
        )

        cleaned = generator._strip_xml_tags(raw)

        assert "methods" not in cleaned
        assert "```" not in cleaned
        assert cleaned.startswith("def hello():")

    def test_normalize_indentation_resets_base_indent_and_preserves_body(self):
        generator = _new_generator()
        raw = textwrap.dedent(
            """
                @task
            def sample(self):
                if True:
                    return 1

            
            def already_aligned(self):
                return 2
            """
        ).strip("\n")

        normalized = generator._normalize_indentation(raw)
        lines = normalized.splitlines()

        assert lines[0] == "@task"
        assert lines[1] == "def sample(self):"
        assert "    if True:" in lines
        assert "        return 1" in lines
        assert "def already_aligned(self):" in lines
        assert lines[-1] == "    return 2"

