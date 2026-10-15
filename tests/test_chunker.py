import pytest

from code_assistant.chunker import chunk_python


PATH = "ultralytics/engine/example.py"
SHA = "a" * 40


def test_decorated_method_keeps_qualified_name_and_original_source():
    source = (
        "import os\n"
        "\n"
        "class Outer:\n"
        "    class Inner:\n"
        "        @staticmethod\n"
        "        def run(x):\n"
        '            """Add one."""\n'
        "            # Offset the input.\n"
        "            return x + 1\n"
    )
    chunks = chunk_python(PATH, source, SHA)
    method = next(c for c in chunks if c.symbol_full == "Outer.Inner.run")
    assert (method.start_line, method.end_line) == (5, 9)
    assert method.content == "".join(source.splitlines(keepends=True)[4:9])
    assert any(c.symbol_type == "module" and "import os" in c.content for c in chunks)


def test_class_source_is_kept_around_methods_without_duplicate_lines():
    source = (
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class Derived(Base):\n"
        '    """Describe the class.\n'
        + "    Details that must remain citable.\n" * 80
        + '    """\n'
        "    threshold: float = 0.5\n"
        "    # Class setting, outside methods.\n"
        "    def run(self):\n"
        "        return self.threshold\n"
        "    alias = run\n"
        "    class Nested:\n"
        "        enabled = True\n"
    )
    chunks = chunk_python(PATH, source, SHA, max_tokens=240)
    lines = source.splitlines(keepends=True)
    covered = []
    for c in chunks:
        assert c.content == "".join(lines[c.start_line - 1:c.end_line])
        covered.extend(range(c.start_line, c.end_line + 1))
    assert len(covered) == len(set(covered))
    assert {n for n, line in enumerate(lines, 1) if line.strip()} <= set(covered)
    class_text = "".join(c.content for c in chunks if c.symbol_full == "Derived")
    assert "@dataclass\nclass Derived(Base):" in class_text
    assert "threshold: float = 0.5" in class_text and "alias = run" in class_text
    assert "def run(self)" not in class_text


def test_long_function_preserves_source_and_respects_counter_budget():
    source = (
        "def collect(ready):\n"
        "    values = []\n"
        "    if ready:\n"
        + "".join(f"        values.append({i})\n" for i in range(40))
        + "    else:\n"
        "        values.append(-1)\n"
        "    return values\n"
    )
    budget = 240
    chunks = chunk_python(PATH, source, SHA, max_tokens=budget, count_tokens=len)
    parts = [c for c in chunks if c.symbol_full == "collect"]
    lines = source.splitlines(keepends=True)
    assert len(parts) > 1
    assert "".join(c.content for c in parts) == source
    assert len({c.chunk_id for c in parts}) == len(parts)
    assert all(c.content == "".join(lines[c.start_line - 1:c.end_line]) for c in parts)
    assert all(len(c.embedding_text) <= budget for c in parts)


def test_syntax_error_fallback_preserves_source_in_bounded_windows():
    source = "def broken(:\n" + "    pass\n" * 44
    window = 20
    chunks = chunk_python(PATH, source, SHA, fallback_lines=window)
    lines = source.splitlines(keepends=True)
    assert len(chunks) > 1
    assert "".join(c.content for c in chunks) == source
    assert all(c.end_line - c.start_line + 1 <= window for c in chunks)
    assert all(c.symbol_type == "window" for c in chunks)
    assert all(c.content == "".join(lines[c.start_line - 1:c.end_line]) for c in chunks)


@pytest.mark.parametrize("prefix", ["", "def broken(:\n"])
def test_single_line_over_budget_is_rejected_with_file_and_line(prefix):
    source = prefix + 'VALUE = "' + 'x' * 500 + '"\n'
    line = 2 if prefix else 1
    with pytest.raises(ValueError, match=f"{PATH}:L{line}.*complete source line exceeds"):
        chunk_python(PATH, source, SHA, max_tokens=240, count_tokens=len)
