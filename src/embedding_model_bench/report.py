"""Render results as Markdown for the summaries under results/."""

from collections.abc import Iterable, Sequence


def markdown_table(
    headers: Sequence[str], rows: Iterable[Sequence[object]], footer: Sequence[str] = ()
) -> str:
    """Render a table, then any footer lines after a blank line.

    Returns:
        The Markdown, ending in a newline.
    """
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    if footer:
        lines += ["", *footer]
    return "\n".join(lines) + "\n"
