"""Load the local markdown knowledge base used by the RAG agent."""

from __future__ import annotations

from pathlib import Path

DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"


def load_knowledge_base(kb_dir: Path | str | None = None) -> dict[str, str]:
    """Load every `*.md` file in the knowledge base into a name → text map.

    Args:
        kb_dir: Directory to read from. Defaults to the project-level
            `knowledge_base/` folder.

    Returns:
        Mapping of filename (without path) to file contents. Empty if the
        directory does not exist — callers should treat that as a signal
        that retrieval is unavailable rather than as an error.
    """
    directory = Path(kb_dir) if kb_dir else DEFAULT_KB_DIR
    if not directory.exists():
        return {}

    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(directory.glob("*.md"))
    }
