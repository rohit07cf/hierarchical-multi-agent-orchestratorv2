#!/usr/bin/env python3
"""Render the High-Level Design document (docs/hld.html) to PDF.

Usage:
    python docs/build_pdf.py            # docs/hld.html -> docs/HLD-Agentic-Customer-Support.pdf
    python docs/build_pdf.py in.html out.pdf

Requires WeasyPrint (see docs/requirements-docs.txt):
    pip install -r docs/requirements-docs.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
DEFAULT_SRC = DOCS / "hld.html"
DEFAULT_OUT = DOCS / "HLD-Agentic-Customer-Support.pdf"


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    try:
        from weasyprint import HTML
    except ImportError:
        print(
            "WeasyPrint is not installed. Run:\n"
            "    pip install -r docs/requirements-docs.txt",
            file=sys.stderr,
        )
        return 1

    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        return 1

    HTML(str(src)).write_pdf(str(out))
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
