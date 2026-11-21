#!/usr/bin/env python3
"""Build overleaf/bace20-overleaf.zip: the LaTeX sources and figures, flat, for Overleaf.

    python make_overleaf.py          # (re)write the zip
    python make_overleaf.py --check  # exit 1 if the committed zip is out of date (CI)

The zip is deterministic (sorted entries, fixed timestamps), so --check compares bytes.
Open it in Overleaf with the badge in README.md, or upload it via New Project -> Upload.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "overleaf" / "bace20-overleaf.zip"
FILES = ["bace20.tex", "refs.bib", "numbers.tex"]


def sources() -> list[Path]:
    figs = sorted(
        p
        for p in (ROOT / "figures").iterdir()
        if p.suffix == ".pdf" or p.name.startswith("lab_")
    )
    return [ROOT / f for f in FILES] + figs


def build() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sources():
            info = zipfile.ZipInfo(str(p.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())
    return buf.getvalue()


def main() -> None:
    data = build()
    if "--check" in sys.argv:
        if not OUT.is_file() or OUT.read_bytes() != data:
            print("overleaf/bace20-overleaf.zip is out of date: run python make_overleaf.py")
            sys.exit(1)
        print("overleaf zip up to date")
        return
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(data)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(data) / 1024:.0f} KB, {len(sources())} files)")


if __name__ == "__main__":
    main()
