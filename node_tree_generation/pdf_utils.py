"""
PDF text extraction helpers using PyMuPDF (fitz).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


def extract_pages(pdf_path: Path, max_pages: Optional[int] = None) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Returns a list of dicts:
        [{"page": 1, "text": "...", "char_count": 1234}, ...]
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    limit = max_pages or doc.page_count
    for i in range(min(limit, doc.page_count)):
        page = doc[i]
        text = page.get_text("text")
        pages.append({
            "page": i + 1,
            "text": text,
            "char_count": len(text),
        })
    doc.close()
    return pages


def extract_full_text(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    """Return the full text of the PDF as a single string."""
    pages = extract_pages(pdf_path, max_pages)
    return "\n\n".join(p["text"] for p in pages)


def page_count(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    count = doc.page_count
    doc.close()
    return count
