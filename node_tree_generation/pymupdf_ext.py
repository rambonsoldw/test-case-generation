"""
PyMuPDF TOC extractor — fixed to extract section-specific text rather than
repeating the full page text for every node on the same page.

Fix: when consecutive TOC entries share the same page, _extract_content now
slices the page text between heading boundaries instead of returning the full
page for every node.
"""

from __future__ import annotations

import re
from pathlib import Path

from base import BaseExtractor, ExtractionResult, TreeNode


class PyMuPDFExtractor(BaseExtractor):
    """
    Extract the embedded TOC from PDF metadata using PyMuPDF, with section-
    specific text extraction for same-page siblings.
    """

    name = "pymupdf"

    def extract(self, pdf_path: Path) -> TreeNode:
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

        self.log(f"Opening {pdf_path.name} with PyMuPDF...")

        doc = fitz.open(str(pdf_path))
        toc = doc.get_toc()

        if not toc:
            self.log("No embedded TOC found in PDF metadata")
            return self._fallback_flat(doc, pdf_path)

        self.log(f"Found embedded TOC with {len(toc)} entries")

        root = TreeNode(id="0", title=pdf_path.stem, level=0)
        stack: list[TreeNode] = [root]

        for i, entry in enumerate(toc):
            level      = entry[0]
            title      = entry[1]
            page_num   = entry[2] if len(entry) > 2 else None

            next_entry = toc[i + 1] if i + 1 < len(toc) else None
            next_page  = next_entry[2] if next_entry and len(next_entry) > 2 else None
            next_title = next_entry[1] if next_entry else None

            content = self._extract_content(
                doc, page_num, next_page,
                current_title=title, next_title=next_title
            )

            node = TreeNode(
                id=f"{i + 1}",
                title=title,
                level=level,
                page_start=page_num,
                page_end=(next_page - 1) if next_page and next_page > page_num else page_num,
                content=content,
            )

            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()

            stack[-1].children.append(node)
            stack.append(node)

        doc.close()
        self._merge_list_items(root)
        self._assign_ids(root)
        return root

    def _extract_content(
        self,
        doc,
        start_page: int | None,
        next_page: int | None,
        current_title: str | None = None,
        next_title: str | None = None,
    ) -> str:
        """
        Extract text for a section that runs from start_page up to (but not
        including) next_page.

        Four cases:
          1. No start page                  → ""
          2. Single page, no next section   → full page text
          3. Single page, next on same page → slice between headings
          4. Multi-page                     → slice first page from heading
                                              onward, include middle pages fully,
                                              slice last page up to next heading
                                              (last page == next_page, shared with
                                              the following section)
        """
        if start_page is None:
            return ""

        start_idx = max(0, min(start_page - 1, len(doc) - 1))

        # next_page is where the NEXT section begins (1-indexed).
        # The last page we need to read is next_page itself (to capture the
        # tail of this section before the next heading).  When next_page is
        # None (last TOC entry) we only read start_page.
        if next_page is None:
            last_idx = start_idx
        else:
            last_idx = max(start_idx, min(next_page - 1, len(doc) - 1))

        # ── Case: single page ────────────────────────────────────────────────
        if start_idx == last_idx:
            text = doc[start_idx].get_text().strip()
            if current_title and next_title and text:
                return self._slice_between_headings(text, current_title, next_title)
            return text

        # ── Case: multi-page ─────────────────────────────────────────────────
        parts = []

        # First page: trim anything above this section's heading.
        first_text = doc[start_idx].get_text().strip()
        if first_text:
            sliced = self._slice_from_heading(first_text, current_title) if current_title else ""
            parts.append(sliced if sliced else first_text)

        # Middle pages: fully owned by this section.
        for page_idx in range(start_idx + 1, last_idx):
            text = doc[page_idx].get_text().strip()
            if text:
                parts.append(text)

        # Last page: shared with the next section — trim from the next heading onward.
        last_text = doc[last_idx].get_text().strip()
        if last_text:
            sliced = self._slice_up_to_heading(last_text, next_title) if next_title else last_text
            if sliced:
                parts.append(sliced)

        return "\n\n".join(parts)

    # ── List-item merging ────────────────────────────────────────────────────

    @staticmethod
    def _is_list_item(title: str) -> bool:
        """Return True if the title starts with a list-item marker.

        Matches:
          (a)   (b)   (iv)   (1)   — letter/number in parentheses
          i.    ii.   iii.   iv.   — roman numerals with dot
          1.    2.    3.           — plain numbers with dot
        """
        return bool(re.match(
            r'^(\([a-zA-Z0-9]{1,4}\)|[ivxlcdmIVXLCDM]{1,6}\.|[0-9]{1,3}\.)(\s|$)',
            title.strip()
        ))

    def _merge_list_items(self, node: TreeNode) -> None:
        """
        Recursively merge child nodes that look like enumerated list items
        (e.g. (a), (b), (iii)) into their parent node's content instead of
        keeping them as separate children.
        """
        real_children = []
        for child in node.children:
            self._merge_list_items(child)
            if self._is_list_item(child.title) and not child.children:
                extra = f"{child.title} {child.content}".strip() if child.content else child.title
                node.content = (node.content + "\n" + extra).strip() if node.content else extra
            else:
                real_children.append(child)
        node.children = real_children

    # ── Text-slicing helpers ─────────────────────────────────────────────────

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def _find_title(self, text: str, title: str) -> tuple[int, int]:
        """Return (start_pos, key_len) of the best match for title in text."""
        key = self._norm(title)
        for length in (60, 40, 20):
            k = key[:length]
            idx = text.find(k)
            if idx != -1:
                return idx, len(k)
        return -1, 0

    def _slice_between_headings(
        self, page_text: str, current_title: str, next_title: str
    ) -> str:
        """Text on a single page between current_title and next_title."""
        norm = self._norm(page_text)
        start_pos, key_len = self._find_title(norm, current_title)
        if start_pos == -1:
            return ""
        content_start = start_pos + key_len
        next_pos, _ = self._find_title(norm[content_start:], next_title)
        if next_pos == -1:
            return norm[content_start:].strip()
        return norm[content_start: content_start + next_pos].strip()

    def _slice_from_heading(self, page_text: str, current_title: str) -> str:
        """Text from current_title's heading to the end of the page."""
        norm = self._norm(page_text)
        start_pos, key_len = self._find_title(norm, current_title)
        if start_pos == -1:
            return ""
        return norm[start_pos + key_len:].strip()

    def _slice_up_to_heading(self, page_text: str, next_title: str) -> str:
        """Text from the start of the page up to (not including) next_title."""
        norm = self._norm(page_text)
        next_pos, _ = self._find_title(norm, next_title)
        if next_pos == -1:
            return norm.strip()
        return norm[:next_pos].strip()

    def _fallback_flat(self, doc, pdf_path: Path) -> TreeNode:
        root = TreeNode(id="0", title=pdf_path.stem, level=0)
        max_pages = self.max_pages or len(doc)
        for page_idx in range(min(len(doc), max_pages)):
            text = doc[page_idx].get_text().strip()
            if text:
                root.children.append(TreeNode(
                    id=str(page_idx + 1),
                    title=f"Page {page_idx + 1}",
                    level=1,
                    page_start=page_idx + 1,
                    content=text[:5000],
                ))
        doc.close()
        return root

    def _assign_ids(self, node: TreeNode, prefix: str = "") -> None:
        for i, child in enumerate(node.children, 1):
            child.id = f"{prefix}{i}" if prefix == "" else f"{prefix}.{i}"
            self._assign_ids(child, child.id)
