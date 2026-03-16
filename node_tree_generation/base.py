"""
Base extractor class and normalised tree node model.

Every extractor must convert its native output into the common TreeNode
structure so that comparison is apples-to-apples.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TreeNode:
    """Normalised hierarchical node — the common output format."""

    id: str = ""
    title: str = ""
    level: int = 0
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    content: str = ""  # Section text content (equivalent to 'quote' in reference)
    summary: str = ""  # LLM-generated summary
    children: list["TreeNode"] = field(default_factory=list)

    # ── helpers ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        # strip empty optional fields for cleaner JSON
        if not d["content"]:
            del d["content"]
        if not d["summary"]:
            del d["summary"]
        if d["page_start"] is None:
            del d["page_start"]
        if d["page_end"] is None:
            del d["page_end"]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TreeNode":
        """Reconstruct a TreeNode from a dictionary (e.g., loaded from JSON)."""
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(
            id=d.get("id", ""),
            title=d.get("title", ""),
            level=d.get("level", 0),
            page_start=d.get("page_start"),
            page_end=d.get("page_end"),
            content=d.get("content", ""),
            summary=d.get("summary", ""),
            children=children,
        )

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)

    def max_depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.max_depth() for c in self.children)

    def pretty(self, indent: int = 0) -> str:
        """Render a text-based tree for quick visual comparison."""
        prefix = "  " * indent
        page_info = ""
        if self.page_start is not None:
            page_info = f"  [pp. {self.page_start}"
            if self.page_end and self.page_end != self.page_start:
                page_info += f"-{self.page_end}"
            page_info += "]"
        lines = [f"{prefix}{'├─ ' if indent else ''}{self.id} {self.title}{page_info}"]
        for child in self.children:
            lines.append(child.pretty(indent + 1))
        return "\n".join(lines)


@dataclass
class ExtractionResult:
    """Wraps the tree plus timing / metadata."""

    extractor_name: str
    tree: TreeNode
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    raw_output: Optional[dict] = None  # original tool output for debugging


class BaseExtractor(ABC):
    """
    Abstract base for all extractors.

    Subclasses implement `extract()` which returns a TreeNode root.
    The runner calls `run()` which wraps extract() with timing + error handling.
    """

    name: str = "base"

    def __init__(self, model: str = "gpt-4o", verbose: bool = False, max_pages: int | None = None):
        self.model = model
        self.verbose = verbose
        self.max_pages = max_pages

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [{self.name}] {msg}")

    @abstractmethod
    def extract(self, pdf_path: Path) -> TreeNode:
        """
        Parse the PDF and return a normalised TreeNode root.
        Raise on failure — the runner will catch it.
        """
        ...

    def run(self, pdf_path: Path) -> ExtractionResult:
        """Execute the extraction with timing and error handling."""
        t0 = time.perf_counter()
        try:
            tree = self.extract(pdf_path)
            elapsed = time.perf_counter() - t0
            return ExtractionResult(
                extractor_name=self.name,
                tree=tree,
                elapsed_seconds=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return ExtractionResult(
                extractor_name=self.name,
                tree=TreeNode(id="0", title="(extraction failed)"),
                elapsed_seconds=round(elapsed, 2),
                error=str(exc),
            )
