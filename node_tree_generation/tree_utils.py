"""
Tree comparison, statistics, and rendering utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from extractors.base import TreeNode, ExtractionResult


def tree_stats(tree: TreeNode) -> dict:
    """Compute summary statistics for a tree."""
    all_nodes = []
    _collect_nodes(tree, all_nodes)

    levels = [n.level for n in all_nodes]
    leaf_count = sum(1 for n in all_nodes if not n.children)

    return {
        "total_nodes": len(all_nodes),
        "max_depth": tree.max_depth(),
        "leaf_nodes": leaf_count,
        "branch_nodes": len(all_nodes) - leaf_count,
        "level_distribution": _level_dist(levels),
        "avg_children": round(
            sum(len(n.children) for n in all_nodes if n.children)
            / max(1, sum(1 for n in all_nodes if n.children)),
            2,
        ),
    }


def _collect_nodes(node: TreeNode, acc: list[TreeNode]) -> None:
    acc.append(node)
    for child in node.children:
        _collect_nodes(child, acc)


def _level_dist(levels: list[int]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for lvl in levels:
        key = f"level_{lvl}"
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items()))


def comparison_summary(results: list[ExtractionResult]) -> dict:
    """Build a side-by-side comparison summary."""
    summary = {
        "extractors": {},
    }
    for r in results:
        stats = tree_stats(r.tree)
        summary["extractors"][r.extractor_name] = {
            "elapsed_seconds": r.elapsed_seconds,
            "error": r.error,
            **stats,
        }
    return summary


def save_tree(result: ExtractionResult, output_dir: Path) -> Path:
    """Save a single extractor's tree as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result.extractor_name}_tree.json"
    out_path = output_dir / filename

    payload = {
        "extractor": result.extractor_name,
        "elapsed_seconds": result.elapsed_seconds,
        "error": result.error,
        "tree": result.tree.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path


def save_comparison(results: list[ExtractionResult], output_dir: Path) -> Path:
    """Save the comparison summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comparison_summary.json"
    summary = comparison_summary(results)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out_path


def load_results_from_dir(output_dir: Path) -> list[ExtractionResult]:
    """
    Load all *_tree.json files from a directory and convert them to ExtractionResults.

    This enables comparing results from previous runs without re-running extractors.
    """
    results = []

    for json_file in sorted(output_dir.glob("*_tree.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))

            # Reconstruct the TreeNode from the dict
            tree = TreeNode.from_dict(data["tree"])

            result = ExtractionResult(
                extractor_name=data["extractor"],
                tree=tree,
                elapsed_seconds=data.get("elapsed_seconds", 0.0),
                error=data.get("error"),
            )
            results.append(result)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Could not load {json_file.name}: {e}")

    return results


def print_comparison_table(results: list[ExtractionResult]) -> None:
    """Print a rich table comparing results (falls back to plain text)."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Document Hierarchy Extraction — Comparison")

        table.add_column("Extractor", style="bold cyan")
        table.add_column("Time (s)", justify="right")
        table.add_column("Nodes", justify="right")
        table.add_column("Depth", justify="right")
        table.add_column("Leaves", justify="right")
        table.add_column("Branches", justify="right")
        table.add_column("Avg Children", justify="right")
        table.add_column("Status")

        for r in results:
            stats = tree_stats(r.tree)
            status = "✓" if not r.error else f"✗ {r.error[:40]}"
            table.add_row(
                r.extractor_name,
                str(r.elapsed_seconds),
                str(stats["total_nodes"]),
                str(stats["max_depth"]),
                str(stats["leaf_nodes"]),
                str(stats["branch_nodes"]),
                str(stats["avg_children"]),
                status,
            )

        console.print(table)

    except ImportError:
        # fallback plain text
        print("\n=== Comparison ===")
        for r in results:
            stats = tree_stats(r.tree)
            print(
                f"  {r.extractor_name:15s}  "
                f"time={r.elapsed_seconds:6.1f}s  "
                f"nodes={stats['total_nodes']:4d}  "
                f"depth={stats['max_depth']:2d}  "
                f"{'ERROR: ' + r.error if r.error else 'OK'}"
            )
        print()
