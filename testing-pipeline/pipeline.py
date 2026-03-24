#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py

Orchestrates the full test-case generation pipeline for one or more documents.
For each document it runs:
  1. nodetree_generate.py  (PDF → PageIndex tree → filtered entries + tag bank)
  2. testcase_generate.py  (filtered entries + tag bank → Q&A test cases)

Output is saved per document under:  <out_dir>/<document_stem>/

Usage:
  python pipeline.py --config pipeline_config.yaml --out_dir ./output

To run a subset of documents only:
  python pipeline.py --config pipeline_config.yaml --out_dir ./output --docs doc1 doc2

Config format (YAML or JSON):
  defaults:                          # applied to every document
    model: gpt-4o-mini
    industry: education
    base_questions_per_topic: 3
    variations: 2
    workers: 8
    max_pageindex_pages: 300
    debug: false
    import_flag: false

  documents:
    - input: /path/to/doc1.pdf       # only 'input' is required per document
    - input: /path/to/doc2.pdf
      industry: insurance            # overrides the default for this doc only
    - input: /path/to/doc3.pdf
      base_questions_per_topic: 5
      variations: 3
      import_flag: true
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---- Script paths (resolved relative to this file) ----
_PIPELINE_DIR  = Path(__file__).parent
_REPO_ROOT     = _PIPELINE_DIR.parent
_NODETREE_SCRIPT = _REPO_ROOT / "node_tree_generation" / "nodetree_generate.py"
_TESTCASE_SCRIPT = _REPO_ROOT / "test-case-creation"   / "testcase_generate.py"

# ---- Defaults ----
DEFAULT_ARGS: Dict[str, Any] = {
    "model":                    "gpt-4o-mini",
    "industry":                 "",
    "base_questions_per_topic": 3,
    "variations":               2,
    "workers":                  8,
    "max_pageindex_pages":      300,
    "debug":                    False,
    "import_flag":              False,
}


# ---------------- Config loader ----------------
def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise SystemExit(
                    "PyYAML is required for YAML configs.\n"
                    "Install it with: pip install pyyaml"
                )
            return yaml.safe_load(f) or {}
        return json.load(f)


def resolve_args(defaults: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
    """Merge: script defaults → config defaults → per-document overrides."""
    merged = {**DEFAULT_ARGS, **defaults}
    merged.update({k: v for k, v in doc.items() if k != "input"})
    return merged


# ---------------- Command builders ----------------
def _flag(args: Dict[str, Any], key: str) -> List[str]:
    return [f"--{key}"] if args.get(key) else []


def build_nodetree_cmd(args: Dict[str, Any], input_path: str, out_dir: str) -> List[str]:
    return [
        sys.executable, str(_NODETREE_SCRIPT),
        "--input",               input_path,
        "--out_dir",             out_dir,
        "--model",               str(args["model"]),
        "--industry",            str(args["industry"]),
        "--max_pageindex_pages", str(args["max_pageindex_pages"]),
        *_flag(args, "debug"),
    ]


def build_testcase_cmd(args: Dict[str, Any], out_dir: str) -> List[str]:
    return [
        sys.executable, str(_TESTCASE_SCRIPT),
        "--filtered_entries",        os.path.join(out_dir, "filtered_entries.json"),
        "--tag_bank",                os.path.join(out_dir, "tag_bank.json"),
        "--out_dir",                 out_dir,
        "--industry",                str(args["industry"]),
        "--model",                   str(args["model"]),
        "--base_questions_per_topic",str(args["base_questions_per_topic"]),
        "--variations",              str(args["variations"]),
        "--workers",                 str(args["workers"]),
        *_flag(args, "debug"),
        *_flag(args, "import_flag"),
    ]


# ---------------- Step runner ----------------
def run_step(cmd: List[str], label: str, doc_name: str) -> bool:
    print(f"\n[PIPELINE] {doc_name} — {label}")
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"[PIPELINE][ERROR] {label} failed for '{doc_name}' "
              f"(exit {result.returncode}, {elapsed:.1f}s)")
        return False
    print(f"[PIPELINE] {label} done ({elapsed:.1f}s)")
    return True


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Run the full nodetree → testcase pipeline for one or more documents."
    )
    ap.add_argument("--config",  required=True,
                    help="Path to pipeline_config.yaml (or .json)")
    ap.add_argument("--out_dir", required=True,
                    help="Root output directory. A subfolder is created per document.")
    ap.add_argument("--docs", nargs="*", metavar="DOC_STEM",
                    help="Run only these documents (filename stems, no extension). "
                         "Omit to run all documents in the config.")
    cli = ap.parse_args()

    config    = load_config(cli.config)
    defaults  = config.get("defaults", {})
    documents = config.get("documents", [])

    if not documents:
        raise SystemExit("No documents found in config.")

    # Optional subset filter
    if cli.docs:
        filter_set = {d.lower() for d in cli.docs}
        documents  = [d for d in documents if Path(d["input"]).stem.lower() in filter_set]
        if not documents:
            raise SystemExit(f"No documents matched: {cli.docs}")

    os.makedirs(cli.out_dir, exist_ok=True)

    results    = []
    total_start = time.time()

    for doc in documents:
        input_path = doc.get("input", "").strip()
        if not input_path:
            print("[PIPELINE][WARN] Skipping entry with no 'input' field.")
            continue

        doc_name    = Path(input_path).stem
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_out_dir = os.path.join(cli.out_dir, f"{timestamp}_{doc_name}")
        os.makedirs(doc_out_dir, exist_ok=True)

        merged = resolve_args(defaults, doc)

        print(f"\n{'='*60}")
        print(f"[PIPELINE] Document : {doc_name}")
        print(f"[PIPELINE] Input    : {input_path}")
        print(f"[PIPELINE] Output   : {doc_out_dir}")
        print(f"[PIPELINE] Args     : {merged}")
        print(f"{'='*60}")

        # Step 1 — nodetree_generate
        ok = run_step(
            build_nodetree_cmd(merged, input_path, doc_out_dir),
            "Step 1/2: nodetree_generate",
            doc_name
        )
        if not ok:
            results.append({"doc": doc_name, "status": "FAILED at Step 1 (nodetree_generate)"})
            continue

        # Step 2 — testcase_generate
        ok = run_step(
            build_testcase_cmd(merged, doc_out_dir),
            "Step 2/2: testcase_generate",
            doc_name
        )
        if not ok:
            results.append({"doc": doc_name, "status": "FAILED at Step 2 (testcase_generate)"})
            continue

        results.append({"doc": doc_name, "status": "OK", "out_dir": doc_out_dir})

    # ---- Summary ----
    total_elapsed = time.time() - total_start
    passed = sum(1 for r in results if r["status"] == "OK")
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"[PIPELINE] COMPLETE — {len(results)} doc(s) in {total_elapsed:.1f}s "
          f"({passed} passed, {failed} failed)")
    print(f"{'='*60}")
    for r in results:
        suffix = f" → {r['out_dir']}" if "out_dir" in r else ""
        print(f"  {'OK' if r['status'] == 'OK' else 'FAIL'}  {r['doc']}{suffix}")


if __name__ == "__main__":
    main()
