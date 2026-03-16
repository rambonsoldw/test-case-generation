#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nodetree_generate_pymupdf.py
PyMuPDF-only variant of nodetree_generate.py for local testing.
No PageIndex API calls — will not incur PageIndex charges.

- Extract PDF tree using PyMuPDFExtractor (requires embedded TOC)
- Flatten tree into entries
- Filter entries with quote length > 100 (these become topics)
- Run global LLM tagging to produce tag_bank.json
- Write out files:
    - <base>_pymupdf_tree.json
    - index_entries_flat.json
    - filtered_entries.json
    - tag_bank.json
    - document_ingestion_metrics.csv
"""
import os
import re
import json
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
from openai import OpenAI

try:
    from pymupdf_ext2 import PyMuPDFExtractor
    from base import TreeNode
except ImportError:
    raise SystemExit(
        "pymupdf_ext / base could not be imported.\n"
        "Ensure pymupdf_ext.py and base.py are on the Python path."
    )

# ---------------- OpenAI helpers ----------------
_client = None

def make_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.S|re.I)
    return m.group(1).strip() if m else s

def _extract_text_from_response(rsp: Any) -> str:
    t = getattr(rsp, "output_text", None)
    if isinstance(t, str) and t.strip(): return t
    out = getattr(rsp, "output", None)
    if out and isinstance(out, list):
        try:
            item = out[0]
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if content and isinstance(content, list):
                node = content[0]
                txt  = node.get("text") if isinstance(node, dict) else getattr(node, "text", None)
                if txt:
                    val = txt.get("value") if isinstance(txt, dict) else getattr(txt, "value", None)
                    if isinstance(val, str) and val.strip(): return val
        except Exception:
            pass
    choices = getattr(rsp, "choices", None)
    if choices and isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
        if msg:
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if isinstance(content, str) and content.strip(): return content
    return str(rsp)

def parse_json_from_resp(rsp: Any, debug: bool=False, tag: str="") -> Dict[str, Any]:
    txt = _extract_text_from_response(rsp)
    if debug:
        prev = (txt[:240] + "…") if len(txt) > 240 else txt
        print(f"[DEBUG] Raw LLM text ({tag}) first 240:\n{prev!r}\n")
    try:
        return json.loads(txt)
    except Exception:
        pass
    block = _strip_code_fences(txt)
    try:
        return json.loads(block)
    except Exception:
        pass
    s = block
    starts = [i for i,ch in enumerate(s) if ch in "{["]
    for st in starts:
        stack, in_str, esc = [], False, False
        for i in range(st, len(s)):
            ch = s[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
                continue
            if ch == '"': in_str = True
            elif ch in "{[": stack.append("}" if ch=="{" else "]")
            elif ch in "}]":
                if not stack: break
                want = stack.pop()
                if (ch == "}" and want != "}") or (ch == "]" and want != "]"): break
                if not stack:
                    cand = s[st:i+1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        break
    raise ValueError(f"No valid JSON found in response: {tag}")

def responses_create(client: OpenAI, **kwargs):
    try:
        return client.responses.create(**kwargs)
    except Exception:
        model = kwargs.get("model")
        inp   = kwargs.get("input", [])
        messages = [{"role": m.get("role","user"), "content": m.get("content","")} for m in inp]
        return client.chat.completions.create(model=model, messages=messages)

# ---------------- Flattener ----------------
def flatten_pageindex_root(pi_obj, max_items=None, default_source="UNKNOWN.json"):
    out = []

    def _to_int(v):
        if v is None: return None
        if isinstance(v, (int, float)): return int(v)
        if isinstance(v, str) and v.strip().lstrip("-").isdigit(): return int(v.strip())
        return None

    def _pick_start_index(node):
        for key in ("start_index", "start_page", "page_start", "page_index",
                    "original_page", "page_number", "page"):
            v = node.get(key) if isinstance(node, dict) else None
            iv = _to_int(v)
            if iv is not None: return iv
        return None

    def _pick_text(node):
        return (node.get("text") or node.get("content") or node.get("body") or "").strip()

    def _pick_title(node):
        return (node.get("title") or node.get("heading") or node.get("section") or node.get("name") or "(section)").strip()

    def _children(node):
        ch = node.get("children") or node.get("nodes") or node.get("items")
        return ch if isinstance(ch, list) else []

    def _emit(node, src, page):
        out.append({
            "page": page if page is not None else -1,
            "original_page": page if page is not None else -1,
            "section": _pick_title(node),
            "quote": _pick_text(node),
            "source_file": src
        })

    def _walk(node, src, inherited_page=None):
        node_page = _pick_start_index(node)
        page_to_use = node_page if node_page is not None else inherited_page
        _emit(node, src, page_to_use)
        for ch in _children(node):
            if isinstance(ch, dict):
                _walk(ch, src, page_to_use)

    def _doc_src(doc, fallback):
        return (doc.get("source_file") or doc.get("file_name") or doc.get("filename") or fallback)

    if isinstance(pi_obj, list):
        for doc in pi_obj:
            if isinstance(doc, dict):
                src = _doc_src(doc, default_source)
                root_inherited = _pick_start_index(doc)
                roots = doc.get("nodes") or doc.get("children")
                if isinstance(roots, list) and roots:
                    for r in roots:
                        if isinstance(r, dict):
                            _walk(r, src, inherited_page=root_inherited)
                else:
                    _walk(doc, src, inherited_page=root_inherited)
            elif isinstance(doc, str):
                out.append({"page": -1, "original_page": -1, "section": "(text)", "quote": doc.strip(), "source_file": default_source})
    elif isinstance(pi_obj, dict):
        src = _doc_src(pi_obj, default_source)
        root_inherited = _pick_start_index(pi_obj)
        roots = pi_obj.get("nodes") or pi_obj.get("children")
        if isinstance(roots, list) and roots:
            for r in roots:
                if isinstance(r, dict):
                    _walk(r, src, inherited_page=root_inherited)
        else:
            _walk(pi_obj, src, inherited_page=root_inherited)

    if max_items and len(out) > max_items:
        out = out[:max_items]
    return out

# ---------------- Document ingestion metrics ----------------
def compute_and_write_ingestion_metrics(
    out_dir: str,
    document_name: str,
    index_entries: List[Dict[str, Any]],
    filtered_entries: List[Dict[str, Any]]
):
    all_pages = {
        e.get("original_page")
        for e in index_entries
        if isinstance(e.get("original_page"), int) and e.get("original_page") >= 0
    }
    page_count = len(all_pages)

    def word_count(text: str) -> int:
        return len(text.split())

    original_word_count = sum(
        word_count(e.get("quote", ""))
        for e in index_entries
        if isinstance(e.get("quote"), str)
    )
    node_tree_word_count = sum(
        word_count(e.get("quote", ""))
        for e in filtered_entries
        if isinstance(e.get("quote"), str)
    )

    topic_count = len(filtered_entries)

    filtered_pages = {
        e.get("original_page")
        for e in filtered_entries
        if isinstance(e.get("original_page"), int) and e.get("original_page") >= 0
    }
    topic_coverage_ratio = len(filtered_pages) / page_count if page_count > 0 else 0.0
    compression_rate = node_tree_word_count / original_word_count if original_word_count > 0 else 0.0

    csv_path = os.path.join(out_dir, "document_ingestion_metrics.csv")
    fieldnames = [
        "document_name", "page_count", "original_word_count",
        "node_tree_word_count", "topic_count", "topic_coverage_ratio", "compression_rate"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "document_name": document_name,
            "page_count": page_count,
            "original_word_count": original_word_count,
            "node_tree_word_count": node_tree_word_count,
            "topic_count": topic_count,
            "topic_coverage_ratio": round(topic_coverage_ratio, 4),
            "compression_rate": round(compression_rate, 4)
        })
    print(f"[INFO] Wrote ingestion metrics CSV to: {csv_path}")

# ---------------- Tagging prompts ----------------
SYSTEM_TAG_GLOBAL = (
    "You are tagging a policy/handbook using ONLY the nodes provided.\n "
    "Output concise, consumer-friendly tags.\n "
    "Each tag is a (key, value) pair where the value is a 1–2 word canonical TOPIC.\n "
    "You MUST ground every tag in the excerpt text, but you MAY abstract wording\n "
    "to a canonical topic (e.g., 'paid annual leave' → value 'annual leave').\n "
    "Do NOT invent facts not supported by the excerpts. \n"
    "Be sure to tag each node tree section AT LEAST ONCE \n"
    "Example keys (adapt to this document): {example_keys}\n"
    "Example values (adapt to this document): {example_values}\n"
)

USER_TAG_GLOBAL_TEMPLATE = (
    "FACETS to use as keys (choose what fits): {example_keys}\n\n"
    "NODE TREE ENTRIES (JSON):\n{index_json}\n\n"
    "Return STRICT JSON exactly like the format below:\n"
    "{{\"candidates\":[{{"
    "\"key\":\"process\",\"value\":\"claims\",\"section\":\"Claims / Lodgement\",\"original_page\":12,"
    "\"source_file\":\"ABCD Handbook.pdf\",\"quote\":\"verbatim excerpt 220-480 chars grounding the tag\","
    "\"score\":0.86}}]}}"
)

# ---------------- Tagging function ----------------
def tag_document_global_with_llm(
    client: OpenAI,
    model: str,
    index_entries: List[Dict[str, Any]],
    industry: str,
    debug: bool=False
) -> List[Dict[str, Any]]:
    MAX_ENTRIES = 300
    MAX_QUOTE = 800
    compact_entries = []
    for e in index_entries[:MAX_ENTRIES]:
        compact_entries.append({
            "original_page": e.get("original_page", -1),
            "source_file": e.get("source_file", "document"),
            "section": (e.get("section") or "(section)")[:180],
            "quote": (e.get("quote") or "")[:MAX_QUOTE]
        })

    index_json = json.dumps(compact_entries, ensure_ascii=False)

    if "insurance" in industry:
        example_keys = "coverage,exclusions,claims_process,premiums,deductibles,limitations"
        example_values = "comprehensive_motor_vehicle,windscreen_replacement,third_party_liability"
    elif "medical" in industry or "health" in industry:
        example_keys = "diagnosis,treatment,medication,procedures,symptoms,contraindications"
        example_values = "acute_conditions,chronic_disease_management,dosage_guidelines"
    elif "math" in industry or "scientific" in industry:
        example_keys = "theorems,formulas,proofs,definitions,applications,examples"
        example_values = "pythagorean_theorem,quadratic_equations,statistical_methods"
    elif "legal" in industry:
        example_keys = "rights,obligations,definitions,procedures,requirements,remedies"
        example_values = "statutory_rights,contractual_obligations,filing_procedures"
    elif "technical" in industry or "software" in industry or "engineer" in industry:
        example_keys = "features,configuration,api_methods,requirements,troubleshooting,security"
        example_values = "authentication_methods,data_structures,error_handling"
    else:
        example_keys = "main_topics,concepts,procedures,definitions,guidelines,requirements"
        example_values = "specific_subtopics,detailed_aspects,particular_cases"

    rsp = responses_create(
        client,
        model=model,
        input=[
            {"role":"system","content":SYSTEM_TAG_GLOBAL.format(example_keys=example_keys, example_values=example_values)},
            {"role":"user","content":USER_TAG_GLOBAL_TEMPLATE.format(example_keys=example_keys, index_json=index_json)}
        ],
        response_format={"type":"json_object"},
        max_output_tokens=8192,
    )
    data = parse_json_from_resp(rsp, debug=debug, tag="tag_global")

    out: List[Dict[str, Any]] = []
    for c in (data.get("candidates") or []):
        key = (c.get("key") or "").strip().lower()
        val = (c.get("value") or "").strip().lower()
        sec = (c.get("section") or "").strip()
        pg  = c.get("original_page", -1)
        src = (c.get("source_file") or "document").strip()
        quo = (c.get("quote") or "").strip()
        try:
            sc = float(c.get("score", 0.0))
        except Exception:
            sc = 0.0
        if not key or not val:
            continue
        out.append({
            "page": pg,
            "source_file": src,
            "section": sec,
            "quote": quo,
            "candidates": [{"key": key, "value": val, "score": sc}]
        })
    return out

# ---------------- TreeNode converter ----------------
def treenode_to_pageindex_dict(node: TreeNode, source_file: str) -> Dict[str, Any]:
    out = {
        "title": node.title,
        "page_start": node.page_start,
        "page_end": node.page_end,
        "text": node.content,
        "source_file": source_file,
        "children": []
    }
    for child in node.children:
        out["children"].append(treenode_to_pageindex_dict(child, source_file))
    return out

# ---------------- PyMuPDF extraction ----------------
def extract_with_pymupdf(pdf_path: str, out_dir: str, debug: bool=False) -> Tuple[str, Any]:
    print("[INFO] Running PyMuPDF extractor...")
    extractor = PyMuPDFExtractor(verbose=debug)
    result = extractor.run(Path(pdf_path))

    if result.error:
        raise SystemExit(f"PyMuPDF extraction failed: {result.error}")

    source_file = os.path.basename(pdf_path)
    pi_like_dict = treenode_to_pageindex_dict(result.tree, source_file)

    base_name = os.path.splitext(source_file)[0]
    out_json_path = os.path.join(out_dir, f"{base_name}_pymupdf_tree.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(pi_like_dict, f, ensure_ascii=False, indent=2)

    print(f"[INFO] PyMuPDF tree written to: {out_json_path}")
    return out_json_path, pi_like_dict

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="PyMuPDF-only document preprocess (no PageIndex)")
    ap.add_argument("--input", required=True, help="Path to PDF or existing tree JSON")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--industry", default="")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    input_path = args.input
    ext_lower = os.path.splitext(input_path)[1].lower()

    if ext_lower == ".pdf":
        pi_path, pi_tree = extract_with_pymupdf(input_path, args.out_dir, args.debug)
    elif ext_lower == ".json":
        pi_path = input_path
        with open(input_path, "r", encoding="utf-8") as f:
            pi_tree = json.load(f)
        print(f"[INFO] Loaded existing tree JSON: {input_path}")
    else:
        raise SystemExit(f"Unsupported input extension: {ext_lower!r}. Expected .pdf or .json")

    client = make_client()

    default_src = os.path.splitext(os.path.basename(pi_path))[0] + ".json"
    index_entries = flatten_pageindex_root(pi_tree, max_items=None, default_source=default_src)

    out_flat_path = os.path.join(args.out_dir, "index_entries_flat.json")
    with open(out_flat_path, "w", encoding="utf-8") as f:
        json.dump(index_entries, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote flattened entries to: {out_flat_path} (count={len(index_entries)})")

    filtered = [e for e in index_entries if isinstance(e.get("quote", ""), str) and len(e.get("quote", "")) > 100]
    out_filtered_path = os.path.join(args.out_dir, "filtered_entries.json")
    with open(out_filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote filtered entries (len>100) to: {out_filtered_path} (count={len(filtered)})")

    if not filtered:
        print("[WARN] No filtered entries (quote length > 100). Tag bank will be empty.")

    tag_bank = tag_document_global_with_llm(client, args.model, filtered, industry=args.industry, debug=args.debug) if filtered else []
    out_tag_path = os.path.join(args.out_dir, "tag_bank.json")
    with open(out_tag_path, "w", encoding="utf-8") as f:
        json.dump(tag_bank, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote tag bank to: {out_tag_path} (entries={len(tag_bank)})")

    compute_and_write_ingestion_metrics(
        out_dir=args.out_dir,
        document_name=os.path.basename(input_path),
        index_entries=index_entries,
        filtered_entries=filtered
    )

    print("[DONE] nodetree_generate_pymupdf completed.")

if __name__ == "__main__":
    main()
