#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
document_preprocess.py
- Convert PDF -> PageIndex tree (if input is PDF) or load existing PageIndex JSON
- Flatten the PageIndex tree into entries
- Filter entries with quote length > 100 (these become topics)
- Run global LLM tagging to produce tag_bank.json
- Write out files:
    - <base>_pageindex_tree.json (if PDF input)
    - index_entries_flat.json
    - filtered_entries.json
    - tag_bank.json
"""
import os
import re
import json
import argparse
import time
import csv
from typing import Any, Dict, List, Tuple
# minimal deps used by tagging (no heavy imports here)
from openai import OpenAI

# PageIndex optional SDK (used only if PDF conversion requested)
try:
    from pageindex import PageIndexClient  # type: ignore
except ImportError:
    PageIndexClient = None

from pathlib import Path

# PyMuPDF extractor
try:
    from pymupdf_ext import PyMuPDFExtractor
    from base import TreeNode
except ImportError:
    PyMuPDFExtractor = None
    TreeNode = None

# ---------------- OpenAI helpers (copied from original) ----------------
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
    # try direct
    try:
        return json.loads(txt)
    except Exception:
        pass
    # fenced
    block = _strip_code_fences(txt)
    try:
        return json.loads(block)
    except Exception:
        pass
    # scan for JSON block
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
    """
    Minimal wrapper: call `client.responses.create` or fallback to chat completions.
    This mirrors the original script's behavior.
    """
    try:
        return client.responses.create(**kwargs)
    except Exception:
        model = kwargs.get("model")
        inp   = kwargs.get("input", [])
        messages = [{"role": m.get("role","user"), "content": m.get("content","")} for m in inp]
        return client.chat.completions.create(model=model, messages=messages)

# ---------------- PageIndex flatteners (copied) ----------------
def _clean_text(s: str) -> str:
    s = (s or "").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def flatten_pageindex_root(pi_obj, max_items=400, default_source="UNKNOWN.json"):
    out = []

    def _to_int(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str) and v.strip().lstrip("-").isdigit():
            return int(v.strip())
        return None

    def _pick_start_index(node):
        for key in ("start_index", "start_page", "page_start", "page_index",
                    "original_page", "page_number", "page"):
            v = node.get(key) if isinstance(node, dict) else None
            iv = _to_int(v)
            if iv is not None:
                return iv
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
                out.append({
                    "page": -1, "original_page": -1,
                    "section": "(text)", "quote": doc.strip(),
                    "source_file": default_source
                })
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
    """
    Computes hierarchical extraction metrics and writes them to CSV.
    Excludes RAG readiness score (handled separately).
    """
    # ---- Page Count (unique pages in full flattened entries) ----
    all_pages = {
        e.get("original_page")
        for e in index_entries
        if isinstance(e.get("original_page"), int) and e.get("original_page") >= 0
    }
    page_count = len(all_pages)

    # ---- Word Counts ----
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

    # ---- Topic Count ----
    topic_count = len(filtered_entries)

    # ---- Topic Coverage Ratio ----
    filtered_pages = {
        e.get("original_page")
        for e in filtered_entries
        if isinstance(e.get("original_page"), int) and e.get("original_page") >= 0
    }

    topic_coverage_ratio = (
        len(filtered_pages) / page_count
        if page_count > 0 else 0.0
    )

    # ---- Compression Rate ----
    compression_rate = (
        node_tree_word_count / original_word_count
        if original_word_count > 0 else 0.0
    )

    # ---- Write CSV ----
    csv_path = os.path.join(out_dir, "document_ingestion_metrics.csv")

    fieldnames = [
        "document_name",
        "page_count",
        "original_word_count",
        "node_tree_word_count",
        "topic_count",
        "topic_coverage_ratio",
        "compression_rate"
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

# ---------------- Tagging prompts  ----------------
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
    
    # Customize prompt based on expert type
        
    if "insurance" in industry:
        doc_type = "insurance document"
        customer_focus = "customers inquiring about insurance"
        example_keys = "coverage,exclusions,claims_process,premiums,deductibles,limitations"
        example_values = "comprehensive_motor_vehicle,windscreen_replacement,third_party_liability"
    elif "medical" in industry or "health" in industry:
        doc_type = "medical/healthcare document"
        customer_focus = "patients or healthcare professionals"
        example_keys = "diagnosis,treatment,medication,procedures,symptoms,contraindications"
        example_values = "acute_conditions,chronic_disease_management,dosage_guidelines"
    elif "math" in industry or "scientific" in industry:
        doc_type = "mathematical/scientific document"
        customer_focus = "students or researchers"
        example_keys = "theorems,formulas,proofs,definitions,applications,examples"
        example_values = "pythagorean_theorem,quadratic_equations,statistical_methods"
    elif "legal" in industry:
        doc_type = "legal document"
        customer_focus = "clients seeking legal guidance"
        example_keys = "rights,obligations,definitions,procedures,requirements,remedies"
        example_values = "statutory_rights,contractual_obligations,filing_procedures"
    elif "technical" in industry or "software" in industry or "engineer" in industry:
        doc_type = "technical documentation"
        customer_focus = "developers or technical users"
        example_keys = "features,configuration,api_methods,requirements,troubleshooting,security"
        example_values = "authentication_methods,data_structures,error_handling"
    else:
        # Generic fallback
        doc_type = "document"
        customer_focus = "readers"
        example_keys = "main_topics,concepts,procedures,definitions,guidelines,requirements"
        example_values = "specific_subtopics,detailed_aspects,particular_cases"

    rsp = responses_create(
        client,
        model=model,
        input=[
            {"role":"system","content":SYSTEM_TAG_GLOBAL.format(example_keys = example_keys, example_values = example_values)},
            {"role":"user","content":USER_TAG_GLOBAL_TEMPLATE.format(example_keys = example_keys, index_json=index_json)}
        ],
        response_format={"type":"json_object"},
        max_output_tokens=8192,
    )
    data = parse_json_from_resp(rsp, debug=debug, tag="tag_global")

    out: List[Dict[str, Any]] = []
    for c in (data.get("candidates") or []):
        key  = (c.get("key") or "").strip().lower()
        val  = (c.get("value") or "").strip().lower()
        sec  = (c.get("section") or "").strip()
        pg   = c.get("original_page", -1)
        src  = (c.get("source_file") or "document").strip()
        quo  = (c.get("quote") or "").strip()
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

# ---------------- PDF TOC Check ------------------------------

def pdf_has_embedded_toc(pdf_path: str) -> bool:
    """
    Fast check: does this PDF contain an embedded TOC in metadata?
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False

    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        doc.close()
        return bool(toc)
    except Exception:
        return False

# ---------------- TreeNode shape converter ------------------------

def treenode_to_pageindex_dict(node: TreeNode, source_file: str) -> Dict[str, Any]:
    """
    Convert normalized TreeNode into a PageIndex-compatible dict
    so flatten_pageindex_root can process it unchanged.
    """
    out = {
        "title": node.title,
        "page_start": node.page_start,
        "page_end": node.page_end,
        "text": node.content,
        "source_file": source_file,
        "children": []
    }

    for child in node.children:
        out["children"].append(
            treenode_to_pageindex_dict(child, source_file)
        )

    return out

# ---------------- Helper: load_or_convert_pageindex (copied) ----------------
def load_or_convert_pageindex(input_path: str, out_dir: str, debug: bool=False) -> Tuple[str, Any]:
    """
    Hybrid loader:

    - If PDF:
        - If embedded TOC exists → use PyMuPDFExtractor
        - Else → fallback to PageIndex API
    - If JSON → load as before
    """
    path = input_path
    root, ext = os.path.splitext(path)
    ext_lower = ext.lower()

    if not ext_lower:
        path = path + ".json"
        ext_lower = ".json"

    # ---------------- PDF CASE ----------------
    if ext_lower == ".pdf":

        # ---- Try PyMuPDF first (if installed + TOC exists) ----
        if PyMuPDFExtractor is not None and pdf_has_embedded_toc(path):
            print("[INFO] Embedded TOC detected. Using PyMuPDF extractor.")

            extractor = PyMuPDFExtractor(verbose=debug)
            result = extractor.run(Path(path))

            if result.error:
                print(f"[WARN] PyMuPDF extraction failed: {result.error}")
                print("[INFO] Falling back to PageIndex...")
            else:
                tree_root = result.tree
                source_file = os.path.basename(path)

                # Convert TreeNode → PageIndex-compatible dict
                pi_like_dict = treenode_to_pageindex_dict(tree_root, source_file)

                # Persist tree for debugging parity
                base_name = os.path.splitext(os.path.basename(path))[0]
                out_json_path = os.path.join(out_dir, f"{base_name}_pymupdf_tree.json")

                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(pi_like_dict, f, ensure_ascii=False, indent=2)

                print(f"[INFO] PyMuPDF tree written to: {out_json_path}")

                return out_json_path, pi_like_dict

        # ---- Fallback to PageIndex ----
        if PageIndexClient is None:
            raise SystemExit(
                "PDF input detected, but neither PyMuPDF TOC nor PageIndex SDK is available.\n"
                "Install one of:\n"
                "  pip install pymupdf\n"
                "  pip install pageindex"
            )

        api_key = os.environ.get("PAGEINDEX_API_KEY")
        if not api_key:
            raise SystemExit(
                "PDF input detected, but PAGEINDEX_API_KEY is not set."
            )

        print("[INFO] No embedded TOC found. Falling back to PageIndex.")

        pi_client = PageIndexClient(api_key=api_key)
        submit_res = pi_client.submit_document(path)
        doc_id = submit_res.get("doc_id") or submit_res.get("id")

        if not doc_id:
            raise SystemExit(f"Failed to obtain doc_id: {submit_res!r}")

        max_wait_seconds = 7200  # 2-hour ceiling for large documents
        poll_start = time.time()
        consecutive_errors = 0

        while True:
            elapsed = time.time() - poll_start
            if elapsed > max_wait_seconds:
                raise SystemExit(
                    f"PageIndex polling timed out after {max_wait_seconds}s for doc_id={doc_id}"
                )

            try:
                tree_result = pi_client.get_tree(doc_id)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                print(f"[WARN] PageIndex poll error (attempt {consecutive_errors}): {exc}")
                if consecutive_errors >= 5:
                    raise SystemExit(
                        f"PageIndex polling failed after 5 consecutive network errors: {exc}"
                    )
                time.sleep(10.0)
                continue

            status = tree_result.get("status")

            if debug:
                print(f"[DEBUG] PageIndex status for doc_id={doc_id}: {status} (elapsed={elapsed:.0f}s)")

            if status == "completed":
                tree = tree_result.get("result")
                if not tree:
                    raise SystemExit(
                        f"PageIndex returned completed but tree result is empty: {tree_result!r}"
                    )
                base_name = os.path.splitext(os.path.basename(path))[0]
                out_json_path = os.path.join(out_dir, f"{base_name}_pageindex_tree.json")

                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(tree, f, ensure_ascii=False, indent=2)

                print(f"[INFO] PageIndex tree written to: {out_json_path}")
                return out_json_path, tree

            if status == "failed":
                raise SystemExit(f"PageIndex processing failed: {tree_result!r}")

            time.sleep(5.0)

    # ---------------- JSON CASE ----------------
    with open(path, "r", encoding="utf-8") as f:
        pi = json.load(f)

    return path, pi


# ---------------- Main for preprocessing ----------------
def main():
    ap = argparse.ArgumentParser(description="Document preprocess: PageIndex conversion, flatten, filter, tag bank")
    ap.add_argument("--input", required=True, help="Path to PDF or PageIndex JSON")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--industry", default="")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    input_path = args.input
    client = make_client()

    # Convert or load
    pi_path, pi_tree = load_or_convert_pageindex(input_path, args.out_dir, args.debug)
    # Flatten
    default_src = os.path.splitext(os.path.basename(pi_path))[0] + ".json"
    index_entries = flatten_pageindex_root(pi_tree, max_items=None, default_source=default_src)

    # Persist full flattened
    out_flat_path = os.path.join(args.out_dir, "index_entries_flat.json")
    with open(out_flat_path, "w", encoding="utf-8") as f:
        json.dump(index_entries, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote flattened entries to: {out_flat_path}")

    # Filter entries with quote length > 100
    filtered = [e for e in index_entries if isinstance(e.get("quote",""), str) and len(e.get("quote","")) > 100]
    out_filtered_path = os.path.join(args.out_dir, "filtered_entries.json")
    with open(out_filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote filtered entries (len>100) to: {out_filtered_path} (count={len(filtered)})")

    if not filtered:
        print("[WARN] No filtered entries (quote length > 100). Tag bank will be empty.")
    # Tag bank generation
    tag_bank = tag_document_global_with_llm(client, args.model, filtered, industry = args.industry, debug=args.debug) if filtered else []
    out_tag_path = os.path.join(args.out_dir, "tag_bank.json")
    with open(out_tag_path, "w", encoding="utf-8") as f:
        json.dump(tag_bank, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote tag bank to: {out_tag_path} (entries={len(tag_bank)})")

    # Ingestion Metrics
    document_name = os.path.basename(input_path)
    compute_and_write_ingestion_metrics(
        out_dir=args.out_dir,
        document_name=document_name,
        index_entries=index_entries,
        filtered_entries=filtered
    )

    print("[DONE] document_preprocess completed.")

if __name__ == "__main__":
    main()
