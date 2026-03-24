#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
testcase_generate.py
- Loads:
    - filtered_entries.json  (topics)
    - tag_bank.json          (LLM-produced global tags)
  produced by document_preprocess.py
- Runs QGen per topic, QA (grounded), variants, attaches SBERT tags, outputs CSVs/JSON.
"""
import os
import re
import json
import math
import argparse
import time
import csv
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from openai import OpenAI

# ---------------- Concurrency + retry helpers (copied) ----------------
def _with_retries(fn, retries=3, backoff=1.0, *args, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            wait = backoff * (2 ** (attempt - 1))
            print(f"[WARN] Call failed (attempt {attempt}/{retries}). Retrying after {wait:.1f}s. Error: {e}")
            time.sleep(wait)
    # Final attempt (raise if fails)
    return fn(*args, **kwargs)

def run_threaded_calls(callable_fn, arg_list, max_workers=8, debug: bool=False):
    results = [None] * len(arg_list)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {}
        for idx, (args, kwargs) in enumerate(arg_list):
            future = ex.submit(_with_retries, callable_fn, 3, 1.0, *args, **kwargs)
            future_to_idx[future] = idx
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"[ERROR] Task idx={idx} failed permanently: {e}\n{traceback.format_exc()}")
                results[idx] = None
    return results

# ---------------- OpenAI helpers (copied) ----------------
_client = None
LLM_USAGE_LOG: List[Dict[str, int]] = []

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

def _usage_numbers(rsp):
    u = getattr(rsp, "usage", None)
    def _get(obj, *names):
        for n in names:
            v = getattr(obj, n, None) if obj else None
            if v is not None:
                return v
        return None

    input_tok  = _get(u, "input_tokens", "prompt_tokens")
    output_tok = _get(u, "output_tokens", "completion_tokens")
    total_tok  = _get(u, "total_tokens")

    if (input_tok is None or output_tok is None or total_tok is None):
        out = getattr(rsp, "output", None)
        if isinstance(out, list) and out:
            u2 = getattr(out[0], "usage", None) if not isinstance(out[0], dict) else out[0].get("usage")
            if u2:
                input_tok  = input_tok  if input_tok  is not None else _get(u2, "input_tokens", "prompt_tokens")
                output_tok = output_tok if output_tok is not None else _get(u2, "output_tokens", "completion_tokens")
                total_tok  = total_tok  if total_tok is not None else _get(u2, "total_tokens")

    if total_tok is None and (isinstance(input_tok, int) and isinstance(output_tok, int)):
        total_tok = input_tok + output_tok

    return input_tok, output_tok, total_tok

def responses_create(client: OpenAI, **kwargs):
    """
    Wrapper around client.responses.create / chat.completions.create
    that also logs token usage into LLM_USAGE_LOG.
    """
    global LLM_USAGE_LOG
    try:
        rsp = client.responses.create(**kwargs)
    except Exception:
        model = kwargs.get("model")
        inp   = kwargs.get("input", [])
        messages = [{"role": m.get("role","user"), "content": m.get("content","")} for m in inp]
        rsp = client.chat.completions.create(model=model, messages=messages)

    try:
        it, ot, tt = _usage_numbers(rsp)
        if any(v is not None for v in (it, ot, tt)):
            LLM_USAGE_LOG.append({
                "input_tokens": it or 0,
                "output_tokens": ot or 0,
                "total_tokens": tt or ((it or 0) + (ot or 0))
            })
    except Exception:
        pass

    return rsp

# ---------------- Embeddings helper ----------------
def _embed_norm(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        dim = getattr(model, "get_sentence_embedding_dimension", lambda: 384)() or 384
        return np.zeros((0, dim), dtype=np.float32)
    X = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return normalize(X)

# ---------------- Prompts (copied) ----------------

SYSTEM_QGEN_PI = (
    "You are an {industry} expert creating test questions and answers for a given topic"
    "You write realistic, personal, and natural-sounding consumer questions (not formal or stiff). "
    "Use ONLY the provided PageIndex entries for grounding when generating the questions."
    "ONLY ask questions that are answerable from content provided in the PageIndex entries"
    "DO NOT generate questions that are not covered by the content in the PageIndex entries"

    "QUESTION STYLE (IMPORTANT):"
    "{question_style}"
)

USER_QGEN_TEMPLATE_PI = (
    "Task: Write {n} realistic questions a human might ask based ONLY on the PageIndex entries below.\n"
    "PAGEINDEX ENTRIES (JSON):\n{index_json}\n\n"
    "Return STRICT JSON: {{\"questions\":[\"...\"]}}"
)

SYSTEM_QA_PI = (
    "Answer questions using only the provided PageIndex entries. "
    "Each answer MUST be grounded using the provided PageIndex entries"
    "Answer each question by paraphrasing the grounded context"
    "DO NOT refer to the document when answering. Use a neutral but welcoming tone when answering"
    "Include a citation 'P.{original_page} - {source_file} - {section}'."
    "If not answerable from the entries, respond exactly 'Unknown from document'."
)

USER_QA_TEMPLATE_PI = (
    "Questions JSON:\n{qs_json}\n\n"
    "Ground only with these PageIndex entries:\n{index_json}\n\n"
    "Return STRICT JSON: {{\"items\":[{{\"question\":\"\",\"answer\":\"\",\"citation\":\"P.{{original_page}} - {{source_file}} - {{section}}\"}}]}}"
)

VARIANT_SYSTEM = "You are an expert rewriter. Create alternative phrasing for the following input questions preserving EXACT meaning and the correct answer intent. Vary tone, formality and style between variants but do NOT change the factual content or the answer's scope."
VARIANT_USER_TEMPLATE = (
    "Base question:\n{q}\n\n"
    "Instructions (CRITICAL): Produce {n} distinct rewrites of the base question.\n"
    "- Keep the same overall meaning and expected answer.\n"
    "- Each variant must have different question phrasing"
    "- Vary tone, style, wording, and formality.\n"
    "{variation_examples}"
    "- Return STRICT JSON: {{\"variants\":[\"...\",\"...\"]}}"
)

# ---------------- QGen / QA functions ----------------
def step2_generate_questions_pageindex(
    client: OpenAI, model: str, index_entries: List[Dict[str, Any]],
    n: int, industry: str, debug: bool=False
) -> List[str]:
    if "education" in industry or "teacher" in industry or "school" in industry:
        question_style = """Write questions as if a teacher or staff member is asking an education administrator in a casual chat.
        Examples of good conversational questions:
        - "Hi, I was wondering what the dress code policy is for staff?"
        - "Can you tell me about the different types of employment at the college?"
        - "I need to know what training I'm required to complete for child protection?"
        - "What happens if I need to work late on campus after 8pm?"
        Keep questions natural, direct, and conversational - like real workplace questions."""
                
    elif "insurance" in industry:
        question_style = """Write questions as if a policyholder is asking their insurance agent in a casual conversation.
        Examples of good conversational questions:
        - "What's covered if my car gets stolen?"
        - "How do I make a claim for storm damage?"
        - "Can you explain what the excess is and when I need to pay it?"
        - "Am I covered if I lend my car to a friend?"
        Keep questions natural and conversational - like real customer inquiries."""
                
    elif "medical" in industry or "health" in industry:
        question_style = """Write questions as if a patient is asking their doctor or healthcare provider.
        Examples of good conversational questions:
        - "What are the symptoms I should watch out for?"
        - "How often should I take this medication?"
        - "Are there any side effects I need to know about?"
        - "What should I do if the treatment isn't working?"
        Keep questions natural and conversational - like real patient questions."""
                
    elif "legal" in industry:
        question_style = """Write questions as if a client is asking their lawyer for advice.
        Examples of good conversational questions:
        - "What are my rights in this situation?"
        - "Do I need to file any paperwork for this?"
        - "What happens if I don't comply with this requirement?"
        - "Can you explain what this legal term means?"
        Keep questions natural and conversational - like real client inquiries."""
                
    else:
                    # Generic conversational style
        question_style = """Write questions in a natural, conversational style as if someone is asking an expert directly.
        Examples:
        - "Can you explain how this works?"
        - "What do I need to do if I want to...?"
        - "What are the requirements for...?"
        - "How does this policy apply to...?"
        Keep questions natural and conversational - like real person-to-person inquiries."""

    index_json = json.dumps(index_entries[:300], ensure_ascii=False)
    rsp = responses_create(
        client,
        model=model,
        input=[
            {"role":"system","content":SYSTEM_QGEN_PI.format(industry=industry, question_style=question_style)},
            {"role":"user","content":USER_QGEN_TEMPLATE_PI.format(n=n, index_json=index_json)}
        ],
        response_format={"type":"json_object"},
        max_output_tokens=8192,
    )
    data = parse_json_from_resp(rsp, debug=debug, tag="qgen_pi")
    qs: List[str] = []
    for q in (data.get("questions") or []):
        if isinstance(q, str) and q.strip():
            qs.append(q.strip())
    return qs[:n]

def step2_answer_questions_pageindex(
    client: OpenAI, model: str, questions: List[str], index_entries: List[Dict[str, Any]],
    debug: bool=False, sbert_model: SentenceTransformer = None
) -> Dict[str, Any]:
    index_json = json.dumps(index_entries[:300], ensure_ascii=False)
    qs_json = json.dumps(questions, ensure_ascii=False)
    rsp = responses_create(
        client,
        model=model,
        input=[
            {"role":"system","content":SYSTEM_QA_PI},
            {"role":"user","content":USER_QA_TEMPLATE_PI.format(qs_json=qs_json, index_json=index_json)}
        ],
        response_format={"type":"json_object"},
    )
    data = parse_json_from_resp(rsp, debug=debug, tag="qa_pi")

    # Attach best-effort citations even for Unknown
    try:
        entry_texts = [e.get("quote","") for e in index_entries]
        S = _embed_norm(sbert_model, entry_texts, batch_size=64) if sbert_model is not None else None
    except Exception:
        S = None

    items_in = data.get("items") or []
    items_out: List[Dict[str, str]] = []
    for i, it in enumerate(items_in):
        q = str(it.get("question","")).strip()
        a = str(it.get("answer","")).strip()
        c = str(it.get("citation","")).strip()
        if (not c) or (c.upper()=="N/A") or ("Unknown from document" in a):
            if sbert_model is not None and S is not None and len(index_entries):
                Q = _embed_norm(sbert_model, [q or questions[i] if i < len(questions) else ""], batch_size=1)
                sims = (Q[0] @ S.T).astype(np.float32)
                idx = int(np.argmax(sims))
                e = index_entries[idx]
                pg_raw = e.get("original_page", None)
                if isinstance(pg_raw, (int, float)):
                    pg = int(pg_raw)
                elif isinstance(pg_raw, str) and pg_raw.strip().lstrip("-").isdigit():
                    pg = int(pg_raw.strip())
                else:
                    pg = "N/A"
                src = e.get("source_file", "document")
                sec = e.get("section", "(section)")
                c = f"P.{pg} - {src} - {sec}"
            else:
                c = c or "N/A"
        items_out.append({"question": q, "answer": a, "citation": c})
    data["items"] = items_out
    return data

def generate_variants_llm(client: OpenAI, model: str, base_q: str, n: int, industry: str, debug: bool=False) -> List[str]:
    if "education" in industry or "teacher" in industry or "school" in industry:
        variation_examples = """EXAMPLES (Education context):
        Original: "What are the Department Head positions and their contact information?"
        ✓ Good: "Who holds Department Head roles and how can I reach them?"
        ✓ Good: "Can you tell me who the Department Heads are and their contact details?"
        ✗ Bad: "Who leads the Mathematics Department?" (too narrow - asks about ONE dept not ALL)

        Original: "What training do I need to complete for child protection?"
        ✓ Good: "Can you tell me what child protection training is required?"
        ✓ Good: "What are the mandatory training requirements for safeguarding?"
        ✗ Bad: "Do I need training?" (too vague - doesn't specify what kind)"""
                
    elif "insurance" in industry:
        variation_examples = """EXAMPLES (Insurance context):
        Original: "What's covered if my car is stolen?"
        ✓ Good: "Am I covered for theft of my vehicle?"
        ✓ Good: "If someone steals my car, what does the policy cover?"
        ✗ Bad: "What theft coverage do I have?" (too broad - doesn't specify car)

        Original: "How do I make a claim for storm damage?"
        ✓ Good: "What's the process for claiming storm damage?"
        ✓ Good: "Can you walk me through making a storm damage claim?"
        ✗ Bad: "How do I claim?" (too vague - doesn't specify storm damage)"""
                
    else:
                    # Generic examples
        variation_examples = """EXAMPLES:
        Original: "What are the Department Head positions and their contact information?"
        ✓ Good: "Who holds Department Head roles and how can they be contacted?"
        ✓ Good: "Can you tell me about the Department Heads and their contact details?"
        ✗ Bad: "Who leads the Mathematics Department?" (too narrow - one dept not all)"""
    if n <= 0:
        return []
    rsp = responses_create(
        client,
        model=model,
        input=[
            {"role":"system","content":VARIANT_SYSTEM},
            {"role":"user","content":VARIANT_USER_TEMPLATE.format(q=base_q, n=n, variation_examples = variation_examples)}
        ],
        response_format={"type":"json_object"},
        max_output_tokens=8192,
    )
    data = parse_json_from_resp(rsp, debug=debug, tag="variants")
    variants: List[str] = []
    for v in (data.get("variants") or []):
        if isinstance(v, str) and v.strip():
            variants.append(v.strip())
    return variants[:n]

# ---------------- Tagging attach (SBERT) ----------------
@dataclass
class TagHit:
    key: str
    value: str
    page: int
    section: str
    quote: str
    long_citation: str
    score: float
    source_file: str
    original_page: int

def build_hits_from_index_tagged(tagged: List[Dict[str, Any]]) -> List[TagHit]:
    hits: List[TagHit] = []
    for entry in tagged:
        page = int(entry.get("page") or 0)
        src  = entry.get("source_file") or "document"
        sec  = entry.get("section") or ""
        quo  = entry.get("quote") or ""
        for c in (entry.get("candidates") or []):
            key = (c.get("key") or "").strip().lower()
            val = (c.get("value") or "").strip().lower()
            try:
                sc = float(c.get("score", 0.0))
            except Exception:
                sc = 0.0
            if not key or not val:
                continue
            hits.append(TagHit(
                key=key, value=val, page=page, section=sec,
                quote=quo, long_citation=quo, score=sc,
                source_file=src, original_page=page
            ))
    return hits

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def attach_tags_to_qa(
    hits: List[TagHit],
    qa_rows: List[Dict[str, str]],
    sbert_model: SentenceTransformer,
    max_tags_per_qa: int = 10,
    conf_thresh: float = 0.4
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tag_meta: List[Tuple[int, str, str, str]] = []
    tag_texts: List[str] = []
    for h in hits:
        txt = (h.long_citation or "") + " \n " + (h.quote or "")
        tag_meta.append((h.page, (h.key or "").strip().lower(), (h.value or "").strip().lower(), (h.section or "")))
        tag_texts.append(txt if txt.strip() else (h.section or ""))

    if not qa_rows:
        return [], {
            "qa_with_at_least_1_tag": 0, "total_qa": 0, "qa_tagged_pct": 0.0,
            "unique_tag_pairs_from_step1": len({(h.key, h.value) for h in hits}),
            "unique_tag_pairs_used_in_step3": 0, "tag_pairs_used_pct": 0.0
        }

    T = _embed_norm(sbert_model, tag_texts, batch_size=64)
    qa_texts: List[str] = [(row.get("question","") or "") + "\n" + (row.get("answer","") or "") for row in qa_rows]
    Q = _embed_norm(sbert_model, qa_texts, batch_size=64)

    used_pairs = set()
    qa_with_tags = 0
    enriched: List[Dict[str, Any]] = []

    for i, row in enumerate(qa_rows):
        q = row.get("question", "") or ""
        a = row.get("answer", "") or ""
        cit = row.get("citation", "") or ""
        qa_plain = (q + " \n " + a).lower()

        cited_page = 0
        cited_section = ""
        if cit and cit.lower().startswith("p.") and "-" in cit:
            try:
                pnum = re.findall(r"\d+", cit.split("-", 1)[0])
                cited_page = int(pnum[0]) if pnum else 0
            except Exception:
                cited_page = 0
            cited_section = re.sub(r"[^a-z0-9]+", " ", cit.split("-", 1)[1].lower()).strip() if "-" in cit else ""

        sims = (Q[i] @ T.T).astype(np.float32) if T.shape[0] else np.zeros((0,), dtype=np.float32)
        sims = np.clip(sims, 0.0, 1.0)

        scored: List[Tuple[float, Tuple[str, str], Dict[str, float]]] = []
        for j, base in enumerate(sims):
            p, key, val, section = tag_meta[j]
            sec_norm = re.sub(r"[^a-z0-9]+", " ", (section or "").lower()).strip()
            bonus_val = 0.05 if (val and val in qa_plain) else 0.0
            bonus_page = 0.06 if (cited_page and p == cited_page) else 0.0
            bonus_sec = 0.06 if (cited_section and sec_norm and sec_norm in cited_section) else 0.0
            final = float(base) + bonus_val + bonus_page + bonus_sec
            scored.append((final, (key, val), {
                "cosine": float(base),
                "bonus_value_literal": bonus_val,
                "bonus_same_page": bonus_page,
                "bonus_section_match": bonus_sec,
                "final": float(final),
            }))

        scored.sort(key=lambda x: x[0], reverse=True)

        ranked_all: List[Dict[str, Any]] = []
        seen_pairs = set()
        for sc, (key, val), detail in scored:
            if (key, val) in seen_pairs:
                continue
            ranked_all.append({
                "key": key,
                "value": val,
                "confidence": round(detail["final"], 4),
                "cosine": round(detail["cosine"], 4),
                "bonus_value_literal": detail["bonus_value_literal"],
                "bonus_same_page": detail["bonus_same_page"],
                "bonus_section_match": detail["bonus_section_match"]
            })
            seen_pairs.add((key, val))
            if len(ranked_all) >= max_tags_per_qa:
                break

        chosen_ranked: List[Dict[str, Any]] = [it for it in ranked_all if it["confidence"] >= conf_thresh]
        if not chosen_ranked and ranked_all:
            chosen_ranked = [ranked_all[0]]

        def _shorten_tag_value(val: str) -> str:
            import re
            words = re.findall(r"[A-Za-z0-9]+", val)
            stop = {"the","and","of","policy","policies","procedure","procedures","process"}
            filtered = [w for w in words if w.lower() not in stop] or words
            short = " ".join(filtered[:2])
            return short.title().strip()

        by_key: Dict[str, List[str]] = {}
        for it in chosen_ranked:
            short_val = _shorten_tag_value(it["value"])
            if short_val:
                snake = re.sub(r"[^a-z0-9]+", "_", short_val.lower()).strip("_")
                by_key.setdefault(it["key"], []).append(snake)

        cit = row.get("citation", "")
        parts = cit.split(" - ")
        src = parts[1] if len(parts) > 1 else "unknown"
        by_key["source_file"] = [src]

        if chosen_ranked:
            qa_with_tags += 1
            for it in chosen_ranked:
                used_pairs.add((it["key"], it["value"]))

        enriched.append({
            "question": q,
            "answer": a,
            "citation": cit,
            "sbert_ranked_tags": json.dumps(ranked_all, ensure_ascii=False),
            "meta_tags": json.dumps(by_key, ensure_ascii=False)
        })

    metrics = {
        "qa_with_at_least_1_tag": qa_with_tags,
        "total_qa": len(qa_rows),
        "qa_tagged_pct": round(100.0 * qa_with_tags / max(1, len(qa_rows)), 2),
        "unique_tag_pairs_from_step1": len({(h.key, h.value) for h in hits}),
        "unique_tag_pairs_used_in_step3": len(used_pairs),
        "tag_pairs_used_pct": round(100.0 * len(used_pairs) / max(1, len({(h.key, h.value) for h in hits})), 2)
    }
    return enriched, metrics


# ---------------- Main pipeline (uses files from preprocess) ----------------
def main():
    start_time = time.time()
    ap = argparse.ArgumentParser(description="Testcase generation from preprocessed PageIndex + tag bank")
    ap.add_argument("--filtered_entries", required=True, help="Path to filtered_entries.json (topics)")
    ap.add_argument("--tag_bank", required=True, help="Path to tag_bank.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--industry", required=True)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--base_questions_per_topic", type=int, required=True)
    ap.add_argument("--variations", type=int, required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--import_flag", dest="import_flag", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    sbert_name_check = os.environ.get("SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    default_sbert = "sentence-transformers/all-MiniLM-L6-v2"
    if not hf_token:
        if sbert_name_check != default_sbert:
            raise SystemExit(
                f"HF_TOKEN is not set but SBERT_MODEL_NAME is '{sbert_name_check}' (non-default).\n"
                "A Hugging Face token is required to download private or gated models.\n"
                "Set it with: export HF_TOKEN=hf_..."
            )
        print("[WARN] HF_TOKEN is not set. Proceeding with default public SBERT model.")

    sbert_model = SentenceTransformer(sbert_name_check)

    # Load precomputed files
    with open(args.filtered_entries, "r", encoding="utf-8") as f:
        topics = json.load(f)

    with open(args.tag_bank, "r", encoding="utf-8") as f:
        tagged_entries_from_step1 = json.load(f)

    Y = len(topics)
    if Y == 0:
        raise SystemExit("No topics (filtered entries) found. Run document_preprocess.py first.")

    base_questions_per_topic = int(args.base_questions_per_topic)
    variations = int(args.variations)
    total_base = Y * base_questions_per_topic
    total_cases = Y * base_questions_per_topic * variations

    if variations < 1:
        raise SystemExit("variations must be >= 1")

    if args.debug:
        print(f"[DEBUG] Topics: {Y} | Variations per base: {variations} | Base per topic: {base_questions_per_topic} | Total base: {total_base} | Total cases (final): {total_cases}")

    client = make_client()

    # ---------------- Parallel QGEN ----------------
    print("[INFO] Generating base questions for each topic (parallel)...")
    qgen_args = []
    for ti, topic_entry in enumerate(topics):
        qgen_args.append(((client, args.model, [topic_entry], base_questions_per_topic, args.industry, args.debug), {}))

    qgen_results = run_threaded_calls(step2_generate_questions_pageindex, qgen_args, max_workers=args.workers, debug=args.debug)

    all_base_questions = []
    for ti, qs in enumerate(qgen_results):
        if not qs:
            if args.debug:
                print(f"[WARN] QGEN returned no questions for topic {ti} ({topics[ti].get('section')})")
            continue
        for q in qs:
            all_base_questions.append({
                "topic_idx": ti,
                "section": topics[ti].get("section"),
                "source_file": topics[ti].get("source_file"),
                "original_page": topics[ti].get("original_page"),
                "quote": topics[ti].get("quote"),
                "question": q
            })

    if len(all_base_questions) < total_base:
        print(f"[WARN] Expected {total_base} base questions but LLM returned {len(all_base_questions)}. Proceeding with available base questions.")
        total_base = len(all_base_questions)
        total_cases = max(total_cases, total_base * variations)

    # ---------------- Parallel QA (per topic) ----------------
    print("[INFO] Answering base questions per topic (parallel)...")
    qa_tasks = []
    for ti in range(Y):
        qs_for_topic = [b["question"] for b in all_base_questions if b["topic_idx"] == ti]
        if not qs_for_topic:
            continue
        topic_entry = topics[ti]
        qa_tasks.append(((client, args.model, qs_for_topic, [topic_entry], args.debug), {"sbert_model": sbert_model}))

    qa_results = run_threaded_calls(step2_answer_questions_pageindex, qa_tasks, max_workers=args.workers, debug=args.debug)

    # Collect answered items in the same order as all_base_questions
    answered_base_items = []
    topic_answers_map = {}
    task_idx = 0
    for ti in range(Y):
        qs_for_topic = [b["question"] for b in all_base_questions if b["topic_idx"] == ti]
        if not qs_for_topic:
            continue
        res = qa_results[task_idx] if task_idx < len(qa_results) else None
        task_idx += 1
        items = (res.get("items") if isinstance(res, dict) else None) or []
        topic_answers_map[ti] = items

    topic_counters = {}
    for b in all_base_questions:
        ti = b["topic_idx"]
        items = topic_answers_map.get(ti, [])
        idx = topic_counters.get(ti, 0)
        if idx < len(items):
            it = items[idx]
            answered_base_items.append({
                "topic_idx": ti,
                "section": b.get("section"),
                "question": it.get("question") or b.get("question"),
                "answer": it.get("answer") or "Unknown from document",
                "citation": it.get("citation") or "N/A",
                "source_file": b.get("source_file"),
                "original_page": b.get("original_page")
            })
        else:
            answered_base_items.append({
                "topic_idx": ti,
                "section": b.get("section"),
                "question": b.get("question"),
                "answer": "Unknown from document",
                "citation": "N/A",
                "source_file": b.get("source_file"),
                "original_page": b.get("original_page")
            })
        topic_counters[ti] = idx + 1

    qa_raw_out = {"items": answered_base_items}
    with open(os.path.join(args.out_dir, "step2_qa_raw.json"), "w", encoding="utf-8") as f:
        json.dump(qa_raw_out, f, ensure_ascii=False, indent=2)

    qa_rows_base = [{"question": it["question"], "answer": it["answer"], "citation": it["citation"]} for it in answered_base_items]
    pd.DataFrame(qa_rows_base).to_csv(os.path.join(args.out_dir, "step2_qa_docgrounded.csv"),
                                     index=False, encoding="utf-8-sig")

    # ---------------- Parallel Variant Generation ----------------
    print("[INFO] Generating variants for each base question (parallel)...")
    variant_tasks = []
    rewrites_needed = max(0, variations - 1)
    for b in answered_base_items:
        base_q = b["question"]
        variant_tasks.append(((client, args.model, base_q, rewrites_needed, args.industry, args.debug), {}))

    variant_results = run_threaded_calls(generate_variants_llm, variant_tasks, max_workers=args.workers, debug=args.debug)

    final_rows = []
    for i, b in enumerate(answered_base_items):
        base_q = b["question"]
        base_a = b["answer"]
        base_c = b["citation"]
        rewrites = variant_results[i] or []
        final_rows.append({"question": base_q, "answer": base_a, "citation": base_c})
        for r in rewrites:
            final_rows.append({"question": r, "answer": base_a, "citation": base_c})
        if len(rewrites) < rewrites_needed and args.debug:
            print(f"[DEBUG] For base question '{base_q}' expected {rewrites_needed} rewrites, got {len(rewrites)}")

    if len(final_rows) < total_cases:
        print(f"[WARN] Final number of generated questions ({len(final_rows)}) is less than requested/adjusted total ({total_cases}).")

    # ---------------- Quality metrics ----------------
    # Unanswered rate: % of final rows where LLM could not ground an answer
    unknown_count = sum(1 for r in final_rows if "Unknown from document" in (r.get("answer") or ""))
    unanswered_rate_pct = round(100.0 * unknown_count / max(1, len(final_rows)), 2)

    # Base-to-variant similarity: mean cosine sim between each base question and its variants
    # Values near 1.0 = variants too similar; near 0.0 = variants may have drifted in meaning
    _bv_sims: List[float] = []
    if rewrites_needed > 0:
        _bv_base_qs: List[str] = []
        _bv_all_variants: List[str] = []
        _bv_counts: List[int] = []
        for i, b in enumerate(answered_base_items):
            rewrites = variant_results[i] or []
            if not rewrites:
                continue
            _bv_base_qs.append(b["question"])
            _bv_all_variants.extend(rewrites)
            _bv_counts.append(len(rewrites))
        if _bv_base_qs:
            _base_embs = _embed_norm(sbert_model, _bv_base_qs)
            _var_embs  = _embed_norm(sbert_model, _bv_all_variants)
            _var_offset = 0
            for bi, count in enumerate(_bv_counts):
                _bv_sims.extend((_base_embs[bi] @ _var_embs[_var_offset:_var_offset + count].T).tolist())
                _var_offset += count
    mean_base_variant_sim = round(float(np.mean(_bv_sims)), 4) if _bv_sims else None

    # ---------------- Tag merging and SBERT attach ----------------
    print("[INFO] Attaching SBERT tags from tag_bank to Q&A...")
    hits = build_hits_from_index_tagged(tagged_entries_from_step1)
    enriched, metrics = attach_tags_to_qa(hits, final_rows, sbert_model=sbert_model, max_tags_per_qa=10, conf_thresh=0.4)

    pd.DataFrame(final_rows).to_csv(os.path.join(args.out_dir, "step3_qa_with_variants_docgrounded.csv"),
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame(enriched).to_csv(os.path.join(args.out_dir, "step3_qa_with_metatags.csv"),
                                  index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out_dir, "step3_qa_with_metatags.json"), "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "step3_tag_attach_stats.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if args.import_flag:
        doc_name = os.path.splitext(os.path.basename(args.filtered_entries))[0]
        import_path = os.path.join(args.out_dir, f"step3_import_format_{doc_name}.csv")
        fieldnames = [
            "test_plan_name","test_case_id","test_case_name","test_case_type",
            "active","action","step_number","object","value","meta_tags"
        ]
        with open(import_path, "w", newline="", encoding="utf-8-sig") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for row in enriched:
                q = row.get("question", "") or ""
                a = row.get("answer", "") or ""
                meta_raw = row.get("meta_tags") or "{}"
                try:
                    meta_obj = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                except Exception:
                    meta_obj = {}
                meta_list = []
                if isinstance(meta_obj, dict):
                    for k, vals in meta_obj.items():
                        if isinstance(vals, list):
                            for v in vals:
                                meta_list.append({"key": k, "value": str(v)})
                        else:
                            meta_list.append({"key": k, "value": str(vals)})
                else:
                    meta_list.append({"key": "meta", "value": str(meta_obj)})
                meta_for_import = json.dumps(meta_list, ensure_ascii=False)
                writer.writerow({
                    "test_plan_name": doc_name,
                    "test_case_id": "",
                    "test_case_name": q,
                    "test_case_type":"baseline",
                    "active":"",
                    "action": "start-session",
                    "step_number": "1",
                    "object": "event",
                    "value": "start-session",
                    "meta_tags": meta_for_import
                })
                writer.writerow({
                    "test_plan_name": doc_name,
                    "test_case_id": "",
                    "test_case_name": q,
                    "test_case_type":"baseline",
                    "active":"",
                    "action": "user-input",
                    "step_number":"2",
                    "object": "text",
                    "value": q,
                    "meta_tags": meta_for_import
                })
                writer.writerow({
                    "test_plan_name": doc_name,
                    "test_case_id": "",
                    "test_case_name": q,
                    "test_case_type":"baseline",
                    "active":"",
                    "action": "object-semantics",
                    "step_number":"3",
                    "object": "text",
                    "value": a,
                    "meta_tags": meta_for_import
                })
        print(f"[INFO] Import-format CSV written to: {import_path}")

    print(f"Wrote outputs to: {os.path.abspath(args.out_dir)}")
    print(f"Topics: {Y} | Base questions: {total_base} | Final generated rows: {len(final_rows)}")

    # Debug stats
    llm_calls = len(LLM_USAGE_LOG)
    total_in = sum(x.get("input_tokens", 0) for x in LLM_USAGE_LOG)
    total_out = sum(x.get("output_tokens", 0) for x in LLM_USAGE_LOG)
    total_tok = sum(x.get("total_tokens", 0) for x in LLM_USAGE_LOG)
    debug_stats = {
        "topics": Y,
        "base_questions": total_base,
        "final_rows": len(final_rows),
        "runtime_seconds": time.time() - start_time,
        "quality_metrics": {
            # % of rows (base + variants) where the LLM returned "Unknown from document"
            "unanswered_rate_pct": unanswered_rate_pct,
            "unanswered_count": unknown_count,
            # Mean cosine similarity between each base question and its variants (0–1 scale)
            # Ideal range ~0.75–0.92: distinct enough to be useful, similar enough to be faithful
            "mean_base_variant_similarity": mean_base_variant_sim,
        },
        "llm_usage": {"llm_calls": llm_calls, "total_input_tokens": total_in, "total_output_tokens": total_out, "total_tokens": total_tok}
    }
    with open(os.path.join(args.out_dir, "step_debug_stats.json"), "w", encoding="utf-8") as f:
        json.dump(debug_stats, f, ensure_ascii=False, indent=2)

    print("[DONE] testcase_generate completed.")

if __name__ == "__main__":
    main()
