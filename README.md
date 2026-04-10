# Test Case Generation Suite

Automated pipeline for generating Q&A test cases from PDF documents. Designed to support RAG system evaluation and chatbot testing by extracting structured topics from documents and producing grounded question-answer pairs with semantic variants.

---

## Overview

The suite consists of two core scripts orchestrated by a single pipeline runner:

```
PDF document
    │
    ▼
[1] nodetree_generate.py   — Extracts structured content tree, filters topics, builds LLM tag bank
    │
    ▼
[2] testcase_generate.py   — Generates base Q&A pairs per topic, produces variants, attaches SBERT tags
    │
    ▼
Output CSVs / JSON per document
```

---

## Capabilities

- **PDF ingestion** via three strategies (in priority order):
  - PyMuPDF extractor when an embedded TOC is detected
  - PageIndex API for documents without a TOC under the page limit
  - Page-chunk fallback for very large documents
- **LLM-powered tagging** — batched, parallelised tag bank generation grounded in document content
- **Industry-aware prompting** — tailored prompts for insurance, health, education, legal, technical, and general documents
- **Q&A generation** — base questions and grounded answers generated per topic
- **Variant generation** — configurable number of rephrasings per base question
- **SBERT semantic tagging** — attaches the most semantically relevant tags from the tag bank to each Q&A pair
- **Import-format output** — optional CSV formatted for direct test platform import (`--import_flag`)
- **Ingestion metrics** — page count, word counts, topic coverage ratio, and compression rate per document

---

## Repository Structure

```
test-case-generation/
├── node_tree_generation/
│   ├── nodetree_generate.py       # Step 1: PDF → filtered topics + tag bank
│   ├── base.py                    # TreeNode dataclass
│   ├── pymupdf_ext.py             # PyMuPDF TOC extractor
│   ├── tree_utils.py              # Tree traversal utilities
│   └── pdf_utils.py               # PDF helpers
├── test-case-creation/
│   └── testcase_generate.py       # Step 2: topics + tags → Q&A test cases
├── testing-pipeline/
│   ├── pipeline.py                # Orchestrator — runs both steps per document
│   ├── pipeline_config.yaml       # Document list and parameter configuration
│   └── documents/                 # Place input PDFs here
│       └── output/                # Generated outputs (timestamped per run)
└── requirements.txt
```

---

## Requirements

### Python
Python 3.10+

### Dependencies

```
pip install -r requirements.txt
```

| Package | Used by |
|---|---|
| `pyyaml` | pipeline.py |
| `openai` | nodetree_generate.py, testcase_generate.py |
| `pymupdf` | nodetree_generate.py |
| `pageindex` | nodetree_generate.py (optional fallback) |
| `pandas` | testcase_generate.py |
| `numpy` | testcase_generate.py |
| `scikit-learn` | testcase_generate.py |
| `sentence-transformers` | testcase_generate.py |

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM calls |
| `PAGEINDEX_API_KEY` | Conditional | Required only if processing PDFs without an embedded TOC and under the page limit |
| `HF_TOKEN` | Conditional | Required only if using a private or gated SBERT model |
| `SBERT_MODEL_NAME` | No | Override the default SBERT model (`sentence-transformers/all-MiniLM-L6-v2`) |

---

## Configuration

Edit `testing-pipeline/pipeline_config.yaml` to define your documents and parameters:

```yaml
defaults:                          # Applied to every document unless overridden
  model: gpt-5-mini
  industry: insurance
  base_questions_per_topic: 3      # Base Q&A pairs generated per topic
  variations: 2                    # Rephrasings per base question
  workers: 20                      # Parallel API workers
  max_pageindex_pages: 300         # Page limit before switching to page-chunk fallback
  debug: false
  import_flag: false               # Output an import-format CSV for test platforms

documents:
  - input: ./documents/my-policy.pdf
  - input: ./documents/member-guide.pdf
    industry: health               # Per-document override
  - input: ./documents/handbook.pdf
    base_questions_per_topic: 5
    import_flag: true
```

**Supported industry values:** `insurance`, `health`, `medical`, `education`, `legal`, `technical`, `software`, `engineering`, `math`, `scientific` — or leave blank to use generic prompts.

---

## Usage

### Run the full pipeline

```bash
python testing-pipeline/pipeline.py \
  --config testing-pipeline/pipeline_config.yaml \
  --out_dir testing-pipeline/output
```

### Run a subset of documents

```bash
python testing-pipeline/pipeline.py \
  --config testing-pipeline/pipeline_config.yaml \
  --out_dir testing-pipeline/output \
  --docs my-policy member-guide
```

---

## Output Files

Each document gets its own timestamped folder under `out_dir/`:

| File | Description |
|---|---|
| `*_pymupdf_tree.json` | Raw extracted content tree (PyMuPDF path) |
| `index_entries_flat.json` | All flattened document entries |
| `filtered_entries.json` | Topics (entries with quote length > 100 chars) |
| `tag_bank.json` | LLM-generated semantic tags grounded in document content |
| `document_ingestion_metrics.json` | Page count, word counts, coverage ratio, compression rate |
| `step2_qa_raw.json` | Raw base Q&A pairs before tagging |
| `step2_qa_docgrounded.csv` | Base Q&A pairs with source citations |
| `step3_qa_with_variants_docgrounded.csv` | Final Q&A including all variant questions |
| `step3_qa_with_metatags.csv` / `.json` | Final Q&A with SBERT-attached semantic tags |
| `step3_tag_attach_stats.json` | Tag attachment coverage metrics |
| `step3_import_format_*.csv` | Import-ready CSV (only when `import_flag: true`) |
| `step_debug_stats.json` | LLM token usage and timing stats |

---

## Sample Output

A completed sample run on the RACV Motor Insurance PDS is available under:
```
testing-pipeline/output/20260324_210244_RACV motor-insurance-pds-current/
```
102 topics → 306 base questions → 612 final Q&A rows.
