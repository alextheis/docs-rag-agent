# OSU 2018 Engineering Strategic Plan — Deep Agents RAG Guide

This guide explains how the existing Deep Agents RAG application (`agent.py`) was adapted to ingest and query Ohio State University's **2018 College of Engineering Strategic Plan** (`Engineering_Strategic_Plan_2018.pdf`) while preserving Alex's original LangChain documentation modules (`indexing.py`, `baseline.py`).

---

## 1. What Was Added & Preserved

- **Preserved untouched:** `indexing.py` and `baseline.py` remain completely unmodified.
- **New PDF Ingestion Module (`osu_indexing.py`):**
  - Downloads and caches `Engineering_Strategic_Plan_2018.pdf` locally under `data/` (ignored by Git).
  - Loads pages using `PyPDFLoader` (`langchain-community` + `pypdf`).
  - Preserves source URL and three explicit page metadata fields:
    - `page`: original 0-indexed PDF page index from `PyPDFLoader`
    - `pdf_page`: 1-indexed PDF page number (`page + 1`)
    - `printed_page`: printed page number parsed from the page header
  - Splits text with Goal-aware `RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=250, separators=["\nGoal", "\n\n", "\n", " ", ""])` so that the MGEL strategy/metric table on printed page 12 stays intact and ranks **#1** in similarity search.
  - Embeds chunks locally using `sentence-transformers/all-mpnet-base-v2` (normalized) into a LangChain `InMemoryVectorStore`.
- **Configurable Agent (`agent.py`):**
  - Reuses the exact same `search_documentation` tool, `StateBackend` offloading (`/retrieved/{batch_id}/chunk_*.md`), `chunk-analyst` subagent, and 5-step workflow without duplicating logic.
  - Automatically includes `# Printed Page: <n> | PDF Page: <m> (0-indexed: <i>)` headers in offloaded Markdown chunks whenever PDF page metadata is present, while keeping LangChain web doc formatting unchanged.
  - Supports `--source langchain` (default) and `--source osu` via CLI flags.

---

## 2. Page Numbering Note (0-Indexed vs. Printed Page 12)

`Engineering_Strategic_Plan_2018.pdf` has **30 PDF pages** and includes an unnumbered *Mission/Vision Statement* insert on PDF page 2 (`page=1`). Consequently:
- PDF index `0` (PDF page 1) $\to$ **Printed Page 1**
- PDF index `1` (PDF page 2) $\to$ *Unnumbered Mission/Vision insert* (`printed_page=None`)
- PDF index `2` (PDF page 3) $\to$ **Printed Page 2**
- **PDF index `12` (PDF page 13) $\to$ Printed Page 12** *(contains the PDEP/MGEL strategy and metrics table)*

`osu_indexing.py` parses the printed page header directly and stores all three numbers so citations are unambiguous.

---

## 3. Setup Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install all dependencies (including langchain-community and pypdf)
pip install -r requirements.txt

# Configure Gemini API key for live LLM execution
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=<your_key>
```

---

## 4. Exact Commands to Run

### A. Inspect OSU PDF Indexing & Top-4 Retrieval (No API Key Required)
```bash
python osu_indexing.py
```

### B. Run Deep Agents RAG on the OSU Strategic Plan (Requires `GOOGLE_API_KEY`)
```bash
# Default OSU MGEL target query
python agent.py --source osu

# With structured execution trace
python agent.py --source osu --debug

# Custom question against the OSU PDF
python agent.py --source osu --query "What are the PDEP office goals by 2020?"
```

### C. Run Original LangChain Documentation RAG (Preserved Default)
```bash
python agent.py
# Equivalent to:
python agent.py --source langchain
```

### D. Run Direct-LLM Baseline for Comparison (Requires `GOOGLE_API_KEY`)
```bash
python baseline.py
```

---

## 5. Automated Testing

```bash
# Run all unit, security, and deterministic OSU PDF RAG tests (27 tests)
pytest -v tests/test_agent.py tests/test_security.py tests/test_osu_rag.py

# Run only the OSU PDF RAG test suite (7 tests)
pytest -v tests/test_osu_rag.py

# Run LangChain live-download integration test (requires network access)
pytest -v tests/test_integration.py
```

---

## 6. Saved Artifacts

- [artifacts/osu_mgel_retrieved_chunks.md](artifacts/osu_mgel_retrieved_chunks.md) — Actual top-4 retrieved chunks with source URL and page metadata headers.
- [artifacts/osu_mgel_rag_result.md](artifacts/osu_mgel_rag_result.md) — Grounded MGEL answer, page-specific citations, fact verification checklist, and baseline vs. RAG comparison.
- [artifacts/osu_test_results.txt](artifacts/osu_test_results.txt) — Verbatim `pytest` output for all 27 automated tests.
