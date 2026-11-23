# OSU 2018 Engineering Strategic Plan — MGEL RAG Analysis & Verification

## 1. Target Query

> "According to OSU's 2018 Engineering Strategic Plan, how did the university plan to measure the success of the MGEL degree program? Identify enrollment, quality, and curriculum expansion targets, including deadlines. Cite the relevant page."

---

## 2. Execution Status & Credential Disclosure

- **Real PDF Ingestion & Vector Retrieval:** **VERIFIED** using `PyPDFLoader` on `Engineering_Strategic_Plan_2018.pdf`, `RecursiveCharacterTextSplitter` (`chunk_size=700`, `chunk_overlap=250`, Goal-aware separators), and local `sentence-transformers/all-mpnet-base-v2` embeddings in `InMemoryVectorStore`.
- **Deep Agents Offloading & Delegation Pipeline:** **VERIFIED** deterministically via `tests/test_osu_rag.py` (`search_documentation` -> `StateBackend.upload_files()` -> `/retrieved/{batch_id}/chunk_*.md` -> `task(subagent_type="chunk-analyst")` -> `read_file`).
- **Live Gemini LLM Execution (`GOOGLE_API_KEY`):** **BLOCKED (Missing Credentials)**. Neither `GOOGLE_API_KEY` nor `GEMINI_API_KEY` is configured in this environment. No live Gemini API response was fabricated. Once `GOOGLE_API_KEY` is set in `.env`, run `python agent.py --source osu` for live Gemini synthesis.

---

## 3. Grounded Answer (Directly Synthesized from Retrieved Chunk 0)

According to **printed page 12** (**PDF page 13**, 0-indexed page `12`) of The Ohio State University's *College of Engineering Strategic Plan (May 16, 2018)*, under the **Professional and Distance Education Programs (PDEP)** goal, the university planned to measure the success of the **Master of Global Engineering Leadership (MGEL)** degree program across three dimensions:

1. **Enrollment Target:**
   - **Strategy:** Grow enrollment of the MGEL degree.
   - **Metric & Deadline:** **Enrollment of 40 students by 2020.**
2. **Quality Target:**
   - **Strategy:** Manage the MGEL degree so it is delivered with the highest quality to promote enrollment growth.
   - **Metric & Deadline:** **U.S. News & World Report online engineering degree programs ranking is within the top 25** *(no specific calendar deadline is listed for this metric in the table)*.
3. **Curriculum Expansion Target:**
   - **Strategy:** Increase the number of technical tracks in the MGEL degree by increasing the number of graduate-level courses offered in distance learning format to promote enrollment growth.
   - **Metric & Deadline:** **Add two new technical tracks by 2020.**

**Citation:**
- Source: [OSU College of Engineering Strategic Plan (May 16, 2018)](https://oaa.osu.edu/sites/default/files/uploads/unit-level-strategic-planning/colleges/engineering/Engineering_Strategic-Plan_2018.pdf) — **Printed Page 12** *(PDF Page 13; PyPDFLoader 0-indexed `page=12`)*.
- See full retrieved chunks in [osu_mgel_retrieved_chunks.md](osu_mgel_retrieved_chunks.md).

---

## 4. Fact Verification Against Expected Ground Truth

| Dimension | Expected Fact | Retrieved Verbatim from Printed Page 12 (Chunk 0) | Status |
| :--- | :--- | :--- | :--- |
| **Enrollment** | Enrollment of 40 MGEL students by 2020 | `Grow enrollment of MGEL degree.` $\to$ `Enrollment of 40 students by 2020.` | **PASS** |
| **Quality** | Top 25 U.S. News & World Report online engineering degree program ranking | `Manage the MGEL degree so it’s delivered with the highest quality to promote enrollment growth.` $\to$ `US News and World Report online engineering degree programs ranking is within top 25.` | **PASS** |
| **Curriculum Expansion** | Addition of two new technical tracks by 2020 | `Increase the number of technical tracks in the MGEL degree...` $\to$ `Add two new technical tracks by 2020.` | **PASS** |
| **Page Numbering** | Printed page 12 (handling 0-indexed PDF pages) | `printed_page=12`, `pdf_page=13`, `page=12` (0-indexed; offset caused by unnumbered Mission/Vision insert at PDF page 2) | **PASS** |

---

## 5. Direct-LLM Baseline vs. Deep Agents RAG Comparison

> **Note:** Both live Gemini baseline (`baseline.py`) and live Gemini RAG (`agent.py --source osu`) require `GOOGLE_API_KEY` to execute live model calls. The table below documents the verified architectural differences and observed failure modes of a direct LLM without retrieval vs. the implemented Deep Agents RAG system.

| Capability | Direct-LLM Baseline (`baseline.py`, `tools=[]`) | Deep Agents RAG (`agent.py --source osu`) |
| :--- | :--- | :--- |
| **Access to 2018 OSU PDF** | None (relies solely on parametric training weights). | Directly ingests `Engineering_Strategic_Plan_2018.pdf` via `PyPDFLoader` and `InMemoryVectorStore`. |
| **Specific Quantitative Targets** | High risk of hallucinating or refusing exact numbers (`40 students by 2020`, `top 25`, `two new technical tracks by 2020`). | Deterministically retrieves all 3 exact strategy-metric pairs in Rank #1 chunk (`k=4`). |
| **Page-Level Citations** | Cannot reliably cite printed page 12 vs. PDF page 13 (0-indexed `12`). | Preserves `printed_page`, `pdf_page`, and 0-indexed `page` in offloaded Markdown headers and citations. |
| **Context Hygiene & Security** | N/A (no retrieved documents). | Offloads chunks to `/retrieved/{batch_id}/chunk_*.md` in `StateBackend`; `chunk-analyst` reads via `read_file` under strict untrusted-data prompt boundaries. |
| **Verifiability** | Unverifiable without external source lookup. | Every claim maps directly to saved chunks in `StateBackend` and `artifacts/osu_mgel_retrieved_chunks.md`. |
