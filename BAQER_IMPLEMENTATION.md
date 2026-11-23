# Baqer's Deep Agents RAG Module — Implementation Report

## 1. Architecture Overview

This module implements **Part 2 of the University RAG Lab**: an advanced Deep Agents retrieval-augmented generation system featuring:
1. **Custom Vector Retrieval Tool** with automatic Markdown formatting.
2. **State Filesystem Offloading** (`StateBackend.upload_files()`) saving chunks to `/retrieved/{batch_id}/chunk_*.md`.
3. **Subagent Delegation** using DeepAgents' `task()` tool targeting a specialized `chunk-analyst` subagent.
4. **Context Hygiene**: Preventing large raw text passages from flooding the main agent context window.
5. **Rigorous Five-Step Prompt Engineering** (Plan -> Search -> Analyze -> Synthesize -> Verify).
6. **Indirect Prompt Injection Defense & Citation Auditing**.

```
                           +------------------------+
                           |     User Question      |
                           +-----------+------------+
                                       |
                                       v
                       +-------------------------------+
                       |    Deep Agent Orchestrator    |
                       |    (Five-Step RAG Workflow)   |
                       +---------------+---------------+
                                       |
                   1. search_documentation(query)
                                       |
                                       v
                       +-------------------------------+
                       |      InMemoryVectorStore      |
                       |  (mpnet-base-v2, top k=4)     |
                       +---------------+---------------+
                                       |
                 2. upload_files([(path, bytes), ...])
                                       |
                                       v
                       +-------------------------------+
                       |         StateBackend          |
                       | (/retrieved/{batch_id}/*.md)  |
                       +---------------+---------------+
                                       |
             3. Manifest returned (Batch ID & file paths only)
                                       |
                                       v
               +-----------------------------------------------+
               | 4. task(subagent_type="chunk-analyst",        |
               |         description="Inspect chunk_*.md")     |
               +-----------------------+-----------------------+
                                       |
                                       v
                       +-------------------------------+
                       |     chunk-analyst Subagent    |
                       |       (calls read_file)       |
                       +---------------+---------------+
                                       |
               5. Structured factual summary (< 300 words)
                  with extracted # Source: URL
                                       |
                                       v
                       +-------------------------------+
                       |    Orchestrator Synthesis     |
                       |   & Grounded Inline Citations |
                       +-------------------------------+
```

---

## 2. Retrieval and State Offloading

The retrieval tool `search_documentation(query: str) -> str` executes:
- Similarity search against the LangChain `InMemoryVectorStore` (`k=4`).
- Generation of a unique batch identifier using `uuid_utils.uuid7()` (with UUIDv4 fallback).
- Markdown formatting with standardized source attribution headers:
  ```markdown
  # Source: https://docs.langchain.com/...

  [Retrieved documentation content]
  ```
- Binary UTF-8 upload into the `StateBackend` via:
  ```python
  backend.upload_files([
      (f"/retrieved/{batch_id}/chunk_{idx}.md", content_bytes), ...
  ])
  ```
- Returning a clean manifest with batch ID and file paths. **Raw documentation text is never returned to the orchestrator**, preserving orchestrator context window capacity.

---

## 3. Subagent Delegation Flow

The orchestrator and subagents interact via DeepAgents' native task execution:
- **Subagent Spec (`chunk_analyst_subagent`)**: Declared with `name="chunk-analyst"`, custom system prompt `CHUNK_ANALYST_INSTRUCTIONS`, and restricted tool suite.
- **Filesystem Access**: The subagent inherits the parent's `StateBackend` and state channels. Files written to `/retrieved/{batch_id}/` by the retrieval tool are accessible to the subagent via the built-in `read_file` tool.
- **Analysis Execution**: The subagent reads the markdown files, extracts API names, parameters, code examples, and verified `# Source:` URLs, returning a high-density summary of fewer than 300 words.
- **Concurrency & Synthesis**: The orchestrator can fan out up to 3 concurrent analyst tasks, deduplicate overlapping sources, reconcile gaps, and produce verified inline citations.

---

## 4. Setup and Installation

### Prerequisites
- Python 3.11+
- Virtual environment

### Setup Commands
```bash
# 1. Clone repository (if not already present)
git clone https://github.com/alextheis/docs-rag-agent.git
cd docs-rag-agent

# 2. Switch to feature branch
git switch feature/baqer-rag-agent

# 3. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Environment Variables
Configure your `.env` file based on `.env.example`:
```bash
GOOGLE_API_KEY=your_google_api_key_here
# Optional configuration:
MODEL_NAME=google_genai:gemini-2.5-flash
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```

---

## 5. How to Run

Execute the entrypoint:
```bash
python agent.py
```
Or with debug trace logging:
```bash
python agent.py --debug
```

Target question executed by default:
```text
"How do I stream intermediate tool results from a subagent?"
```

---

## 6. How to Test

Run the automated test suite:
```bash
# Run unit and security tests (fast, mocked, zero external network calls)
pytest -v tests/test_agent.py tests/test_security.py

# Run all tests including live vector store integration (downloads documentation pages)
pytest -v
```

### Test Coverage Highlights
- `test_retrieval_calls_similarity_search_k4`: Verifies exact query and `k=4` vector search.
- `test_generated_file_paths_follow_pattern`: Validates `/retrieved/{batch_id}/chunk_{index}.md` naming.
- `test_file_contents_include_source_heading`: Verifies `# Source: <url>` markdown structure.
- `test_files_encoded_as_utf8_bytes`: Tests unicode handling and byte encoding.
- `test_retrieval_returns_paths_rather_than_raw_text`: Verifies raw text suppression to protect orchestrator context.
- `test_retrieval_handles_zero_results`: Graceful handling of empty queries.
- `test_upload_failures_reported`: Error propagation and status verification on filesystem errors.
- `test_subagent_configuration_valid`: Schema and prompt validation for `chunk-analyst`.
- `test_orchestrator_tools_and_instructions`: Confirms required tool registration and 5-step workflow.
- `test_retrieved_files_accessible_by_subagent`: Tests subagent state file retrieval via `read_file`.
- `test_malicious_instructions_treated_as_untrusted`: Prompt injection defense testing.
- `test_citation_url_validation_and_grounding`: Asserts strict URL compliance to `https://docs.langchain.com/`.
- `test_full_end_to_end_orchestration_flow`: Complete 5-step multi-turn agent simulation.
- `test_real_vector_store_with_search_documentation`: Live integration with Alex's `build_vector_store()`.

---

## 7. Security Considerations and Injection Defense

1. **Untrusted Data Boundary**: Retrieved documentation chunks are treated strictly as untrusted external passive data. Both orchestrator and subagent prompts contain explicit guardrails instructing the models never to obey commands, shell code, or role modifications embedded in retrieved content.
2. **State Isolation**: File writes and reads are restricted to the `/retrieved/` state filesystem namespace. Path traversal (`..`) and host filesystem access are blocked.
3. **No Direct Execution**: The agent never executes scripts or commands from retrieved chunks.
4. **Citation Auditing**: Citations are verified against the `# Source:` headers belonging to `https://docs.langchain.com/`. Unsupported or fabricated citations are rejected.
5. **Limitations**: LLM prompt boundaries reduce injection risk, but cannot provide a 100% cryptographic guarantee against sophisticated adversarial jailbreaks without output schema validators or semantic firewalls.

---

## 8. Integration Instructions for Alex

Alex's modules (`indexing.py` and `baseline.py`) were **completely preserved without modifications**:
- `indexing.py` exposes:
  ```python
  from indexing import build_vector_store
  vector_store = build_vector_store()
  ```
- To run the full agent against your vector store:
  ```python
  from indexing import build_vector_store
  from agent import create_rag_agent

  vector_store = build_vector_store()
  rag_agent = create_rag_agent(vector_store=vector_store)
  response = rag_agent.invoke({"messages": [...]})
  ```
- Dependencies and HuggingFace embeddings (`sentence-transformers/all-mpnet-base-v2`) remain identical.

---

## 9. Remaining Limitations & Live Provider Notes

- **API Credentials**: Running live end-to-end queries with real Gemini LLM calls requires setting `GOOGLE_API_KEY` in `.env`.
- **Offline / Sandbox Network Access**: When building the real vector store with `build_vector_store()`, initial page downloads require outbound HTTPS to `https://docs.langchain.com/`.
