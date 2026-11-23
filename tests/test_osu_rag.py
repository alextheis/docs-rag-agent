"""Deterministic automated tests for the OSU 2018 Engineering Strategic Plan PDF RAG pipeline.

Validates:
1. PyPDFLoader ingestion preserves source URL, 0-indexed PDF page, 1-indexed PDF page,
   and printed document page numbers (specifically printed page 12 == PDF page 13 / index 12).
2. Table extraction integrity on printed page 12 preserves the exact strategy -> metric
   pairing for enrollment, quality, and curriculum expansion.
3. Real InMemoryVectorStore retrieval with `sentence-transformers/all-mpnet-base-v2`
   deterministically retrieves all three MGEL targets and deadlines in top k=4.
4. `search_documentation` offloads OSU chunks to StateBackend with `# Source:` and
   `# Printed Page:` / `# PDF Page:` metadata headers without leaking raw text.
5. Full Deep Agents orchestrator -> search_documentation -> StateBackend -> task(chunk-analyst)
   -> read_file -> synthesis flow executes cleanly using the OSU index without duplicating logic.
6. Configurable index factory (`build_index_for_source`) and CLI argument parser support
   both `langchain` and `osu` modes.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.messages import AIMessage, HumanMessage

import agent
from agent import (
    INSTRUCTIONS,
    OSU_TARGET_QUERY,
    chunk_analyst_subagent,
    create_rag_agent,
    format_chunk_markdown,
    parse_args,
    search_documentation,
    set_retrieval_context,
)
from osu_indexing import (
    DEFAULT_PDF_PATH,
    OSU_PDF_URL,
    build_osu_vector_store,
    extract_printed_page_number,
    load_osu_pdf,
)
from tests.test_agent import MockChatModelWithTools

# Ensure offline HF cache usage in sandboxed test runs when weights are already cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="module")
def osu_pages():
    """Load OSU PDF pages once per test module."""
    if not DEFAULT_PDF_PATH.exists():
        pytest.skip(f"OSU PDF not cached at {DEFAULT_PDF_PATH}")
    return load_osu_pdf(DEFAULT_PDF_PATH)


@pytest.fixture(scope="module")
def osu_vector_store():
    """Build real OSU InMemoryVectorStore once per test module."""
    if not DEFAULT_PDF_PATH.exists():
        pytest.skip(f"OSU PDF not cached at {DEFAULT_PDF_PATH}")
    return build_osu_vector_store(DEFAULT_PDF_PATH)


# ---------------------------------------------------------------------------
# Test 1: PDF loading & zero-indexed vs 1-indexed vs printed page metadata
# ---------------------------------------------------------------------------
def test_osu_pdf_page_metadata_and_numbering(osu_pages):
    """Verify PyPDFLoader metadata preserves source and distinguishes PDF vs printed pages."""
    assert len(osu_pages) == 30

    # PDF index 0 (PDF page 1) is printed page 1
    assert osu_pages[0].metadata["page"] == 0
    assert osu_pages[0].metadata["pdf_page"] == 1
    assert osu_pages[0].metadata["printed_page"] == 1
    assert osu_pages[0].metadata["source"] == OSU_PDF_URL

    # PDF index 1 (PDF page 2) is the unnumbered Mission/Vision insert
    assert osu_pages[1].metadata["page"] == 1
    assert osu_pages[1].metadata["pdf_page"] == 2
    assert osu_pages[1].metadata["printed_page"] is None

    # PDF index 12 (PDF page 13) is printed page 12 containing the MGEL table
    mgel_page = osu_pages[12]
    assert mgel_page.metadata["page"] == 12
    assert mgel_page.metadata["pdf_page"] == 13
    assert mgel_page.metadata["printed_page"] == 12
    assert mgel_page.metadata["source"] == OSU_PDF_URL
    assert "MGEL" in mgel_page.page_content


# ---------------------------------------------------------------------------
# Test 2: Strategy-to-Metric table extraction integrity on printed page 12
# ---------------------------------------------------------------------------
def test_mgel_table_strategy_metric_relationships(osu_pages):
    """Verify PDF text extraction preserves each MGEL strategy immediately before its metric."""
    text = " ".join(osu_pages[12].page_content.split())

    # 1. Enrollment strategy -> 40 students by 2020
    idx_enroll_strat = text.index("Grow enrollment of MGEL degree.")
    idx_enroll_metric = text.index("Enrollment of 40 students by 2020.")
    assert idx_enroll_strat < idx_enroll_metric
    assert idx_enroll_metric - idx_enroll_strat < 100

    # 2. Quality strategy -> US News and World Report online engineering top 25
    idx_qual_strat = text.index(
        "Manage the MGEL degree so it’s delivered with the highest quality"
    )
    idx_qual_metric = text.index(
        "US News and World Report online engineering degree programs ranking is within top 25."
    )
    assert idx_enroll_metric < idx_qual_strat < idx_qual_metric
    assert idx_qual_metric - idx_qual_strat < 150

    # 3. Curriculum expansion strategy -> Add two new technical tracks by 2020
    idx_curr_strat = text.index("Increase the number of technical tracks in the MGEL degree")
    idx_curr_metric = text.index("Add two new technical tracks by 2020.")
    assert idx_qual_metric < idx_curr_strat < idx_curr_metric
    assert idx_curr_metric - idx_curr_strat < 220


# ---------------------------------------------------------------------------
# Test 3: Deterministic retrieval of all 3 expected MGEL facts (k=4)
# ---------------------------------------------------------------------------
def test_osu_vector_store_retrieves_all_three_mgel_facts(osu_vector_store):
    """Verify similarity_search(OSU_TARGET_QUERY, k=4) retrieves all 3 required MGEL facts."""
    results = osu_vector_store.similarity_search(OSU_TARGET_QUERY, k=4)
    assert len(results) == 4

    # Top chunk must come from printed page 12 (PDF page 13, 0-indexed 12)
    top_chunk = results[0]
    assert top_chunk.metadata["printed_page"] == 12
    assert top_chunk.metadata["pdf_page"] == 13
    assert top_chunk.metadata["page"] == 12

    combined_text = " ".join(doc.page_content for doc in results)
    normalized = " ".join(combined_text.split()).lower()

    # Fact 1: Enrollment of 40 MGEL students by 2020
    assert "40 students by 2020" in normalized
    # Fact 2: Top 25 U.S. News & World Report online engineering degree program ranking
    assert "us news and world report" in normalized
    assert "top 25" in normalized
    # Fact 3: Addition of two new technical tracks by 2020
    assert "add two new technical tracks by 2020" in normalized


# ---------------------------------------------------------------------------
# Test 4: Offloaded Markdown preserves page metadata without leaking raw text
# ---------------------------------------------------------------------------
def test_osu_offload_preserves_page_metadata_and_hides_raw_text(osu_vector_store):
    """Verify search_documentation writes # Source and # Printed Page / # PDF Page headers."""
    backend = StateBackend()
    set_retrieval_context(osu_vector_store, backend)

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_documentation",
                    "args": {"query": OSU_TARGET_QUERY},
                    "id": "call_osu_search",
                }
            ],
        ),
        AIMessage(content="Retrieved OSU chunks."),
    ]
    model = MockChatModelWithTools(messages=iter(responses))
    rag_agent = create_rag_agent(
        model=model, vector_store=osu_vector_store, backend=backend
    )

    with patch("agent.generate_batch_id", return_value="osu_batch_01"):
        result = rag_agent.invoke({"messages": [HumanMessage(content=OSU_TARGET_QUERY)]})

    # Manifest returned to orchestrator contains file paths & page tags, NOT raw chunk text
    tool_msg = next(
        m for m in result["messages"] if getattr(m, "type", "") == "tool"
    )
    assert "/retrieved/osu_batch_01/chunk_0.md" in tool_msg.content
    assert "printed page 12" in tool_msg.content
    assert "PDF page 13" in tool_msg.content
    assert "Enrollment of 40 students by 2020" not in tool_msg.content

    # Offloaded chunk_0.md in StateBackend contains full headers and all 3 MGEL facts
    files = result.get("files", {})
    chunk0_text = files["/retrieved/osu_batch_01/chunk_0.md"]["content"]
    assert chunk0_text.startswith(f"# Source: {OSU_PDF_URL}")
    assert "# Printed Page: 12 | PDF Page: 13 (0-indexed: 12)" in chunk0_text
    assert "Enrollment of 40 students by 2020." in chunk0_text
    assert "top 25" in chunk0_text.lower()
    assert "Add two new technical tracks by 2020." in chunk0_text


# ---------------------------------------------------------------------------
# Test 5: Full End-to-End Deep Agents Delegation on OSU PDF Index
# ---------------------------------------------------------------------------
def test_osu_full_rag_delegation_pipeline(osu_vector_store):
    """Test Search -> StateBackend Offload -> chunk-analyst task -> read_file -> Synthesis."""
    backend = StateBackend()
    set_retrieval_context(osu_vector_store, backend)

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_documentation",
                    "args": {"query": OSU_TARGET_QUERY},
                    "id": "call_search_osu",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {
                        "subagent_type": "chunk-analyst",
                        "description": "Read /retrieved/osu_e2e_12/chunk_0.md and extract MGEL targets, deadlines, and page numbers.",
                    },
                    "id": "call_task_osu",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "/retrieved/osu_e2e_12/chunk_0.md"},
                    "id": "call_read_osu",
                }
            ],
        ),
        AIMessage(
            content=(
                "Printed Page 12 (PDF Page 13, 0-indexed: 12) lists three MGEL metrics: "
                "1) Enrollment of 40 students by 2020; "
                "2) US News and World Report online engineering degree programs ranking within top 25; "
                "3) Add two new technical tracks by 2020. "
                f"Source: {OSU_PDF_URL}"
            )
        ),
        AIMessage(
            content=(
                "According to printed page 12 (PDF page 13) of OSU's 2018 Engineering Strategic Plan:\n"
                "- Enrollment: 40 MGEL students by 2020.\n"
                "- Quality: Top 25 U.S. News & World Report online engineering degree programs ranking.\n"
                "- Curriculum expansion: Add two new technical tracks by 2020.\n"
                f"Source: {OSU_PDF_URL} (Printed Page 12 / PDF Page 13)."
            )
        ),
    ]

    model = MockChatModelWithTools(messages=iter(responses))
    orchestrator = create_deep_agent(
        model=model,
        tools=[search_documentation],
        backend=backend,
        system_prompt=INSTRUCTIONS,
        subagents=[chunk_analyst_subagent],
    )

    with patch("agent.generate_batch_id", return_value="osu_e2e_12"):
        result = orchestrator.invoke(
            {"messages": [HumanMessage(content=OSU_TARGET_QUERY)]}
        )

    assert "/retrieved/osu_e2e_12/chunk_0.md" in result.get("files", {})
    final_answer = result["messages"][-1].content
    assert "40" in final_answer and "2020" in final_answer
    assert "top 25" in final_answer.lower()
    assert "two new technical tracks" in final_answer.lower()
    assert "12" in final_answer


# ---------------------------------------------------------------------------
# Test 6: CLI & Configurable Index Selector
# ---------------------------------------------------------------------------
def test_cli_source_configuration():
    """Verify agent.py CLI parser cleanly supports both langchain and osu sources."""
    default_args = parse_args([])
    assert default_args.source == "langchain"
    assert default_args.debug is False

    osu_args = parse_args(["--source", "osu", "--debug"])
    assert osu_args.source == "osu"
    assert osu_args.debug is True

    with pytest.raises(ValueError, match="Unsupported index source"):
        agent.build_index_for_source("unknown_corpus")


def test_printed_page_extractor_edge_cases():
    """Verify printed page header parser handles numbered and unnumbered pages."""
    assert (
        extract_printed_page_number(
            "College of Engineering Strategic Plan  May 16, 2018 \n  \n12 \n"
        )
        == 12
    )
    assert (
        extract_printed_page_number("The Ohio State University\nMission/Vision Statement")
        is None
    )
