"""Integration tests combining Alex's indexing module and Baqer's agent module."""

import pytest
from deepagents.backends import StateBackend
from langchain_core.messages import AIMessage, HumanMessage

import agent
from agent import (
    INSTRUCTIONS,
    chunk_analyst_subagent,
    create_rag_agent,
    search_documentation,
    set_retrieval_context,
)
from tests.test_agent import MockChatModelWithTools


@pytest.mark.integration
def test_real_vector_store_with_search_documentation():
    """Verify integration between Alex's build_vector_store and Baqer's search_documentation tool."""
    from indexing import build_vector_store

    vector_store = build_vector_store()
    backend = StateBackend()
    set_retrieval_context(vector_store, backend)

    query = "How do I stream intermediate tool results from a subagent?"

    # We test inside an agent invocation context so StateBackend has valid CONFIG_KEY_SEND
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_documentation",
                    "args": {"query": query},
                    "id": "call_search_real",
                }
            ],
        ),
        AIMessage(content="Search completed and chunks saved to state."),
    ]
    model = MockChatModelWithTools(messages=iter(responses))

    app = create_rag_agent(
        model=model,
        vector_store=vector_store,
        backend=backend,
    )

    result = app.invoke({"messages": [HumanMessage(content=query)]})

    files = result.get("files", {})
    assert len(files) > 0, "Expected chunks to be saved in state backend"

    retrieved_paths = [p for p in files.keys() if p.startswith("/retrieved/")]
    assert len(retrieved_paths) <= 4
    assert len(retrieved_paths) > 0

    first_path = retrieved_paths[0]
    first_chunk = files[first_path]
    assert "content" in first_chunk
    assert first_chunk["content"].startswith("# Source: https://docs.langchain.com")
