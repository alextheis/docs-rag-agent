"""OSU Engineering Strategic Plan (2018) PDF Ingestion and Indexing Module.

Loads `Engineering_Strategic_Plan_2018.pdf` via `PyPDFLoader`, preserves source
URL and page-number metadata (0-indexed PDF page, 1-indexed PDF page, and
printed document page), splits text with `RecursiveCharacterTextSplitter`, and
indexes chunks into a LangChain `InMemoryVectorStore` using the local
`sentence-transformers/all-mpnet-base-v2` HuggingFace embedding model.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OSU_PDF_URL = (
    "https://oaa.osu.edu/sites/default/files/uploads/unit-level-strategic-planning/"
    "colleges/engineering/Engineering_Strategic-Plan_2018.pdf"
)

DEFAULT_PDF_PATH = Path("data/Engineering_Strategic_Plan_2018.pdf")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

OSU_TARGET_QUERY = (
    "According to OSU's 2018 Engineering Strategic Plan, how did the university "
    "plan to measure the success of the MGEL degree program? Identify enrollment, "
    "quality, and curriculum expansion targets, including deadlines. Cite the relevant page."
)

# Header pattern on numbered pages:
#   "College of Engineering Strategic Plan  May 16, 2018"
#   "12"
_PRINTED_PAGE_RE = re.compile(
    r"College of Engineering Strategic Plan\s+May 16,\s*2018\s*\n+\s*(\d+)\b",
    re.IGNORECASE,
)


def download_osu_pdf(
    url: str = OSU_PDF_URL,
    dest_path: str | Path = DEFAULT_PDF_PATH,
    *,
    force: bool = False,
    timeout: int = 30,
) -> Path:
    """Download the OSU 2018 Engineering Strategic Plan PDF if not cached locally."""
    target = Path(dest_path)
    if target.exists() and target.stat().st_size > 0 and not force:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def extract_printed_page_number(page_text: str) -> int | None:
    """Extract the printed page number from an OSU Strategic Plan page header.

    Note: The PDF includes an unnumbered Mission/Vision insert at PDF page 2
    (0-indexed page 1), so from printed page 2 onward the printed page number
    equals the 0-indexed PDF page number rather than `page + 1`.
    """
    match = _PRINTED_PAGE_RE.search(page_text)
    if match:
        return int(match.group(1))
    return None


def load_osu_pdf(
    pdf_path: str | Path | None = None,
    url: str = OSU_PDF_URL,
) -> list[Document]:
    """Load the OSU Strategic Plan PDF with PyPDFLoader and enrich page metadata.

    Preserves:
    - `source`: canonical OSU PDF URL
    - `local_path`: local filesystem path of the loaded PDF
    - `page`: original 0-indexed PDF page number from PyPDFLoader
    - `pdf_page`: 1-indexed PDF page number (`page + 1`)
    - `printed_page`: printed page number parsed from the page header (if present)
    """
    resolved_path = Path(pdf_path) if pdf_path is not None else DEFAULT_PDF_PATH
    if not resolved_path.exists():
        resolved_path = download_osu_pdf(url=url, dest_path=resolved_path)

    loader = PyPDFLoader(str(resolved_path))
    pages = loader.load()

    enriched: list[Document] = []
    for doc in pages:
        raw_page = doc.metadata.get("page")
        pdf_page = int(raw_page) + 1 if raw_page is not None else None
        printed_page = extract_printed_page_number(doc.page_content)

        metadata = {
            **doc.metadata,
            "source": url,
            "local_path": str(resolved_path),
            "page": int(raw_page) if raw_page is not None else None,
            "pdf_page": pdf_page,
            "printed_page": printed_page,
        }
        enriched.append(Document(page_content=doc.page_content, metadata=metadata))

    return enriched


def _build_embeddings() -> HuggingFaceEmbeddings:
    """Initialize local HuggingFace embeddings matching Alex's indexing.py."""
    model_kwargs: dict[str, object] = {}
    if os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true"):
        model_kwargs["local_files_only"] = True

    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        # Fallback to local cache if network/hub metadata lookup fails offline
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )


def build_osu_vector_store(
    pdf_path: str | Path | None = None,
    *,
    chunk_size: int = 700,
    chunk_overlap: int = 250,
) -> InMemoryVectorStore:
    """Load, split, embed, and index the OSU 2018 Engineering Strategic Plan PDF.

    Uses Goal-aware `RecursiveCharacterTextSplitter` settings (`chunk_size=700`,
    `chunk_overlap=250`, separating on `\\nGoal` boundaries) so that multi-row
    strategy/metric tables such as the MGEL table on printed page 12 remain
    cohesive and retrievable in the top-k similarity search results.
    """
    docs = load_osu_pdf(pdf_path=pdf_path)
    print(f"Loaded {len(docs)} PDF pages from OSU Engineering Strategic Plan.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\nGoal", "\n\n", "\n", " ", ""],
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split OSU PDF into {len(all_splits)} chunks.")

    embeddings = _build_embeddings()
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=all_splits)
    print(f"Indexed {len(all_splits)} OSU PDF chunks.")

    return vector_store


if __name__ == "__main__":
    store = build_osu_vector_store()
    results = store.similarity_search(OSU_TARGET_QUERY, k=4)
    print(f"\nTop {len(results)} retrieved chunks for:\n  {OSU_TARGET_QUERY}\n")
    for idx, doc in enumerate(results, start=1):
        printed = doc.metadata.get("printed_page")
        pdf_page = doc.metadata.get("pdf_page")
        zero_idx = doc.metadata.get("page")
        print(
            f"--- Chunk {idx} | Printed Page: {printed} | "
            f"PDF Page: {pdf_page} (0-indexed: {zero_idx}) ---"
        )
        print(f"Source: {doc.metadata.get('source')}")
        print(doc.page_content[:400], "...\n")
