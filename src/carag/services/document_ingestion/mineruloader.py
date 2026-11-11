"""LangChain PDF loader using local MinerU pipeline with PyPDF fallback."""

import tempfile
import math
from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from loguru import logger
from pypdf import PdfReader
from typing_extensions import override

from carag.core.config import get_settings
from carag.services.document_ingestion.pdfmdconverter import parse_doc_to_markdown
from carag.services.document_ingestion.utils import fix_markdown_tables


class MineruLoader(BaseLoader):
    """Load PDF pages using local MinerU pipeline, falling back to PyPDF."""

    def __init__(self, file_path: str) -> None:
        self.file_path = str(file_path)

    @override
    def lazy_load(self) -> Iterator[Document]:
        file_path = Path(self.file_path).expanduser().resolve()

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are supported: {file_path}")
        if not file_path.is_file():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        settings = get_settings()
        batch_size = max(1, int(settings.MINERU_PAGE_BATCH_SIZE))
        prepared_pages: list[tuple[int, str]] = []
        empty_pages: list[int] = []

        # Determine the source page count once. Each MinerU invocation below
        # receives an explicit, inclusive 0-based page range.
        with file_path.open("rb") as pdf_stream:
            reader = PdfReader(pdf_stream)
            total_pages = len(reader.pages)
        if total_pages < 1:
            raise RuntimeError(f"PDF contains no pages: {file_path}")

        total_batches = math.ceil(total_pages / batch_size)

        # Keep intermediate files alive while yielded Documents are consumed,
        # but isolate each batch's MinerU output to avoid stale middle-JSON reuse.
        with tempfile.TemporaryDirectory(prefix="agrichat_mineru_") as temp_dir:
            try:
                for batch_index, start_page_id in enumerate(
                    range(0, total_pages, batch_size), start=1
                ):
                    end_page_id = min(start_page_id + batch_size, total_pages) - 1
                    expected_pages = set(
                        range(start_page_id + 1, end_page_id + 2)
                    )
                    batch_output_dir = (
                        Path(temp_dir) / f"batch_{batch_index:04d}"
                    )

                    batch_markdowns = parse_doc_to_markdown(
                        [file_path],
                        str(batch_output_dir),
                        start_page_id=start_page_id,
                        end_page_id=end_page_id,
                        write_markdown=False,
                    )

                    actual_pages = [
                        int(item["page_number"]) for item in batch_markdowns
                    ]
                    if len(actual_pages) != len(set(actual_pages)):
                        raise RuntimeError(
                            f"MinerU returned duplicate page records in batch "
                            f"{batch_index}/{total_batches}: {actual_pages}."
                        )
                    missing_pages = sorted(expected_pages - set(actual_pages))
                    unexpected_pages = sorted(set(actual_pages) - expected_pages)
                    if missing_pages or unexpected_pages:
                        raise RuntimeError(
                            f"MinerU page coverage mismatch in batch "
                            f"{batch_index}/{total_batches}: missing={missing_pages}, "
                            f"unexpected={unexpected_pages}."
                        )

                    batch_content_pages = 0
                    for page_data in sorted(
                        batch_markdowns,
                        key=lambda item: int(item["page_number"]),
                    ):
                        page_number = int(page_data["page_number"])
                        page_content = str(page_data.get("content") or "")

                        if not page_content.strip():
                            empty_pages.append(page_number)
                            continue

                        page_content = fix_markdown_tables(page_content)
                        if page_content.strip():
                            prepared_pages.append((page_number, page_content))
                            batch_content_pages += 1
                        else:
                            empty_pages.append(page_number)

                    logger.info(
                        "MinerU page batch {}/{} completed: PDF pages {}–{}, "
                        "{} page(s) with text, {} without text.",
                        batch_index,
                        total_batches,
                        start_page_id + 1,
                        end_page_id + 1,
                        batch_content_pages,
                        len(expected_pages) - batch_content_pages,
                    )
                    del batch_markdowns

                if not prepared_pages:
                    raise RuntimeError(
                        f"MinerU returned no usable page content for {file_path}."
                    )

            except Exception:
                # Do not yield a partial MinerU result and then append a full-PDF
                # fallback. Clear prior batches before taking the fallback path.
                prepared_pages.clear()
                empty_pages.clear()
                logger.exception(
                    "Local MinerU pipeline failed for {}. Falling back to PyPDF.",
                    file_path.name,
                )

            else:
                prepared_pages.sort(key=lambda item: item[0])
                logger.info(
                    "Local MinerU pipeline extracted text from {} of {} PDF page(s) "
                    "in {} batch(es).",
                    len(prepared_pages),
                    total_pages,
                    total_batches,
                )
                if empty_pages:
                    logger.warning(
                        "No Markdown text was produced for PDF page(s): {}. "
                        "These pages will not create text chunks; inspect them if "
                        "they may contain important visual information.",
                        sorted(set(empty_pages)),
                    )

                for page_number, page_content in prepared_pages:
                    yield Document(
                        page_content=page_content,
                        metadata={
                            "source": str(file_path),
                            "converted_from": "pdf",
                            "file_type": "markdown",
                            "original_pdf": file_path.name,
                            "page_number": page_number,
                            "extraction_method": "mineru-pipeline",
                        },
                    )
                return

        # Fallback for text-based PDFs or when local MinerU is unavailable.
        reader = PdfReader(str(file_path))
        has_content = False

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                continue

            has_content = True

            yield Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "converted_from": "pdf",
                    "file_type": "text",
                    "original_pdf": file_path.name,
                    "page_number": page_number,
                    "extraction_method": "pypdf",
                },
            )

        if not has_content:
            raise RuntimeError(
                f"Could not extract any text from PDF using PyPDF: {file_path}"
            )
