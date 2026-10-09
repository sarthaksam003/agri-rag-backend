import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing_extensions import override
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from markdownify import markdownify as md
from pypdf import PdfReader
from carag.core.config import get_settings
from carag.services.document_ingestion.pdfmdconverter import parse_doc_to_markdown
from carag.services.document_ingestion.utils import (
    fix_markdown_tables,
)

settings = get_settings()


class MineruLoader(BaseLoader):
    """A document loader that converts PDFs to markdown and loads them."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path: str = file_path

    def _get_expected_markdown_path(self, pdf_path: Path, output_base: Path) -> Path:
        """Get the expected path where the markdown file will be saved based on your directory structure."""
        pdf_file_name = pdf_path.stem
        # This matches your directory structure: {output_base}/{pdf_file_name}/vlm/{pdf_file_name}.md
        return output_base / pdf_file_name / "vlm" / f"{pdf_file_name}.md"

    @override
    def lazy_load(self) -> Iterator[Document]:
        """Load a PDF using MinerU when available, with pypdf fallback."""

        file_path = Path(self.file_path)

        if file_path.suffix.lower() != ".pdf":
            raise FileNotFoundError("Only PDF Files supported")

        # First try the existing MinerU pipeline.
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                page_markdowns = parse_doc_to_markdown(
                    [file_path],
                    str(output_dir),
                    server_url,
                )

                if not page_markdowns:
                    raise RuntimeError(
                        f"MinerU returned no page content for {file_path}"
                    )

                for page_data in page_markdowns:
                    page_number = page_data["page_number"]
                    page_content = page_data["content"]

                    page_content = fix_markdown_tables(
                        md(page_content)
                    )

                    if not page_content.strip():
                        continue

                    yield Document(
                        page_content=page_content,
                        metadata={
                            "source": str(file_path),
                            "converted_from": "pdf",
                            "file_type": "markdown",
                            "original_pdf": file_path.name,
                            "page_number": page_number,
                        },
                    )

                return

            except Exception as e:
                # MinerU is GPU-backed and may not be available on the
                # demonstration machine. Fall back to CPU PDF extraction.
                print(
                    f"MinerU conversion unavailable: {e}. "
                    "Falling back to pypdf."
                )

        # CPU fallback.
        reader = PdfReader(str(file_path))

        pages = []
        has_content = False

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            has_content = True

            yield Document(
                page_content=text.strip(),
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
                f"Could not extract any text from PDF: {file_path}"
            )
