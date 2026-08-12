import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing_extensions import override
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from markdownify import markdownify as md

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
        """Lazy load documents by converting PDF to markdown first if needed."""
        file_path = Path(self.file_path)

        # Check if file is PDF
        if file_path.suffix.lower() == ".pdf":
            # Use a temporary directory for the conversion process
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Convert PDF to markdown - this will create the directory structure in temp_dir
                    parse_doc_to_markdown(
                        [file_path],
                        temp_dir,  # This becomes the base output directory
                        settings.PDFMD_API_BASE,
                    )

                    # Find the markdown file based on the expected structure
                    expected_md_path = self._get_expected_markdown_path(
                        file_path, Path(temp_dir)
                    )

                    if not expected_md_path.exists():
                        # Try alternative path search if the structure is different
                        md_files = list(Path(temp_dir).rglob("*.md"))
                        if md_files:
                            expected_md_path = md_files[0]
                        else:
                            raise FileNotFoundError(
                                f"Markdown file not found at {expected_md_path}"
                            )

                    # Read the entire markdown content
                    with open(expected_md_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    fixed_md_text = fix_markdown_tables(md(content))

                    # Yield the entire document as one chunk
                    yield Document(
                        page_content=fixed_md_text,
                        metadata={
                            "source": str(file_path),
                            "converted_from": "pdf",
                            "file_type": "markdown",
                            "original_pdf": file_path.name,
                        },
                    )

                except Exception as e:
                    raise RuntimeError(f"Failed to convert PDF to markdown: {e}") from e

        else:
            raise FileNotFoundError("Only PDF Files supported")
