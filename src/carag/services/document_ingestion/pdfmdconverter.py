import os
from pathlib import Path

from loguru import logger
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.cli.common import prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode

from carag.services.document_ingestion.title_maker import Titler


def parse_doc_to_markdown(
    path_list: list[Path],
    output_dir: str,
    server_url: str,
    start_page_id: int = 0,
    end_page_id: int | None = None,
):
    """
    Convert PDFs to markdown using vlm-http-client backend

    Parameters:
    - path_list: List of PDF file paths
    - output_dir: Output directory for markdown files
    - server_url: VLM server URL (e.g., "http://127.0.0.1:30000")
    - start_page_id: Start page (0-based)
    - end_page_id: End page (None = all pages)
    """
    titler = Titler()
    for path in path_list:
        pdf_bytes = read_fn(path)
        pdf_file_name = Path(path).stem

        # Prepare output directories
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, "vlm")
        image_writer, md_writer = (
            FileBasedDataWriter(local_image_dir),
            FileBasedDataWriter(local_md_dir),
        )

        # Analyze document
        middle_json, _ = vlm_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend="http-client",
            server_url=server_url,
        )

        # Generate markdown
        pdf_info_tmp = middle_json["pdf_info"]

        logger.debug(f"Applying titler to: {pdf_file_name}")
        pdf_info = titler.llm_aided_title(pdf_info_tmp)

        image_dir = os.path.basename(local_image_dir)
        md_content = vlm_union_make(pdf_info, MakeMode.MM_MD, image_dir)

        # Write markdown file
        md_writer.write_string(f"{pdf_file_name}.md", md_content)

        logger.info(f"Markdown generated: {local_md_dir}/{pdf_file_name}.md")


if __name__ == "__main__":
    from carag.core.config import get_settings

    settings = get_settings()
    # Configuration
    pdf_files_dir = settings.PDFMD_DATA_RAW
    output_dir = settings.PDFMD_DATA_PROCESSED
    server_url = settings.PDFMD_API_BASE

    # Get PDF files
    pdf_paths = [p for p in Path(pdf_files_dir).glob("*.pdf")]

    # Convert to markdown
    parse_doc_to_markdown(pdf_paths, output_dir, server_url)
