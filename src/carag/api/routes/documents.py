import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import shutil
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from loguru import logger
from fastapi.responses import FileResponse
from pathlib import Path
from carag.api.dependencies import (
    get_ingestion_engine,
    get_knowledge_base_tenant_id,
)
from carag.api.schemas import (
    BatchDocumentIngestionResponse,
    DeleteDocumentResponse,
    DeleteTenantResponse,
    DocumentIngestionResponse,
    ErrorResponse,
    SuccessResponse,
)
from carag.worker import celery_app
from carag.core.config import get_settings
from carag.auth.dependencies import require_admin
from carag.auth.models import User

settings = get_settings()
router = APIRouter()
_pool = ThreadPoolExecutor(max_workers=4)


async def _run_in_thread(fn, *args, **kwargs):
    """Run a *blocking* function in a thread-pool and await the result."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_pool, fn, *args, **kwargs)


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    try:
        res = AsyncResult(job_id, app=celery_app)

        if res.state == "PENDING":
            return {
                "job_id": job_id,
                "status": "processing",
            }

        if res.state == "SUCCESS":
            return {
                "job_id": job_id,
                **res.result,
            }

        if res.state == "FAILURE":
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(res.info),
            }

        return {
            "job_id": job_id,
            "status": res.state.lower(),
        }

    except Exception as e:
        logger.error(
            f"Error checking job status {job_id}: {e}"
        )

        return {
            "job_id": job_id,
            "status": "error",
            "error": str(e),
        }


@router.post(
    "/ingest",
    status_code=202,
    response_model=BatchDocumentIngestionResponse,
    responses={400: {"model": ErrorResponse}},
)
async def ingest_documents(
    _: User = Depends(require_admin),
    files: list[UploadFile] = File(...),
    tenant_id: str = Depends(get_knowledge_base_tenant_id),
) -> BatchDocumentIngestionResponse:
    """
    Accepts PDFs, creates a job per file, returns immediately.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    processed_files = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"{file.filename} is not a PDF")

        document_id = str(uuid.uuid4())
        storage_dir = (
            Path(settings.DOCUMENT_STORAGE_DIR).resolve()
            / tenant_id
        )
        storage_dir.mkdir(parents=True, exist_ok=True)

        document_path = (
            storage_dir
            / f"{document_id}.pdf"
        ).resolve()

        try:
            with document_path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    output.write(chunk)
        except Exception:
            # Do not leave a partial PDF if persistence itself fails.
            try:
                document_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                logger.error(
                    f"Could not remove incomplete upload {document_path}: {cleanup_error}"
                )
            raise

        meta = {
            "document_id": document_id,
            "original_filename": file.filename,
            "ingested_at": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
        }

        from carag.worker.tasks import ingest_document
        # print("DEBUG document_path:", document_path)
        # print("DEBUG absolute:", document_path.is_absolute())
        # print("DEBUG exists:", document_path.exists())
        # print("DEBUG documents.py:", __file__)
        try:
            task = ingest_document.delay(
                tenant_id,
                file.filename,
                str(document_path),
                meta,
            )
        except Exception:
            # The API persisted the PDF, but no worker accepted the job.
            # Remove this orphaned upload and let the request fail visibly.
            try:
                document_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                logger.error(
                    f"Could not remove unqueued upload {document_path}: {cleanup_error}"
                )
            raise

        processed_files.append(
            DocumentIngestionResponse(
                document_id=document_id,
                filename=file.filename,
                status="processing",
                job_id=task.id,  # <-- new field
            )
        )

    return BatchDocumentIngestionResponse(
        processed_files=processed_files,
        total_processed=len(processed_files),
        total_failed=0,
    )


@router.get(
    "/",
    response_model=SuccessResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_ingested_documents(
    _: User = Depends(require_admin),
    tenant_id: str = Depends(get_knowledge_base_tenant_id),
) -> SuccessResponse:
    """
    List all ingested documents for the current tenant.
    Uses Qdrant to fetch document metadata.
    """
    try:
        ingestion_engine = get_ingestion_engine(tenant_id)

        # Get collection info to show document count
        docs = ingestion_engine.list_documents()
        # TODO: Implement detailed document listing from Qdrant metadata
        # For now, return basic collection info

        return SuccessResponse(
            message="Documents retrieved successfully",
            data={
                "tenant_id": tenant_id,
                "total_documents": len(docs),
                "documents": docs,
            },
        )

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{document_id}/file",
    response_class=FileResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_document_file(

    document_id: str,
    _: User = Depends(require_admin),
    tenant_id: str = Depends(get_knowledge_base_tenant_id),
):
    """
    Return the persisted PDF for a document belonging to the current tenant.
    """
    try:
        document_path = (
            Path(settings.DOCUMENT_STORAGE_DIR).resolve()
            / tenant_id
            / f"{document_id}.pdf"
        ).resolve()

        if not document_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Document file {document_id} not found",
            )

        return FileResponse(
            path=document_path,
            media_type="application/pdf",
            filename=f"{document_id}.pdf",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error serving document {document_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def delete_document(
    document_id: str,
    _: User = Depends(require_admin),
    tenant_id: str = Depends(get_knowledge_base_tenant_id),
) -> DeleteDocumentResponse:
    """
    Delete a specific document, its chunks, and its persisted PDF
    for the current tenant.
    """
    try:
        document_path = (
            Path(settings.DOCUMENT_STORAGE_DIR).resolve()
            / tenant_id
            / f"{document_id}.pdf"
        ).resolve()
        stored_file_exists = document_path.is_file()

        ingestion_engine = get_ingestion_engine(tenant_id)
        success, deleted_chunks = (
            ingestion_engine.delete_documents_by_document_id(document_id)
        )

        # A failed ingestion can leave a retained source PDF but no Qdrant
        # points. Allow admins to remove that file through the same endpoint.
        if not success and not stored_file_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found",
            )

        if stored_file_exists:
            document_path.unlink()

        return DeleteDocumentResponse(
            message=f"Document {document_id} deleted successfully",
            deleted_chunks=deleted_chunks,
            document_id=document_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error deleting document {document_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@router.delete(
    "/tenant/all",
    response_model=DeleteTenantResponse,
    responses={500: {"model": ErrorResponse}},
)
async def delete_all_tenant_documents(
    _: User = Depends(require_admin),
    tenant_id: str = Depends(get_knowledge_base_tenant_id),
) -> DeleteTenantResponse:
    """
    Delete ALL documents, chunks, and persisted PDF files
    for the current tenant.
    Use with caution - this action cannot be undone.
    """
    try:
        ingestion_engine = get_ingestion_engine(tenant_id)

        # Get current chunk count before deletion
        before_count = (
            ingestion_engine.get_tenant_document_count()
        )

        # Delete all Qdrant tenant data
        success = ingestion_engine.delete_tenant_data()

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete tenant data",
            )

        # Delete all persisted PDFs for this tenant
        storage_dir = (
            Path(settings.DOCUMENT_STORAGE_DIR).resolve()
            / tenant_id
        ).resolve()

        if storage_dir.is_dir():
            for document_path in storage_dir.glob("*.pdf"):
                try:
                    document_path.unlink()
                except OSError as e:
                    logger.error(
                        f"Failed to delete stored document "
                        f"{document_path}: {e}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"Failed to delete stored document "
                            f"{document_path.name}"
                        ),
                    )

            # Remove the tenant directory if it is now empty
            try:
                storage_dir.rmdir()
            except OSError:
                # Directory may contain unrelated files; leave it in place.
                pass

        return DeleteTenantResponse(
            message=(
                f"All documents for tenant {tenant_id} "
                "deleted successfully"
            ),
            deleted_chunks=before_count,
            tenant_id=tenant_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error deleting all documents for tenant "
            f"{tenant_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
