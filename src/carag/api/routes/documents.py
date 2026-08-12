import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from celery.result import AsyncResult
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from loguru import logger

from carag.api.dependencies import get_ingestion_engine, get_tenant_id
from carag.api.schemas import (
    BatchDocumentIngestionResponse,
    DeleteDocumentResponse,
    DeleteTenantResponse,
    DocumentIngestionResponse,
    ErrorResponse,
    SuccessResponse,
)
from carag.worker import celery_app
from carag.worker.tasks import ingest_document

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

        if res.result is None and res.state == "PENDING":
            try:
                res.get(timeout=0.1)
            except Exception:
                return {
                    "job_id": job_id,
                    "status": "not_found",
                    "error": "Job does not exist",
                }
        if res.state == "PENDING":
            return {"job_id": job_id, "status": "processing"}
        elif res.state == "SUCCESS":
            return {"job_id": job_id, **res.result}
        else:  # FAILURE
            return {"job_id": job_id, "status": "failed", "error": str(res.info)}
    except Exception as e:
        logger.error(f"Error checking job status {job_id}: {e}")
        return {"job_id": job_id, "status": "error", "error": str(e)}


@router.post(
    "/ingest",
    status_code=202,
    response_model=BatchDocumentIngestionResponse,
    responses={400: {"model": ErrorResponse}},
)
async def ingest_documents(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    tenant_id: str = Depends(get_tenant_id),
) -> BatchDocumentIngestionResponse:
    """
    Accepts PDFs, creates a job per file, returns immediately.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    processed_files = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")

        document_id = str(uuid.uuid4())
        blob = await file.read()
        meta = {
            "document_id": document_id,
            "original_filename": file.filename,
            "ingested_at": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
        }

        # Fire-and-forget Celery task
        task = ingest_document.delay(tenant_id, file.filename, blob, meta)

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
    tenant_id: str = Depends(get_tenant_id),
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


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_document(
    document_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> DeleteDocumentResponse:
    """
    Delete a specific document and all its chunks for the current tenant.
    """
    try:
        ingestion_engine = get_ingestion_engine(tenant_id)

        success, deleted_chunks = ingestion_engine.delete_documents_by_document_id(
            document_id
        )

        if not success or deleted_chunks == 0:
            raise HTTPException(
                status_code=404, detail=f"Document {document_id} not found"
            )

        return DeleteDocumentResponse(
            message=f"Document {document_id} deleted successfully",
            deleted_chunks=deleted_chunks,
            document_id=document_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/tenant/all",
    response_model=DeleteTenantResponse,
    responses={500: {"model": ErrorResponse}},
)
async def delete_all_tenant_documents(
    tenant_id: str = Depends(get_tenant_id),
) -> DeleteTenantResponse:
    """
    Delete ALL documents and chunks for the current tenant.
    Use with caution - this action cannot be undone.
    """
    try:
        ingestion_engine = get_ingestion_engine(tenant_id)

        # Get current count before deletion
        before_count = ingestion_engine.get_tenant_document_count()

        # Delete all tenant data
        success = ingestion_engine.delete_tenant_data()

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete tenant data")

        deleted_chunks = before_count

        return DeleteTenantResponse(
            message=f"All documents for tenant {tenant_id} deleted successfully",
            deleted_chunks=deleted_chunks,
            tenant_id=tenant_id,
        )

    except Exception as e:
        logger.error(f"Error deleting all documents for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
