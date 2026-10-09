from fastapi import APIRouter, Depends, HTTPException, status

from carag.auth.dependencies import require_admin
from carag.auth.models import User
from carag.api.schemas import RagConfigurationUpdate, SuccessResponse
from carag.core.rag_config import RagConfiguration

router = APIRouter()


async def _get_rag_configuration() -> RagConfiguration:
    configuration = await RagConfiguration.find_all().first_or_none()

    if configuration is None:
        configuration = RagConfiguration(
            rag_mode="simple",
            max_queries=2,
        )
        await configuration.insert()

    return configuration


@router.get("/rag-config", response_model=SuccessResponse)
async def get_rag_configuration(
    _admin: User = Depends(require_admin),
) -> SuccessResponse:
    configuration = await _get_rag_configuration()

    return SuccessResponse(
        message="RAG configuration retrieved successfully",
        data={
            "rag_mode": configuration.rag_mode,
            "max_queries": configuration.max_queries,
        },
    )


@router.put("/rag-config", response_model=SuccessResponse)
async def update_rag_configuration(
    request: RagConfigurationUpdate,
    _admin: User = Depends(require_admin),
) -> SuccessResponse:
    configuration = await _get_rag_configuration()

    configuration.rag_mode = request.rag_mode
    configuration.max_queries = request.max_queries

    await configuration.save()

    return SuccessResponse(
        message="RAG configuration updated successfully",
        data={
            "rag_mode": configuration.rag_mode,
            "max_queries": configuration.max_queries,
        },
    )