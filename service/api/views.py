import random
from typing import List

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        200: {
            "description": "Successful response",
            "model": RecoResponse,
            "example": {
                "value": {"user_id": 123, "items": [1, 2, 3]},
                "summary": "Example response for user_id=123",
            },
        },
        400: {
            "description": "Invalid input",
            "model": str,
        },
        404: {
            "description": "User or model not found",
            "model": str,
            "example": {"value": "Invalid model_name", "summary": "Example response for invalid model_name"},
        },
        500: {
            "description": "Internal server error",
            "model": str,
        },
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if model_name == "random":
        random_recs = random.sample(range(1000000), 10)
        reco = random_recs
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
