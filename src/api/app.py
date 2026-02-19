"""
FastAPI application factory for the RAG API.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers.health import router as health_router
from src.api.routers.query import router as query_router
from src.api.routers.chats import router as chats_router
from src.api.routers.ws_chat import router as ws_chat_router
from src.api.service import RAGApiService
from src.config.settings import AppConfig, config


def create_app(app_config: Optional[AppConfig] = None, force_recreate: bool = False) -> FastAPI:
    cfg = app_config or config
    service = RAGApiService(cfg, force_recreate=force_recreate)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        app.state.rag_service = service
        yield

    app = FastAPI(
        title="Fine Arts RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5500",
            "http://localhost:5500",
            "null",  # file:// origin in some browsers
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(chats_router)
    app.include_router(ws_chat_router)
    return app


app = create_app()
