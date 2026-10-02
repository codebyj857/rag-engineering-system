"""
FastAPI application entry point for the RAG Engineering System.

This module is responsible for creating and configuring the FastAPI
application, registering API routes, and configuring application-level
middleware.

Business logic must remain in the appropriate application layers rather
than being implemented here.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_engine.api.routes.chat import router as chat_router
from rag_engine.api.routes.evaluate import router as evaluate_router
from rag_engine.api.routes.health import router as health_router
from rag_engine.api.routes.ingest import router as ingest_router
from rag_engine.api.routes.retrieve import router as retrieve_router
from rag_engine.api.routes.sessions import router as sessions_router
from rag_engine.config import get_settings


# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------


settings = get_settings()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Fully configured FastAPI application instance.
    """

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "A production-oriented Retrieval-Augmented Generation "
            "system with multiple retrieval strategies, "
            "conversational memory, verification, and evaluation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # -----------------------------------------------------------------------
    # CORS
    # -----------------------------------------------------------------------

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # API routes
    # -----------------------------------------------------------------------

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(chat_router)
    app.include_router(retrieve_router)
    app.include_router(sessions_router)
    app.include_router(evaluate_router)

    return app


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------


app = create_app()
