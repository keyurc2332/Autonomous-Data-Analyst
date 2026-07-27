"""Aggregates all route modules. New routers get registered here only."""
from fastapi import APIRouter

from app.api.routes import analysis, chat, datasets, health, projects

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(projects.router)
api_router.include_router(datasets.router)
api_router.include_router(analysis.router)
api_router.include_router(chat.router)

