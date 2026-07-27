"""Shared FastAPI dependencies.

`get_current_user` is the seam where real authentication lands in a later
phase. Until then it resolves a single development user, created on first
use. Everything downstream already takes a User, so swapping in JWT auth
changes this function and nothing else.
"""
from __future__ import annotations

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import User
from app.db.session import get_db

DEV_USER_EMAIL = "dev@localhost"


async def get_current_user(db: AsyncSession = Depends(get_db)) -> User:
    if settings.ENVIRONMENT == "production":
        # Fail loudly rather than silently granting access to a shared account.
        raise RuntimeError(
            "Authentication is not implemented. Refusing to serve production traffic."
        )

    result = await db.execute(select(User).where(User.email == DEV_USER_EMAIL))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(email=DEV_USER_EMAIL, display_name="Development User")
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user
