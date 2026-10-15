"""Shared FastAPI dependencies: database session and optional API-key authorization."""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db


def require_api_key(x_api_key: str | None = Header(default=None), settings: Settings = Depends(get_settings)) -> None:
    """When API_KEY is configured every request must carry a matching X-API-Key header."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


DB = Depends(get_db)
Auth = Depends(require_api_key)

__all__ = ["DB", "Auth", "Session", "get_db", "require_api_key"]
