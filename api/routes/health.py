from fastapi import APIRouter
from typing import Dict

router = APIRouter(prefix="/api/health", tags=["Health"])


@router.get("/")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """Readiness check endpoint"""
    return {"status": "ready"}

