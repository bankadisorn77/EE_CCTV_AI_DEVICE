"""Validated request schemas for the edge-device HTTP API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AIControlRequest(BaseModel):
    action: str = Field(..., min_length=1)
    roi: Optional[List[Dict[str, Any]]] = None
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
