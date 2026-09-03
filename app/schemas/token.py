"""Authentication token request and response schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class APITokenCreate(BaseModel):
    name: str = Field(default="default", min_length=1, max_length=80)
    expires_in_days: int | None = Field(default=None, ge=1, le=365)


class APITokenOut(BaseModel):
    api_token: str
    name: str
    expires_at: datetime | None
