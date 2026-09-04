"""User request and response schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.user import UserRole


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)
    preferred_locale: Literal["en", "fr"] = "en"


class UserLogin(BaseModel):
    username: str
    password: str


class UserUpdate(BaseModel):
    username: str | None = Field(
        default=None, min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$"
    )
    preferred_locale: Literal["en", "fr"] | None = None


class UserOut(BaseModel):
    id: int
    username: str
    role: UserRole
    preferred_locale: str
    is_active: bool

    model_config = ConfigDict(from_attributes=True)
