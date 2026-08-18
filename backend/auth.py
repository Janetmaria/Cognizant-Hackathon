"""
Module 8 - Authentication: JWT issuing/verification, password hashing.

Exposes:
  POST /auth/login           — issues JWT for valid credentials
  get_current_user()         — FastAPI dependency; raises 401 on bad token
  require_admin()            — FastAPI dependency; raises 403 if role != admin

Token spec (per api_contract.md v2):
  - Algorithm : HS256
  - Expiry    : 28800 seconds (8 hours), no refresh endpoint in v2 scope
  - Role claim: embedded in payload so every request carries its own role
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from backend.database import get_user, init_db, verify_password

load_dotenv()

# ---------------------------------------------------------------------------
# Config (read from env; sensible defaults for local dev only)
# ---------------------------------------------------------------------------

SECRET_KEY: str = os.getenv("SECRET_KEY", "CHANGE_ME_before_production_use")
ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_SECONDS: int = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_SECONDS", "28800")
)

# ---------------------------------------------------------------------------
# FastAPI router + security scheme
# ---------------------------------------------------------------------------

router = APIRouter()
bearer_scheme = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    role: str
    expires_in: int


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def _create_access_token(username: str, role: str) -> str:
    """Create a signed HS256 JWT containing sub (username) and role."""
    expire = datetime.now(timezone.utc) + timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict:
    """Decode and validate a JWT; raises JWTError on any failure."""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> dict:
    """
    Dependency that validates the Bearer token on every protected endpoint.
    Returns the decoded payload dict {sub, role} on success.
    Raises HTTP 401 if the token is missing, expired, or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": True, "message": "Not authenticated", "status_code": 401},
        )
    try:
        payload = _decode_token(credentials.credentials)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": True, "message": "Invalid or expired token", "status_code": 401},
        )
    return payload


def require_admin(
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """
    Dependency that further restricts an endpoint to admin role only.
    Raises HTTP 403 if the authenticated user's role is not 'admin'.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": True, "message": "Admin access required", "status_code": 403},
        )
    return current_user


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest):
    """
    POST /auth/login

    Validates credentials against the seeded user DB, issues a signed JWT.
    No auth required on this endpoint itself.

    Returns 401 with the standard error envelope on bad credentials.
    """
    # Ensure DB is initialised (idempotent — safe to call every request)
    init_db()

    user = get_user(body.username)

    if user is None or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "message": "Invalid username or password",
                "status_code": 401,
            },
        )

    token = _create_access_token(username=user["username"], role=user["role"])

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        expires_in=ACCESS_TOKEN_EXPIRE_SECONDS,
    )
