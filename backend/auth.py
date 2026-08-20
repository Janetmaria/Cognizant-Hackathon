"""
Module 8 - Authentication & Role-Based Access Control (RBAC).

Owner: Module 8.
This module handles:
1. Secure password hashing and verification using bcrypt.
2. JWT (JSON Web Token) creation, signing, and verification (HS256).
3. Server-side role enforcement (admin vs. manager) using FastAPI dependency injection.
4. User login endpoints supporting both OAuth2 form data (/auth/token) and JSON payloads (/auth/login).
5. User profile inspection (/auth/me).

Persistent users are stored in SQLite via backend.database (data/app.db).
Seed accounts are created on application startup by database.init_db().
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import backend.database as database

# Try importing jwt (supports PyJWT and python-jose)
try:
    import jwt
    from jwt.exceptions import ExpiredSignatureError, PyJWTError
except ImportError:
    try:
        from jose import ExpiredSignatureError, JWTError as PyJWTError, jwt
    except ImportError:
        jwt = None
        PyJWTError = Exception
        ExpiredSignatureError = Exception


# =============================================================================
# 1. SECURITY CONFIGURATION
# =============================================================================

SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "retail-forecasting-platform-secret-key-2026-walmart-hackathon-module8-secure-jwt",
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
ACCESS_TOKEN_EXPIRE_SECONDS = ACCESS_TOKEN_EXPIRE_MINUTES * 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


# =============================================================================
# 2. PASSWORD HASHING UTILITIES
# =============================================================================

def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt with a generated salt."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


# =============================================================================
# 3. USER LOOKUP & AUTHENTICATION (SQLite-backed)
# =============================================================================

def _user_to_dict(user: database.User) -> Dict[str, Any]:
    return {
        "username": user.username,
        "hashed_password": user.hashed_password,
        "role": user.role,
        "assigned_store": getattr(user, "assigned_store", None),
        "full_name": user.full_name,
        "email": user.email,
        "disabled": user.disabled,
    }


def get_user_by_username(username: str, db: Session) -> Optional[Dict[str, Any]]:
    """Look up a user by username in the persistent SQLite store."""
    normalized = username.strip().lower()
    user = db.query(database.User).filter(database.User.username == normalized).first()
    if user is None:
        return None
    return _user_to_dict(user)


def authenticate_user(
    username: str,
    password: str,
    db: Session,
) -> Optional[Dict[str, Any]]:
    """Validate username and password. Returns user dict if valid, else None."""
    user = get_user_by_username(username, db)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    if user.get("disabled", False):
        return None
    return user


# =============================================================================
# 4. JWT TOKEN UTILITIES
# =============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Encode claims into a signed HS256 JWT access token."""
    if jwt is None:
        raise RuntimeError("JWT library (pyjwt or python-jose) is required for token creation.")

    to_encode = data.copy()
    now_utc = datetime.now(timezone.utc)

    if expires_delta:
        expire = now_utc + expires_delta
    else:
        expire = now_utc + timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS)

    to_encode.update({
        "exp": expire,
        "iat": now_utc,
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token's signature and expiration."""
    if jwt is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": True, "message": "JWT library not installed on server.", "status_code": 500},
        )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": True, "message": "Access token has expired. Please log in again.", "status_code": 401},
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (PyJWTError, Exception):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": True, "message": "Invalid authentication credentials or malformed token.", "status_code": 401},
            headers={"WWW-Authenticate": "Bearer"},
        )


# =============================================================================
# 5. PYDANTIC SCHEMAS
# =============================================================================

class Token(BaseModel):
    access_token: str = Field(..., description="Signed HS256 JWT access token")
    token_type: str = Field("bearer", description="Token type prefix (bearer)")
    role: str = Field(..., description="User role ('admin' or 'manager')")
    assigned_store: Optional[str] = Field(None, description="Assigned store identifier for store manager (e.g. '4')")
    expires_in: int = Field(ACCESS_TOKEN_EXPIRE_SECONDS, description="Token lifetime in seconds (8 hours)")


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None
    assigned_store: Optional[str] = None


class UserAuth(BaseModel):
    username: str = Field(..., description="User account username (e.g. 'admin', 'manager1')")
    password: str = Field(..., description="User account password")


class UserOut(BaseModel):
    username: str
    role: str
    assigned_store: Optional[str] = None
    full_name: Optional[str] = None
    email: Optional[str] = None
    disabled: bool = False


# =============================================================================
# 6. FASTAPI DEPENDENCIES & SERVER-SIDE ROLE GUARDS
# =============================================================================

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(database.get_db),
) -> Dict[str, Any]:
    """Extract Bearer token, validate it, and return the authenticated user."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": True, "message": "Missing or invalid authorization token.", "status_code": 401},
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    username: Optional[str] = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": True, "message": "Token payload missing subject identifier.", "status_code": 401},
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_username(username, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": True, "message": f"User '{username}' no longer exists in system.", "status_code": 401},
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.get("disabled", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": True, "message": "User account has been disabled.", "status_code": 403},
        )

    return user


def require_role(allowed_roles: List[str]) -> Callable:
    """RBAC dependency factory enforcing one of the allowed roles."""

    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user),
    ) -> Dict[str, Any]:
        user_role = current_user.get("role", "").lower()
        normalized_allowed = [role.lower() for role in allowed_roles]

        if user_role not in normalized_allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": True,
                    "message": (
                        f"Access forbidden: User role '{user_role}' is not authorized. "
                        f"Required role(s): {', '.join(allowed_roles)}"
                    ),
                    "status_code": 403,
                },
            )
        return current_user

    return role_checker


def enforce_store_access(store_id: str, current_user: Dict[str, Any]) -> None:
    """
    Enforces that store managers can only view/query their assigned store.
    Admin users can access all stores without restriction.
    """
    user_role = current_user.get("role", "").lower()
    if user_role == "admin":
        return  # Admins have unrestricted chain-wide access across all stores

    assigned_store = current_user.get("assigned_store")
    # Only restrict if an explicit assigned_store is set on the manager account
    if assigned_store and str(assigned_store).strip():
        if str(assigned_store).strip() != str(store_id).strip():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": True,
                    "message": f"Access forbidden: You are assigned to Store {assigned_store} and cannot access data for Store {store_id}.",
                    "status_code": 403,
                },
            )


require_admin = require_role(["admin"])
require_manager_or_admin = require_role(["manager", "admin"])


# =============================================================================
# 7. ROUTER DEFINITION & ENDPOINTS
# =============================================================================

auth_router = APIRouter(prefix="/auth", tags=["Authentication & RBAC"])
router = auth_router


@auth_router.post(
    "/login",
    response_model=Token,
    summary="User Login (JSON Body)",
    description="Authenticates with username & password in JSON body and returns signed JWT access token.",
)
def login_json(
    credentials: UserAuth,
    db: Session = Depends(database.get_db),
) -> Token:
    user = authenticate_user(credentials.username, credentials.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": True, "message": "Invalid username or password", "status_code": 401},
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={
        "sub": user["username"],
        "role": user["role"],
        "assigned_store": user.get("assigned_store"),
    })
    return Token(
        access_token=access_token,
        token_type="bearer",
        role=user["role"],
        assigned_store=user.get("assigned_store"),
        expires_in=ACCESS_TOKEN_EXPIRE_SECONDS,
    )


@auth_router.post(
    "/token",
    response_model=Token,
    summary="OAuth2 Form Login (Swagger UI Compatible)",
    description="Authenticates with form data (x-www-form-urlencoded) for Swagger UI interactive login.",
)
def login_oauth2_form(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(database.get_db),
) -> Token:
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": True, "message": "Invalid username or password", "status_code": 401},
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={
        "sub": user["username"],
        "role": user["role"],
        "assigned_store": user.get("assigned_store"),
    })
    return Token(
        access_token=access_token,
        token_type="bearer",
        role=user["role"],
        assigned_store=user.get("assigned_store"),
        expires_in=ACCESS_TOKEN_EXPIRE_SECONDS,
    )


@auth_router.get(
    "/me",
    response_model=UserOut,
    summary="Get Authenticated User Profile",
    description="Returns the profile information and assigned role of the currently authenticated user.",
)
def get_me(current_user: Dict[str, Any] = Depends(get_current_user)) -> UserOut:
    return UserOut(
        username=current_user["username"],
        role=current_user["role"],
        assigned_store=current_user.get("assigned_store"),
        full_name=current_user.get("full_name"),
        email=current_user.get("email"),
        disabled=current_user.get("disabled", False),
    )
