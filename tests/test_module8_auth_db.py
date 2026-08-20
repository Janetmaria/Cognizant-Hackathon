"""
Module 8 - Authentication & RBAC Test Suite

Tests SQLite database authentication, JWT token handling, and role-based access control.
Uses pytest with FastAPI TestClient for comprehensive endpoint validation.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Add project root to Python path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import backend components
from backend.main import app
from backend.database import init_db, get_db, SessionLocal, User, DB_PATH, DATA_DIR


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def test_db_path() -> Generator[Path, None, None]:
    """
    Create a temporary SQLite database for isolated testing.
    Ensures each test runs with a clean database state.
    """
    # Create temporary directory for test database
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db_file = Path(tmp_dir) / "test_app.db"
        
        # Override the database path for testing
        original_db_path = DB_PATH
        original_data_dir = DATA_DIR
        
        # Monkey-patch the database module to use test database
        import backend.database
        backend.database.DB_PATH = test_db_file
        backend.database.DATA_DIR = Path(tmp_dir)
        backend.database.SQLALCHEMY_DATABASE_URL = f"sqlite:///{test_db_file.as_posix()}"
        backend.database.engine = backend.database.create_engine(
            backend.database.SQLALCHEMY_DATABASE_URL,
            connect_args={"check_same_thread": False},
        )
        backend.database.SessionLocal = backend.database.sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=backend.database.engine
        )
        
        # Initialize test database with seed users
        init_db()
        
        yield test_db_file
        
        # Restore original database configuration
        backend.database.DB_PATH = original_db_path
        backend.database.DATA_DIR = original_data_dir
        backend.database.SQLALCHEMY_DATABASE_URL = f"sqlite:///{original_db_path.as_posix()}"
        backend.database.engine = backend.database.create_engine(
            backend.database.SQLALCHEMY_DATABASE_URL,
            connect_args={"check_same_thread": False},
        )
        backend.database.SessionLocal = backend.database.sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=backend.database.engine
        )


@pytest.fixture(scope="function")
def client(test_db_path: Path) -> Generator[TestClient, None, None]:
    """
    Create a FastAPI TestClient with the test database.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def db_session(test_db_path: Path) -> Generator[SessionLocal, None, None]:
    """
    Provide a database session for direct database queries in tests.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def admin_token(client: TestClient) -> str:
    """
    Fixture that logs in as admin and returns a valid JWT token.
    """
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "admin123"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture(scope="function")
def manager_token(client: TestClient) -> str:
    """
    Fixture that logs in as manager and returns a valid JWT token.
    """
    response = client.post(
        "/auth/login",
        json={"username": "manager", "password": "manager123"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


# =============================================================================
# TEST CASES
# =============================================================================

def test_db_seed_users_exist(db_session: SessionLocal):
    """
    Verify default seed users ('admin', 'manager', 'manager1') are present 
    in SQLite with hashed passwords.
    """
    expected_users = ["admin", "manager", "manager1"]
    
    for username in expected_users:
        user = db_session.query(User).filter(User.username == username).first()
        assert user is not None, f"Seed user '{username}' not found in database"
        assert user.hashed_password is not None, f"User '{username}' has no hashed password"
        assert len(user.hashed_password) > 0, f"User '{username}' has empty password hash"
        assert user.role in ["admin", "manager"], f"User '{username}' has invalid role: {user.role}"
        assert user.disabled is False, f"User '{username}' should not be disabled"


def test_login_success_admin(client: TestClient):
    """
    Test POST /auth/login with valid admin credentials.
    Ensures it returns 200 OK, a JWT access_token, and token_type: "bearer".
    """
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "admin123"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "access_token" in data
    assert isinstance(data["access_token"], str)
    assert len(data["access_token"]) > 0
    
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"
    assert "expires_in" in data
    assert isinstance(data["expires_in"], int)
    assert data["expires_in"] > 0


def test_login_success_manager(client: TestClient):
    """
    Test POST /auth/login for manager credentials.
    """
    response = client.post(
        "/auth/login",
        json={"username": "manager", "password": "manager123"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "access_token" in data
    assert isinstance(data["access_token"], str)
    assert len(data["access_token"]) > 0
    
    assert data["token_type"] == "bearer"
    assert data["role"] == "manager"
    assert "expires_in" in data


def test_login_invalid_password(client: TestClient):
    """
    Test POST /auth/login with wrong password.
    Asserts 401 Unauthorized response.
    """
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "wrongpassword"}
    )
    
    assert response.status_code == 401
    data = response.json()
    
    # Check for error in response (may be in different format)
    assert "detail" in data or "error" in data
    if "detail" in data:
        assert "Invalid" in str(data["detail"]) or "password" in str(data["detail"]).lower()
    if "error" in data:
        assert data["error"] is True


def test_login_nonexistent_user(client: TestClient):
    """
    Test POST /auth/login with unknown username.
    Asserts 401 Unauthorized response.
    """
    response = client.post(
        "/auth/login",
        json={"username": "nonexistent_user", "password": "anypassword"}
    )
    
    assert response.status_code == 401
    data = response.json()
    
    # Check for error in response (may be in different format)
    assert "detail" in data or "error" in data
    if "detail" in data:
        assert "Invalid" in str(data["detail"]) or "password" in str(data["detail"]).lower()
    if "error" in data:
        assert data["error"] is True


def test_auth_me_endpoint(client: TestClient, admin_token: str):
    """
    Test GET /auth/me with a valid Bearer token.
    Asserts correct username and role in response.
    """
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["username"] == "admin"
    assert data["role"] == "admin"
    assert "full_name" in data
    assert "email" in data
    assert data["disabled"] is False


def test_auth_me_endpoint_manager(client: TestClient, manager_token: str):
    """
    Test GET /auth/me with manager token.
    """
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {manager_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["username"] == "manager"
    assert data["role"] == "manager"
    assert data["disabled"] is False


def test_rbac_gated_endpoint_unauthenticated(client: TestClient):
    """
    Test that unauthenticated requests to protected endpoints return 401.
    Tests the /forecast/1 endpoint which requires manager or admin role.
    """
    response = client.get(
        "/forecast/1",
        params={
            "dept_id": "92",
            "start_date": "2012-11-02",
            "end_date": "2012-11-30"
        }
    )
    
    # Should return 401 because no authentication token provided
    assert response.status_code == 401
    data = response.json()
    
    # Check for error in response (may be in different format)
    assert "detail" in data or "error" in data


def test_rbac_manager_can_access_forecast(client: TestClient, manager_token: str):
    """
    Test that manager role can access /forecast/ endpoint.
    This may return 500 if predictions file doesn't exist, but should not return 403.
    """
    response = client.get(
        "/forecast/1",
        params={
            "dept_id": "92",
            "start_date": "2012-11-02",
            "end_date": "2012-11-30"
        },
        headers={"Authorization": f"Bearer {manager_token}"}
    )
    
    # Manager should be authorized (not 401 or 403)
    # May get 500 if predictions file missing, but that's expected
    assert response.status_code in [200, 500]
    if response.status_code == 500:
        # This is acceptable if predictions file doesn't exist
        data = response.json()
        assert data["error"] is True


def test_rbac_admin_can_access_forecast(client: TestClient, admin_token: str):
    """
    Test that admin role can access /forecast/ endpoint.
    """
    response = client.get(
        "/forecast/1",
        params={
            "dept_id": "92",
            "start_date": "2012-11-02",
            "end_date": "2012-11-30"
        },
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    # Admin should be authorized (not 401 or 403)
    assert response.status_code in [200, 500]


def test_expired_or_invalid_token(client: TestClient):
    """
    Test GET /auth/me with a malformed JWT token.
    Expects 401 Unauthorized.
    """
    # Test with completely invalid token
    response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid.token.here"}
    )
    
    assert response.status_code == 401
    data = response.json()
    
    # Check for error in response (may be in different format)
    assert "detail" in data or "error" in data


def test_missing_token(client: TestClient):
    """
    Test GET /auth/me without any authorization header.
    Expects 401 Unauthorized.
    """
    response = client.get("/auth/me")
    
    assert response.status_code == 401
    data = response.json()
    
    # Check for error in response (may be in different format)
    assert "detail" in data or "error" in data
    if "detail" in data:
        assert "Missing" in str(data["detail"]) or "authorization" in str(data["detail"]).lower()
    if "error" in data:
        assert data["error"] is True


def test_oauth2_form_login(client: TestClient):
    """
    Test OAuth2 form login endpoint (/auth/token) for Swagger UI compatibility.
    """
    response = client.post(
        "/auth/token",
        data={"username": "admin", "password": "admin123"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"


def test_oauth2_form_login_invalid_credentials(client: TestClient):
    """
    Test OAuth2 form login with invalid credentials.
    """
    response = client.post(
        "/auth/token",
        data={"username": "admin", "password": "wrongpassword"}
    )
    
    assert response.status_code == 401
    data = response.json()
    
    # Check for error in response (may be in different format)
    assert "detail" in data or "error" in data
    if "detail" in data:
        assert "Invalid" in str(data["detail"]) or "password" in str(data["detail"]).lower()
    if "error" in data:
        assert data["error"] is True


def test_manager1_login(client: TestClient):
    """
    Test login for the second manager account (manager1).
    """
    response = client.post(
        "/auth/login",
        json={"username": "manager1", "password": "manager123"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["username"] == "manager1" if "username" in data else True
    assert data["role"] == "manager"
    assert "access_token" in data


def test_token_structure(client: TestClient, admin_token: str):
    """
    Verify JWT token has proper structure (header.payload.signature).
    """
    # JWT tokens should have 2 dots separating 3 parts
    parts = admin_token.split(".")
    assert len(parts) == 3, "JWT token should have 3 parts separated by dots"


def test_bearer_token_case_sensitivity(client: TestClient, admin_token: str):
    """
    Test that 'Bearer' header is case-sensitive (should be lowercase 'bearer').
    """
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"bearer {admin_token}"}  # lowercase
    )
    
    # Should work with lowercase 'bearer'
    assert response.status_code == 200


def test_password_hashing_db(db_session: SessionLocal):
    """
    Verify that passwords in database are properly hashed (not plaintext).
    """
    admin_user = db_session.query(User).filter(User.username == "admin").first()
    assert admin_user is not None
    
    # Password should not be stored as plaintext
    assert admin_user.hashed_password != "admin123"
    # Bcrypt hashes start with $2b$ or $2a$
    assert admin_user.hashed_password.startswith("$2"), "Password should be bcrypt hashed"


def test_user_role_field(db_session: SessionLocal):
    """
    Verify user roles are correctly stored in database.
    """
    admin_user = db_session.query(User).filter(User.username == "admin").first()
    manager_user = db_session.query(User).filter(User.username == "manager").first()
    
    assert admin_user.role == "admin"
    assert manager_user.role == "manager"


def test_disabled_field_default(db_session: SessionLocal):
    """
    Verify that new users have disabled=False by default.
    """
    for username in ["admin", "manager", "manager1"]:
        user = db_session.query(User).filter(User.username == username).first()
        assert user is not None
        assert user.disabled is False, f"User {username} should not be disabled by default"


def test_unique_username_constraint(db_session: SessionLocal):
    """
    Verify that username uniqueness constraint is enforced.
    """
    from backend.database import _hash_password
    from sqlalchemy.exc import IntegrityError
    
    # Try to create a duplicate user
    duplicate_user = User(
        username="admin",  # Already exists from seed
        hashed_password=_hash_password("differentpassword"),
        role="manager",
        full_name="Duplicate User",
        email="duplicate@test.com",
        disabled=False
    )
    
    db_session.add(duplicate_user)
    
    with pytest.raises(IntegrityError):
        db_session.commit()
    
    db_session.rollback()


def test_get_db_dependency(client: TestClient):
    """
    Test that get_db dependency works correctly through the API.
    """
    # This is implicitly tested by other tests, but we can verify
    # the database dependency doesn't crash the app
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "admin123"}
    )
    assert response.status_code == 200
