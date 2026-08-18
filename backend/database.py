"""
Module 8 - Database connection/session management.

Uses SQLite (built-in sqlite3 — zero-install, single-file DB) to store
users. Seeded on first run with two test accounts per the API contract:
  - admin  / admin123   (role: "admin")
  - manager1 / manager123 (role: "manager")

Exposes get_user(username) for use by auth.py.
"""

import os
import sqlite3
from pathlib import Path

from passlib.context import CryptContext

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "backend" / "users.db"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Schema + seeding
# ---------------------------------------------------------------------------

_SEED_USERS = [
    {"username": "admin",    "password": "admin123",   "role": "admin"},
    {"username": "manager1", "password": "manager123", "role": "manager"},
]


def _get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the users table and seed default accounts if they don't exist.
    Safe to call multiple times (idempotent).
    """
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT    NOT NULL UNIQUE,
                hashed_password TEXT    NOT NULL,
                role            TEXT    NOT NULL CHECK(role IN ('admin', 'manager'))
            )
            """
        )
        conn.commit()

        for user in _SEED_USERS:
            exists = conn.execute(
                "SELECT 1 FROM users WHERE username = ?", (user["username"],)
            ).fetchone()
            if not exists:
                hashed = pwd_context.hash(user["password"])
                conn.execute(
                    "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
                    (user["username"], hashed, user["role"]),
                )
        conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_user(username: str) -> dict | None:
    """
    Return the user row as a dict {id, username, hashed_password, role},
    or None if not found.
    """
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT id, username, hashed_password, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plaintext password against a stored bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)
