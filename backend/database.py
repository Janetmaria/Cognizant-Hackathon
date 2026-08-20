"""
Module 8 - SQLite database configuration, ORM models, and startup seeding.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, TypedDict

import bcrypt
from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "app.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="manager", nullable=False)
    assigned_store = Column(String(10), nullable=True)  # e.g., '4' or None for admin
    full_name = Column(String(100), nullable=True)
    email = Column(String(120), nullable=True)
    disabled = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)


class SeedUser(TypedDict):
    username: str
    password: str
    role: str
    assigned_store: str
    full_name: str
    email: str


DEFAULT_SEED_USERS: List[SeedUser] = [
    {
        "username": "admin",
        "password": "admin123",
        "role": "admin",
        "assigned_store": "",
        "full_name": "System Administrator",
        "email": "admin@retailanalytics.internal",
    },
    {
        "username": "manager",
        "password": "manager123",
        "role": "manager",
        "assigned_store": "1",
        "full_name": "Store 1 Operations Manager",
        "email": "manager@retailanalytics.internal",
    },
    {
        "username": "manager1",
        "password": "manager123",
        "role": "manager",
        "assigned_store": "4",
        "full_name": "Store 4 Inventory Manager",
        "email": "manager1@retailanalytics.internal",
    },
]


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def init_db() -> None:
    """Create tables and seed default accounts when missing."""
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        for seed in DEFAULT_SEED_USERS:
            existing = (
                db.query(User)
                .filter(User.username == seed["username"])
                .first()
            )
            if existing is not None:
                if seed["assigned_store"] and not existing.assigned_store:
                    existing.assigned_store = seed["assigned_store"]
                    db.commit()
                continue

            db.add(
                User(
                    username=seed["username"],
                    hashed_password=_hash_password(seed["password"]),
                    role=seed["role"],
                    assigned_store=seed["assigned_store"] or None,
                    full_name=seed["full_name"],
                    email=seed["email"],
                    disabled=False,
                )
            )
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
    finally:
        db.close()
