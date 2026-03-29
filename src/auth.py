"""
src/auth.py
===========
Secure user authentication utilities.

Passwords are stored as  salt:sha256(salt+password)  — never plaintext.
Legacy plaintext passwords (from the original version) are accepted once
and transparently upgraded to hashed form on next login.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Optional

from .config import DB_PATH


# ── Helpers ────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Return 'salt:digest' for a given plaintext password."""
    salt   = secrets.token_hex(16)
    digest = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{digest}"


def verify_password(password: str, stored: str) -> bool:
    """
    Verify a plaintext password against a stored hash.
    Falls back to direct comparison for legacy plaintext entries.
    """
    if ":" in stored:
        salt, digest = stored.split(":", 1)
        return hashlib.sha256((salt + password).encode()).hexdigest() == digest
    return password == stored    # legacy plaintext


def _load_db() -> dict:
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        return {"users": []}
    with open(DB_PATH) as f:
        return json.load(f)


def _save_db(data: dict) -> None:
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── Public API ─────────────────────────────────────────────────────────────────

def create_user(name: str, email: str, age: int, sex: str,
                password: str) -> str:
    """
    Register a new user.

    Returns
    -------
    'ok'        – success
    'duplicate' – email already registered
    'error'     – unexpected failure
    """
    try:
        db = _load_db()
        if any(u["email"] == email for u in db["users"]):
            return "duplicate"
        db["users"].append({
            "name"    : name,
            "email"   : email,
            "age"     : int(age),
            "sex"     : sex,
            "password": hash_password(password),
        })
        _save_db(db)
        return "ok"
    except Exception:
        return "error"


def authenticate(email: str, password: str) -> Optional[dict]:
    """
    Authenticate by email + password.

    Returns the user dict on success, None on failure.
    Transparently upgrades legacy plaintext passwords to hashed form.
    """
    db = _load_db()
    for user in db["users"]:
        if user["email"] != email:
            continue
        if verify_password(password, user["password"]):
            # Upgrade legacy plaintext password
            if ":" not in user["password"]:
                user["password"] = hash_password(password)
                _save_db(db)
            return user
    return None
