"""Authorization lookup.

Stands in for the LDAP/SCIM directory a real deployment would query. The
structure is role-based so the production swap is a change of backend, not a
change of interface: authorized() is the only function the rest of the system
calls.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

from .schema import DataType

DB_PATH = Path(__file__).resolve().parent.parent / "directory.db"

# role -> data types the role is entitled to access
ROLE_ENTITLEMENTS = {
    "support": ["customer", "internal_doc", "public"],
    "hr": ["employee", "internal_doc", "public"],
    "finance": ["financial", "internal_doc", "public"],
    "engineering": ["internal_doc", "public"],
    "security": ["customer", "employee", "financial", "credential",
                 "internal_doc", "public"],
}

SEED_EMPLOYEES = [
    ("e001", "Dana Okoro", "support"),
    ("e002", "Marcus Webb", "hr"),
    ("e003", "Priya Raman", "finance"),
    ("e004", "Sam Little", "engineering"),
    ("e005", "Alex Diaz", "security"),
]


def init_db(path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS employees "
                 "(id TEXT PRIMARY KEY, name TEXT, role TEXT)")
    conn.executemany("INSERT OR REPLACE INTO employees VALUES (?,?,?)",
                     SEED_EMPLOYEES)
    conn.commit()
    conn.close()


def _row(requester_id: str, path: Path = DB_PATH):
    if not path.exists():
        init_db(path)
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT id, name, role FROM employees WHERE id = ?",
                       (requester_id,)).fetchone()
    conn.close()
    return row


def lookup(requester_id: str):
    row = _row(requester_id)
    if row is None:
        return None
    return {"id": row[0], "name": row[1], "role": row[2]}


def authorized(requester_id: str, data_type: DataType) -> bool:
    """True only if the directory positively confirms entitlement.

    An unknown employee returns False rather than raising, because an
    unrecognised requester is exactly the case that should not be approved.
    """
    person = lookup(requester_id)
    if person is None:
        return False
    return data_type.value in ROLE_ENTITLEMENTS.get(person["role"], [])


def all_employees() -> List[dict]:
    if not DB_PATH.exists():
        init_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, name, role FROM employees").fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "role": r[2]} for r in rows]
