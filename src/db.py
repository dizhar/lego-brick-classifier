"""Database connection for the LEGO part catalog.

Reads PostgreSQL connection details from environment variables. The defaults
match the local Cloud SQL proxy setup; in production you'd override these.

  PG_USER       (default: postgres)
  PG_PASSWORD   (required — no default for security)
  PG_HOST       (default: 127.0.0.1)
  PG_PORT       (default: 5432)
  PG_DB         (default: LEGO_DB)

Usage from the app:

    from src.db import load_part_catalog

    catalog = load_part_catalog()        # cached, loads once per session
    info = catalog.get('3001')           # → {'name': 'Brick 2 x 4', 'broad_category': 'Bricks'}
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def _connection_url() -> str:
    """Build the SQLAlchemy URL from environment variables."""
    user = os.environ.get("PG_USER", "postgres")
    password = os.environ.get("PG_PASSWORD", "")
    host = os.environ.get("PG_HOST", "127.0.0.1")
    port = os.environ.get("PG_PORT", "5432")
    db = os.environ.get("PG_DB", "LEGO_DB")

    if not password:
        raise RuntimeError(
            "PG_PASSWORD environment variable is not set. "
            "Run `export PG_PASSWORD='your_password'` before starting Streamlit."
        )

    return f"postgresql+pg8000://{user}:{password}@{host}:{port}/{db}"


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine. One per Streamlit session."""
    return create_engine(_connection_url(), pool_pre_ping=True)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading LEGO part catalog from database…")
def load_part_catalog() -> Dict[str, Dict[str, str]]:
    """Load the lego_parts table into a dict keyed by bricklink_identifier.

    Returns:
        { bricklink_identifier: {'name': str, 'title': str, 'label': str,
                                 'broad_category': str, 'sub_category': str} }

    The 'name' field is what the UI should display — it's set to whichever of
    title/label is non-empty (title preferred). This lets us swap which column
    drives the UI without touching every call site.

    Cached by Streamlit — runs once per session. If you change the lego_parts
    table while the app is running, restart Streamlit to see the update.
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT bricklink_identifier, title, label, broad_category, sub_category
            FROM lego_parts
            WHERE bricklink_identifier IS NOT NULL
        """))
        rows = result.fetchall()

    catalog: Dict[str, Dict[str, str]] = {}
    for r in rows:
        bricklink = str(r[0])
        title = str(r[1]) if r[1] is not None else ""
        label = str(r[2]) if r[2] is not None else ""
        # Your data has: title = "LEGO Brick 2 x 4 (3001 / 72841)" (cluttered)
        # and:           label = "Brick 2 x 4" (clean) → prefer label.
        display_name = label.strip() or title.strip()
        catalog[bricklink] = {
            "name":           display_name,
            "title":          title,
            "label":          label,
            "broad_category": str(r[3]) if r[3] is not None else "",
            "sub_category":   str(r[4]) if r[4] is not None else "",
        }
    return catalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def lookup_part(
    catalog: Dict[str, Dict[str, str]],
    bricklink_id: str,
) -> Optional[Dict[str, str]]:
    """Look up a part by bricklink ID. Returns None if not found."""
    return catalog.get(bricklink_id)


def format_part_label(
    catalog: Dict[str, Dict[str, str]],
    bricklink_id: str,
    max_len: int = 28,
    fallback_prefix: str = "#",
) -> str:
    """Get a display-friendly label for a bricklink ID.

    If the part is in the catalog, returns the name (truncated to max_len).
    Otherwise returns "<fallback_prefix><bricklink_id>".
    """
    info = catalog.get(bricklink_id)
    if info and info.get("name"):
        name = info["name"]
        if len(name) > max_len:
            name = name[: max_len - 1] + "…"
        return name
    return f"{fallback_prefix}{bricklink_id}"