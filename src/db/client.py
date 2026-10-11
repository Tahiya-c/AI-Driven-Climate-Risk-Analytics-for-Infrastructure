"""One Supabase client for the server, built from the secret key.

The URL and secret key come from ``.streamlit/secrets.toml`` (``SUPABASE_URL``,
``SUPABASE_SECRET_KEY``) or, outside Streamlit, from environment variables / ``.env``
with the same names. The secret key bypasses row-level security, so it must only
ever be used here, on the server -- never sent to the browser.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

from src.config import ROOT_DIR


class DatabaseNotConfigured(RuntimeError):
    pass


def _setting(name: str) -> str | None:
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml, or not running under Streamlit
        pass
    load_dotenv(ROOT_DIR / ".env")
    return os.environ.get(name)


@lru_cache(maxsize=1)
def get_client():
    """The shared Supabase client (created once per process)."""
    from supabase import create_client

    url, key = _setting("SUPABASE_URL"), _setting("SUPABASE_SECRET_KEY")
    if not url or not key:
        raise DatabaseNotConfigured(
            "Supabase is not configured: add SUPABASE_URL and SUPABASE_SECRET_KEY to "
            ".streamlit/secrets.toml (see docs/database_plan.md)."
        )
    return create_client(url, key)
