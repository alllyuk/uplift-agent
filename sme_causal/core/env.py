from __future__ import annotations

"""
Lightweight .env loader.

Imports are side-effectful: on import, attempts to locate and load a
`.env` file into process environment without overriding existing vars.
If `python-dotenv` is unavailable, this module silently does nothing.
"""

try:
    from dotenv import find_dotenv, load_dotenv

    # Search upwards from CWD for a `.env` file and load it, if present.
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # If python-dotenv isn't installed or loading fails, proceed quietly.
    pass
