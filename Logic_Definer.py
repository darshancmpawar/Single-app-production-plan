"""Helpers to persist client configuration into a single massive dump file."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

DIR = os.path.dirname(__file__)
MASSIVE_DUMP_PATH = os.path.join(DIR, "client_massive_dump.json")


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


@lru_cache(maxsize=1)
def load_massive_dump() -> dict[str, Any]:
    data = _read_json(MASSIVE_DUMP_PATH)
    if "clients" not in data:
        data["clients"] = {}
    return data


def _clear_override_caches() -> None:
    load_massive_dump.cache_clear()


def load_logic_overrides() -> dict[str, Any]:
    clients = load_massive_dump().get("clients", {})
    return {ck: (cfg.get("logic") or {}) for ck, cfg in clients.items()}


def load_db_overrides() -> dict[str, Any]:
    clients = load_massive_dump().get("clients", {})
    return {ck: (cfg.get("database") or {}) for ck, cfg in clients.items()}


def save_client_configuration(client_key: str, config: dict[str, Any]) -> None:
    """Persist required Tab-1 fields for a selected client key into one dump file."""
    ck = client_key.strip().lower()

    massive = load_massive_dump()
    clients = massive.setdefault("clients", {})
    clients.setdefault(ck, {})

    clients[ck]["logic"] = {
        "menu_categories": config.get("menu_categories", []),
        "star_categories": config.get("star_categories", []),
        "nonveg_mode": config.get("nonveg_mode", "Optional"),
        "slab_adjustments": config.get("slab_adjustments", []),
        "additional_requirements": config.get("additional_requirements", ""),
    }
    clients[ck]["database"] = {
        "name": config.get("client_name", ck.title()),
    }

    _write_json(MASSIVE_DUMP_PATH, massive)
    _clear_override_caches()
