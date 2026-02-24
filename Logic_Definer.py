"""Helpers to persist modular client configuration overrides as Python modules."""

from __future__ import annotations

import os
from functools import lru_cache
from pprint import pformat
from typing import Any

DIR = os.path.dirname(__file__)
LOGIC_OVERRIDES_MODULE = os.path.join(DIR, "client_logic_overrides.py")
DB_OVERRIDES_MODULE = os.path.join(DIR, "client_database_overrides.py")


def _read_module_dict(path: str, var_name: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    namespace: dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, path, "exec"), {}, namespace)
    data = namespace.get(var_name, {})
    return data if isinstance(data, dict) else {}


def _write_module_dict(path: str, var_name: str, payload: dict[str, Any], doc: str) -> None:
    pretty = pformat(payload, width=100, sort_dicts=True)
    content = f'"""{doc}"""\n\n{var_name} = {pretty}\n'
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@lru_cache(maxsize=1)
def load_logic_overrides() -> dict[str, Any]:
    return _read_module_dict(LOGIC_OVERRIDES_MODULE, "LOGIC_OVERRIDES")


@lru_cache(maxsize=1)
def load_db_overrides() -> dict[str, Any]:
    return _read_module_dict(DB_OVERRIDES_MODULE, "DB_OVERRIDES")


def _clear_override_caches() -> None:
    load_logic_overrides.cache_clear()
    load_db_overrides.cache_clear()


def save_client_configuration(client_key: str, config: dict[str, Any]) -> None:
    """Persist required Tab-1 fields in modular override files."""
    ck = client_key.strip().lower()

    logic_overrides = dict(load_logic_overrides())
    logic_overrides[ck] = {
        "mode": config.get("mode", "Embedded"),
        "menu_categories": config.get("menu_categories", []),
        "star_categories": config.get("star_categories", []),
        "nonveg_mode": config.get("nonveg_mode", "Optional"),
        "nonveg_item_count": int(config.get("nonveg_item_count", 1) or 1),
        "slab_adjustments": config.get("slab_adjustments", []),
        "category_repeats": config.get("category_repeats", []),
        "calculation_config": config.get("calculation_config", {}),
        "custom_bump_pct": float(config.get("custom_bump_pct", 0.0) or 0.0),
    }
    _write_module_dict(
        LOGIC_OVERRIDES_MODULE,
        "LOGIC_OVERRIDES",
        logic_overrides,
        "Auto-generated logic overrides by Config New Client tab.",
    )

    db_overrides = dict(load_db_overrides())
    db_overrides.setdefault(ck, {})["name"] = config.get("client_name", ck.title())
    db_overrides.setdefault(ck, {})["has_embeddings"] = config.get("mode", "Embedded") == "Embedded"
    _write_module_dict(
        DB_OVERRIDES_MODULE,
        "DB_OVERRIDES",
        db_overrides,
        "Auto-generated database overrides by Config New Client tab.",
    )

    _clear_override_caches()
