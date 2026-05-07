"""
Unit tests for Logic_Definer.py — RLock-guarded cache, deepcopy-on-read,
atomic file rename, and concurrent read/write safety.

Run with:  pytest tests/test_logic_definer.py -v
"""
import importlib
import json
import os
import threading
import time

import pytest


@pytest.fixture
def ld(tmp_path, monkeypatch):
    """Reload Logic_Definer with a hermetic dump path under tmp_path."""
    import Logic_Definer as _ld
    importlib.reload(_ld)
    monkeypatch.setattr(_ld, "MASSIVE_DUMP_PATH", str(tmp_path / "dump.json"))
    # Reset the module-level cache so the new path is honored.
    _ld._CACHE = None
    yield _ld
    _ld._CACHE = None


def _save(ld, ck, **fields):
    cfg = {"client_name": ck.title(), "menu_categories": [], **fields}
    ld.save_client_configuration(ck, cfg)


# ═══════════════════════════════════════════════════════════════════════
# READS
# ═══════════════════════════════════════════════════════════════════════
class TestReads:
    def test_load_returns_independent_copy(self, ld):
        _save(ld, "tekion", menu_categories=["a", "b"])
        a = ld.load_massive_dump()
        a["clients"]["tekion"]["logic"]["menu_categories"].append("HACKED")
        a["clients"]["injected"] = "evil"
        b = ld.load_massive_dump()
        assert "injected" not in b["clients"]
        assert b["clients"]["tekion"]["logic"]["menu_categories"] == ["a", "b"]

    def test_load_overrides_returns_independent_copies(self, ld):
        _save(ld, "tekion", menu_categories=["x"])
        d1 = ld.load_db_overrides()
        l1 = ld.load_logic_overrides()
        d1["tekion"]["name"] = "POISONED"
        l1["tekion"]["menu_categories"].append("POISONED")
        d2 = ld.load_db_overrides()
        l2 = ld.load_logic_overrides()
        assert d2["tekion"]["name"] == "Tekion"
        assert l2["tekion"]["menu_categories"] == ["x"]

    def test_empty_dump_has_clients_key(self, ld):
        d = ld.load_massive_dump()
        assert "clients" in d and d["clients"] == {}


# ═══════════════════════════════════════════════════════════════════════
# WRITES — round-trip + atomicity
# ═══════════════════════════════════════════════════════════════════════
class TestWrites:
    def test_save_then_load_roundtrip(self, ld):
        cfg = {
            "client_name": "Tekion",
            "menu_categories": ["Indian Bread", "Veg Curry"],
            "star_categories": ["veg curry"],
            "nonveg_mode": "Required",
            "slab_adjustments": [{"min_mg": 1, "max_mg": 100, "adjustment_pct": 5}],
            "additional_requirements": "no buffer on holidays",
        }
        ld.save_client_configuration("tekion", cfg)
        d = ld.load_massive_dump()
        logic = d["clients"]["tekion"]["logic"]
        assert logic["menu_categories"] == cfg["menu_categories"]
        assert logic["star_categories"] == cfg["star_categories"]
        assert logic["nonveg_mode"] == "Required"
        assert logic["slab_adjustments"] == cfg["slab_adjustments"]
        assert logic["additional_requirements"] == cfg["additional_requirements"]
        assert d["clients"]["tekion"]["database"]["name"] == "Tekion"

    def test_atomic_write_uses_replace(self, ld, monkeypatch):
        calls = []
        real_replace = os.replace

        def spy_replace(src, dst):
            calls.append((src, dst))
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy_replace)
        _save(ld, "tekion", menu_categories=["a"])
        assert len(calls) == 1
        src, dst = calls[0]
        assert dst == ld.MASSIVE_DUMP_PATH
        assert not os.path.exists(src)            # temp consumed by rename

    def test_no_temp_files_left_behind(self, ld):
        for i in range(5):
            _save(ld, "tekion", menu_categories=[f"c{i}"])
        d = os.path.dirname(ld.MASSIVE_DUMP_PATH)
        leftover = [f for f in os.listdir(d) if f.startswith(".massive_dump.")]
        assert leftover == []

    def test_save_failure_does_not_corrupt_file_or_cache(self, ld, monkeypatch):
        _save(ld, "tekion", menu_categories=["original"])
        good_disk = json.loads(open(ld.MASSIVE_DUMP_PATH).read())
        good_cache = ld.load_massive_dump()

        # Make os.replace blow up to simulate a mid-write failure.
        def bad_replace(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", bad_replace)
        with pytest.raises(OSError):
            _save(ld, "tekion", menu_categories=["replacement"])

        # File on disk is untouched.
        assert json.loads(open(ld.MASSIVE_DUMP_PATH).read()) == good_disk
        # Cache is untouched (still pointing at the original snapshot).
        assert ld.load_massive_dump() == good_cache
        # No leftover .tmp files.
        d = os.path.dirname(ld.MASSIVE_DUMP_PATH)
        leftover = [f for f in os.listdir(d) if f.startswith(".massive_dump.")]
        assert leftover == []

    def test_cache_invalidation_after_save(self, ld):
        _save(ld, "tekion", menu_categories=["v1"])
        # On-disk and in-memory must match exactly after save.
        on_disk = json.loads(open(ld.MASSIVE_DUMP_PATH).read())
        from_cache = ld.load_massive_dump()
        assert on_disk == from_cache


# ═══════════════════════════════════════════════════════════════════════
# CONCURRENCY — readers must never observe a torn or empty state
# ═══════════════════════════════════════════════════════════════════════
class TestConcurrency:
    def test_concurrent_reads_during_writes(self, ld):
        # Seed with one client so readers always have something to inspect.
        _save(ld, "tekion", menu_categories=["seed"])

        N_READERS = 6
        READER_ITERS = 1500
        WRITER_ITERS = 80
        errors: list[str] = []
        stop_writers = threading.Event()

        def reader():
            try:
                for _ in range(READER_ITERS):
                    d = ld.load_massive_dump()
                    # Every snapshot must be internally consistent.
                    assert "clients" in d, "missing 'clients' key"
                    for ck, entry in d["clients"].items():
                        assert isinstance(entry, dict), f"client entry not dict: {entry!r}"
                        # logic block, when present, must be a dict (never None / partial)
                        if "logic" in entry:
                            assert isinstance(entry["logic"], dict)
                            assert "menu_categories" in entry["logic"]
            except Exception as e:
                errors.append(f"reader: {e!r}")
            finally:
                stop_writers.set()

        def writer():
            try:
                i = 0
                while not stop_writers.is_set() and i < WRITER_ITERS:
                    ld.save_client_configuration("tekion", {
                        "client_name": "Tekion",
                        "menu_categories": [f"c{i}_{j}" for j in range(20)],
                        "star_categories": [],
                        "nonveg_mode": "Optional",
                        "slab_adjustments": [],
                        "additional_requirements": "",
                    })
                    i += 1
            except Exception as e:
                errors.append(f"writer: {e!r}")

        threads = [threading.Thread(target=reader) for _ in range(N_READERS)] + [
            threading.Thread(target=writer)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert errors == [], f"concurrency errors: {errors}"

    def test_clear_override_caches_thread_safe(self, ld):
        _save(ld, "tekion", menu_categories=["x"])
        errors: list[str] = []

        def churn():
            try:
                for _ in range(500):
                    ld._clear_override_caches()
                    d = ld.load_massive_dump()
                    assert "clients" in d
            except Exception as e:
                errors.append(repr(e))

        threads = [threading.Thread(target=churn) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert errors == []
