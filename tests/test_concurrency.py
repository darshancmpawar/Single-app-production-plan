"""
Unit tests for concurrency.py — gate semantics under threading.

Run with:  pytest tests/test_concurrency.py -v
"""
import importlib
import os
import threading
import time

import pytest


@pytest.fixture
def conc(monkeypatch):
    """Fresh concurrency module with small caps so tests run fast."""
    monkeypatch.setenv("MAX_CONCURRENT_PREDICTS", "2")
    monkeypatch.setenv("PREDICT_TIMEOUT_SEC", "0.5")
    monkeypatch.setenv("TRAIN_TIMEOUT_SEC", "1.0")
    import concurrency as _c
    importlib.reload(_c)
    yield _c
    # restore defaults for other tests
    importlib.reload(_c)


# ═══════════════════════════════════════════════════════════════════════
# PREDICT GATE
# ═══════════════════════════════════════════════════════════════════════
class TestPredictGate:
    def test_acquire_release_basic(self, conc):
        """Single user enters and exits cleanly."""
        with conc.predict_gate():
            assert conc.predict_slots_in_use() == 1
        assert conc.predict_slots_in_use() == 0

    def test_caps_at_max(self, conc):
        """N+1th caller blocks once N slots are held."""
        held = []
        block_done = threading.Event()

        def worker():
            with conc.predict_gate(timeout=2.0):
                held.append(threading.current_thread().name)
                block_done.wait(timeout=2.0)

        t1 = threading.Thread(target=worker, name="A")
        t2 = threading.Thread(target=worker, name="B")
        t1.start(); t2.start()

        # wait until both have grabbed slots
        deadline = time.time() + 1.0
        while len(held) < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert len(held) == 2
        assert conc.predict_slots_in_use() == 2

        # third caller must time out (cap is 2, already full)
        with pytest.raises(conc.GateBusy):
            with conc.predict_gate(timeout=0.2):
                pass

        block_done.set()
        t1.join(); t2.join()
        assert conc.predict_slots_in_use() == 0

    def test_releases_on_exception(self, conc):
        """An exception inside the with-block must still release the slot."""
        with pytest.raises(RuntimeError):
            with conc.predict_gate():
                raise RuntimeError("boom")
        assert conc.predict_slots_in_use() == 0

    def test_queues_then_proceeds(self, conc):
        """When a slot frees, a queued caller proceeds."""
        order = []

        def first():
            with conc.predict_gate(timeout=2.0):
                order.append("first-in")
                time.sleep(0.15)
                order.append("first-out")

        def second():
            time.sleep(0.05)  # arrive after `first`
            with conc.predict_gate(timeout=2.0):
                order.append("second-in")

        # Saturate one slot up-front so `first` is the only blocker for `second`.
        # MAX is 2, so we need to occupy one slot before starting.
        blocker_done = threading.Event()
        def blocker():
            with conc.predict_gate(timeout=2.0):
                blocker_done.wait(timeout=2.0)

        bt = threading.Thread(target=blocker)
        bt.start()
        time.sleep(0.05)  # let blocker grab its slot

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start(); t2.start(); t1.join(); t2.join()

        blocker_done.set(); bt.join()
        assert order == ["first-in", "first-out", "second-in"]


# ═══════════════════════════════════════════════════════════════════════
# TRAIN GATE
# ═══════════════════════════════════════════════════════════════════════
class TestTrainGate:
    def test_per_client_isolated(self, conc):
        """Different clients have independent locks."""
        a_held = threading.Event()
        a_release = threading.Event()

        def hold_a():
            with conc.train_gate("tekion", timeout=2.0):
                a_held.set()
                a_release.wait(timeout=2.0)

        ta = threading.Thread(target=hold_a)
        ta.start()
        a_held.wait(timeout=1.0)

        # Different client should NOT block on tekion's lock.
        with conc.train_gate("clario", timeout=0.2):
            pass

        a_release.set(); ta.join()

    def test_same_client_serializes(self, conc):
        """Two threads on the same client run one after the other, not in parallel."""
        timeline = []
        first_in = threading.Event()
        release = threading.Event()

        def first():
            with conc.train_gate("tekion", timeout=2.0):
                timeline.append("A-in")
                first_in.set()
                release.wait(timeout=2.0)
                timeline.append("A-out")

        def second():
            first_in.wait(timeout=1.0)
            with conc.train_gate("tekion", timeout=2.0):
                timeline.append("B-in")

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start(); t2.start()

        time.sleep(0.1)
        # B should be blocked while A holds the lock
        assert "B-in" not in timeline

        release.set()
        t1.join(); t2.join()
        assert timeline == ["A-in", "A-out", "B-in"]

    def test_timeout_raises_gatebusy(self, conc):
        """If the lock can't be acquired within timeout, raise GateBusy."""
        held = threading.Event()
        release = threading.Event()

        def holder():
            with conc.train_gate("tekion", timeout=2.0):
                held.set()
                release.wait(timeout=2.0)

        t = threading.Thread(target=holder)
        t.start()
        held.wait(timeout=1.0)

        with pytest.raises(conc.GateBusy):
            with conc.train_gate("tekion", timeout=0.1):
                pass

        release.set(); t.join()


# ═══════════════════════════════════════════════════════════════════════
# CACHE LOCK
# ═══════════════════════════════════════════════════════════════════════
class TestCacheLock:
    def test_serializes_writers(self, conc):
        """Multiple threads inside cache_lock execute one at a time."""
        order = []
        N = 5

        def writer(i):
            with conc.cache_lock():
                order.append(("in", i))
                time.sleep(0.02)
                order.append(("out", i))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(N)]
        for t in threads: t.start()
        for t in threads: t.join()

        # Every "in,i" must be immediately followed by "out,i" (no interleaving).
        for j in range(0, len(order), 2):
            assert order[j][0] == "in"
            assert order[j + 1] == ("out", order[j][1])
