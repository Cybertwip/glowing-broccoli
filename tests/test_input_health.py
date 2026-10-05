"""Maintained tests for input_health (stdlib unittest, headless)."""

import sys
import threading
import unittest

import input_health as IH


class _FakeThread:
    def __init__(self, alive=True, boom=False):
        self._alive = alive
        self._boom = boom

    def is_alive(self):
        if self._boom:
            raise RuntimeError("boom")
        return self._alive

    def kill(self):
        self._alive = False


class TestWarmDarwinTrustCache(unittest.TestCase):
    def test_never_raises_returns_known_status(self):
        status = IH.warm_darwin_trust_cache()
        self.assertIsInstance(status, str)
        first = status.split(" ")[0]
        self.assertIn(first, (IH.TRUSTED, IH.UNTRUSTED, IH.SKIPPED, IH.FAILED))

    def test_darwin_resolves_symbol(self):
        if sys.platform != "darwin":
            self.skipTest("macOS-only")
        status = IH.warm_darwin_trust_cache()
        self.assertIn(status, (IH.TRUSTED, IH.UNTRUSTED))
        import HIServices  # noqa: PLC0415
        # Post-warm the symbol is resolved — no lazyimport pop, no KeyError.
        self.assertIsInstance(bool(HIServices.AXIsProcessTrusted()), bool)

    def test_warmed_concurrent_touch_has_no_race(self):
        if sys.platform != "darwin":
            self.skipTest("macOS-only")
        status = IH.warm_darwin_trust_cache()
        if status not in (IH.TRUSTED, IH.UNTRUSTED):
            self.skipTest(f"warm-up did not resolve ({status})")
        import HIServices  # noqa: PLC0415
        barrier = threading.Barrier(2)
        errors = []

        def touch():
            try:
                barrier.wait(timeout=5)
                HIServices.AXIsProcessTrusted()
            except BaseException as exc:  # noqa: BLE001
                errors.append(repr(exc))

        threads = [threading.Thread(target=touch) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestListenerSupervisor(unittest.TestCase):
    def test_all_alive_no_news(self):
        sup = IH.ListenerSupervisor()
        alive, newly = sup.poll({"kb": _FakeThread(True),
                                 "mouse": _FakeThread(True)})
        self.assertEqual(alive, {"kb": True, "mouse": True})
        self.assertEqual(newly, [])

    def test_death_reported_once(self):
        kb, mouse = _FakeThread(True), _FakeThread(True)
        sup = IH.ListenerSupervisor()
        sup.poll({"kb": kb, "mouse": mouse})
        kb.kill()
        alive, newly = sup.poll({"kb": kb, "mouse": mouse})
        self.assertEqual(alive, {"kb": False, "mouse": True})
        self.assertEqual(newly, ["kb"])
        # Second poll: still dead, but not newly.
        _, newly = sup.poll({"kb": kb, "mouse": mouse})
        self.assertEqual(newly, [])
        mouse.kill()
        _, newly = sup.poll({"kb": kb, "mouse": mouse})
        self.assertEqual(newly, ["mouse"])

    def test_raising_is_alive_counts_as_dead(self):
        sup = IH.ListenerSupervisor()
        alive, newly = sup.poll({"kb": _FakeThread(boom=True)})
        self.assertEqual(alive, {"kb": False})
        self.assertEqual(newly, ["kb"])


class TestMessages(unittest.TestCase):
    def test_warning_names_dead_and_guides_untrusted(self):
        msg = IH.format_input_warning(["kb", "mouse"], IH.UNTRUSTED)
        self.assertIn("kb+mouse", msg)
        self.assertIn("Accessibility", msg)
        self.assertIn("restart", msg)

    def test_warning_trusted_says_crash_not_permission(self):
        msg = IH.format_input_warning(["kb"], IH.TRUSTED)
        self.assertIn("kb", msg)
        self.assertIn("not a permission problem", msg)

    def test_access_hint_actionable(self):
        self.assertIn("Accessibility", IH.access_hint())


if __name__ == "__main__":
    unittest.main()
