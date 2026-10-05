"""Maintained tests for eye_tracker (stdlib unittest, headless).

Covers the pure blink/presence logic, the null fallback, and the threaded
tracker's graceful degradation with an invalid camera index. No real camera
or display is required.
"""

import time
import unittest

import eye_tracker as ET


def _feed(det, frames, start=1000.0, step=1.0 / 12.0):
    """Feed (eyes_visible, face_visible) frames; return list of update results."""
    out = []
    t = start
    for eyes, face in frames:
        out.append(det.update(eyes, face, t))
        t += step
    return out


class TestBlinkDetector(unittest.TestCase):
    def test_blink_on_missing_then_visible(self):
        det = ET.BlinkDetector()
        res = _feed(det, [(True, True),          # arm
                          (False, True), (False, True),   # missing x2
                          (True, True)])         # blink completes
        self.assertEqual(res, [False, False, False, True])

    def test_single_missing_frame_counts(self):
        det = ET.BlinkDetector()
        res = _feed(det, [(True, True), (False, True), (True, True)])
        self.assertEqual(res[-1], True)

    def test_too_many_missing_is_not_a_blink(self):
        det = ET.BlinkDetector(max_missing=4)
        res = _feed(det, [(True, True)] + [(False, True)] * 6 + [(True, True)])
        self.assertNotIn(True, res)

    def test_face_loss_resets_without_blink(self):
        det = ET.BlinkDetector()
        res = _feed(det, [(True, True), (False, True),
                          (False, False), (False, False),  # face gone
                          (True, True)])
        self.assertNotIn(True, res)

    def test_eyes_never_seen_no_blink(self):
        det = ET.BlinkDetector()
        res = _feed(det, [(False, True)] * 10)
        self.assertNotIn(True, res)

    def test_debounce(self):
        det = ET.BlinkDetector(min_interval=1.0)
        t = 1000.0
        self.assertFalse(det.update(True, True, t))
        self.assertFalse(det.update(False, True, t + 0.1))
        self.assertTrue(det.update(True, True, t + 0.2))   # blink 1
        self.assertFalse(det.update(False, True, t + 0.3))
        # Too soon after blink 1 → suppressed, but re-arms.
        self.assertFalse(det.update(True, True, t + 0.4))
        self.assertFalse(det.update(False, True, t + 1.5))
        self.assertTrue(det.update(True, True, t + 1.6))   # blink 2

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            ET.BlinkDetector(max_missing=0)
        with self.assertRaises(ValueError):
            ET.BlinkDetector(min_interval=0)


class TestPresenceLatch(unittest.TestCase):
    def test_latch_and_timeout(self):
        latch = ET.PresenceLatch(timeout=1.5)
        self.assertFalse(latch.update(False, 1000.0))  # never seen
        self.assertTrue(latch.update(True, 1001.0))
        self.assertTrue(latch.update(False, 1002.0))   # within timeout
        self.assertFalse(latch.update(False, 1002.6))  # beyond timeout
        self.assertTrue(latch.update(True, 1003.0))    # re-seen

    def test_bad_timeout(self):
        with self.assertRaises(ValueError):
            ET.PresenceLatch(timeout=0)


class TestNullAndFactory(unittest.TestCase):
    def test_gaze_reading_defaults(self):
        r = ET.GazeReading()
        self.assertFalse(r.valid)
        self.assertFalse(r.face_present)
        self.assertEqual(r.blinks_total, 0)

    def test_null_tracker(self):
        n = ET.NullTracker()
        n.start()
        self.assertFalse(n.read().valid)
        n.stop()  # must not raise

    def test_factory_disabled(self):
        t = ET.create_tracker(False)
        self.assertIsInstance(t, ET.NullTracker)
        self.assertFalse(t.read().valid)

    def test_factory_enabled_type(self):
        t = ET.create_tracker(True, index=99)
        self.assertIsInstance(t, ET.EyeTracker)
        self.assertFalse(t.read().valid)  # not started → invalid


class _StubCap:
    """Cap-like object replaying scripted read() results."""

    def __init__(self, script):
        self._script = list(script)
        self.released = False

    def read(self):
        if self._script:
            return self._script.pop(0)
        return False, None

    def release(self):
        self.released = True


class _FakeFrame:
    def __init__(self, size=10, shape=(240, 320, 3)):
        self.size = size
        self.shape = shape


class TestReadFrame(unittest.TestCase):
    def test_immediate_frame(self):
        f = _FakeFrame()
        self.assertIs(ET._read_frame(_StubCap([(True, f)]), 3, 0.0), f)

    def test_fail_then_frame(self):
        f = _FakeFrame()
        cap = _StubCap([(False, None), (True, None), (True, f)])
        self.assertIs(ET._read_frame(cap, 3, 0.0), f)

    def test_all_fail_returns_none(self):
        cap = _StubCap([(False, None), (True, None)])
        self.assertIsNone(ET._read_frame(cap, 5, 0.0))

    def test_empty_frame_rejected(self):
        cap = _StubCap([(True, _FakeFrame(size=0))])
        self.assertIsNone(ET._read_frame(cap, 1, 0.0))

    def test_read_exception_returns_none(self):
        class _Boom:
            def read(self):
                raise RuntimeError("boom")
        self.assertIsNone(ET._read_frame(_Boom(), 2, 0.0))


class TestProbeCameras(unittest.TestCase):
    def test_invalid_index_structure(self):
        res = ET.probe_cameras(indices=[99])
        self.assertEqual(len(res), 1)
        r = res[0]
        self.assertEqual(r["index"], 99)
        for key in ("backend", "opened", "frame_ok", "shape", "error"):
            self.assertIn(key, r)
        self.assertFalse(r["opened"])
        self.assertFalse(r["frame_ok"])

    def test_report_never_raises(self):
        ET.print_camera_report([])
        ET.print_camera_report(ET.probe_cameras(indices=[99]))


class TestEyeTrackerFallback(unittest.TestCase):
    def test_invalid_index_stays_invalid(self):
        t = ET.EyeTracker(index=99)
        t.start()
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline and t.reason == "not started":
                time.sleep(0.05)
            self.assertNotEqual(t.reason, "not started")
            self.assertFalse(t.read().valid)
        finally:
            t0 = time.time()
            t.stop()
            self.assertLess(time.time() - t0, ET.STOP_JOIN_SECS + 2.0)

    def test_double_start_and_stop_safe(self):
        t = ET.EyeTracker(index=99)
        t.start()
        t.start()
        t.stop()
        t.stop()  # must not raise


if __name__ == "__main__":
    unittest.main()
