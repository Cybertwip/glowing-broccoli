"""Maintained tests for vision_techniques (stdlib unittest, headless)."""

import math
import unittest

import vision_techniques as VT


class TestAccomScale(unittest.TestCase):
    def test_range_and_labels(self):
        for t in [0, 1.0, 2.5, 3.0, 5.0, 5.9, 10.0, 123.4]:
            scale, label = VT.accom_scale(t, period=6.0)
            self.assertGreaterEqual(scale, 0.0)
            self.assertLessEqual(scale, 1.0)
            self.assertIn(label, ("NEAR", "FAR"))

    def test_cycle_endpoints(self):
        s0, l0 = VT.accom_scale(0.0, period=6.0)
        s_mid, l_mid = VT.accom_scale(3.0, period=6.0)
        self.assertAlmostEqual(s0, 0.0)
        self.assertEqual(l0, "FAR")
        self.assertAlmostEqual(s_mid, 1.0)
        self.assertEqual(l_mid, "NEAR")

    def test_bad_period(self):
        with self.assertRaises(ValueError):
            VT.accom_scale(0.0, period=0)


class TestFigure8(unittest.TestCase):
    def test_bounds(self):
        for i in range(63):
            x, y = VT.figure8_pos(i * 0.1, 50.0, 40.0, speed=1.0)
            self.assertLessEqual(abs(x), 50.0 + 1e-9)
            self.assertLessEqual(abs(y), 20.0 + 1e-9)

    def test_origin_and_symmetry(self):
        x0, y0 = VT.figure8_pos(0.0, 50.0, 40.0)
        self.assertAlmostEqual(x0, 0.0)
        self.assertAlmostEqual(y0, 0.0)
        xa, ya = VT.figure8_pos(1.0, 50.0, 40.0)
        xb, yb = VT.figure8_pos(-1.0, 50.0, 40.0)
        self.assertAlmostEqual(xa, -xb)
        self.assertAlmostEqual(ya, -yb)

    def test_full_loop_returns(self):
        x0, y0 = VT.figure8_pos(0.0, 50.0, 40.0, speed=1.0)
        x1, y1 = VT.figure8_pos(2.0 * math.pi, 50.0, 40.0, speed=1.0)
        self.assertAlmostEqual(x0, x1, places=6)
        self.assertAlmostEqual(y0, y1, places=6)


class TestSaccade(unittest.TestCase):
    def test_order_covers_grid(self):
        seen = {VT.saccade_target(i) for i in range(9)}
        self.assertEqual(len(seen), 9)
        for c, r in seen:
            self.assertIn(c, (0, 1, 2))
            self.assertIn(r, (0, 1, 2))

    def test_first_jump_is_maximal(self):
        # Opposite corners first: largest possible refixation.
        self.assertEqual(VT.saccade_target(0), (0, 0))
        self.assertEqual(VT.saccade_target(1), (2, 2))

    def test_index_dwell(self):
        self.assertEqual(VT.saccade_index(0.0, dwell=0.8), 0)
        self.assertEqual(VT.saccade_index(0.79, dwell=0.8), 0)
        self.assertEqual(VT.saccade_index(0.8, dwell=0.8), 1)
        with self.assertRaises(ValueError):
            VT.saccade_index(0.0, dwell=0)


class TestPeripheral(unittest.TestCase):
    def test_range_and_desync(self):
        vals = [VT.peripheral_twinkle(1.0, i) for i in range(12)]
        for v in vals:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        # Neighbours must differ (desynchronised twinkle).
        self.assertGreater(max(vals) - min(vals), 0.1)


class TestRecommend(unittest.TestCase):
    def test_blink_first(self):
        self.assertEqual(
            VT.recommend_technique("CORRECTIVE", 90.0, 90.0, 80.0),
            VT.BLINK_BREAK)

    def test_ciliary_second(self):
        self.assertEqual(
            VT.recommend_technique("TYPING", 70.0, 10.0, 0.0),
            VT.ACCOMMODATIVE_ROCK)

    def test_state_mapping(self):
        self.assertEqual(
            VT.recommend_technique("CORRECTIVE", 0.0, 0.0, 0.0),
            VT.SACCADE_GRID)
        self.assertEqual(
            VT.recommend_technique("READING", 0.0, 0.0, 0.0),
            VT.FIGURE8_PURSUIT)
        self.assertEqual(
            VT.recommend_technique("IDLE", 0.0, 0.0, 0.0),
            VT.PERIPHERAL_RING)

    def test_passthrough(self):
        self.assertEqual(
            VT.recommend_technique("TYPING", 0.0, 0.0, 0.0), VT.NONE)
        self.assertEqual(
            VT.recommend_technique("SCROLLING", 0.0, 0.0, 0.0), VT.NONE)
        self.assertEqual(
            VT.recommend_technique("RECOVERY", 0.0, 0.0, 0.0), VT.NONE)

    def test_all_have_instructions(self):
        for t in VT.ALL_TECHNIQUES:
            self.assertTrue(VT.describe(t))
        self.assertEqual(VT.describe(VT.NONE), "")
        self.assertEqual(VT.describe("BOGUS"), "")


class TestBlinkDue(unittest.TestCase):
    def test_interval(self):
        self.assertFalse(VT.blink_due(100.0, 110.0, interval=20.0))
        self.assertTrue(VT.blink_due(100.0, 120.0, interval=20.0))
        with self.assertRaises(ValueError):
            VT.blink_due(0.0, 1.0, interval=0)


if __name__ == "__main__":
    unittest.main()
