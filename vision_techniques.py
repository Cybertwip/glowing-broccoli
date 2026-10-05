"""vision_techniques — pure helpers for vision-hygiene exercises.

These are comfort / habit exercises (accommodation cycling, smooth pursuit,
saccadic refixation, peripheral awareness, blink breaks). They are NOT a
medical treatment and do not correct refractive error (myopia, hyperopia,
astigmatism). See DISCLAIMER.

Deliberately pygame-free so unit tests can import this module headless.
"""

import math

DISCLAIMER = (
    "Comfort exercises only — not medical advice. Nothing here changes "
    "refractive error. Stop if you feel dizzy or get a headache, and see "
    "an eye-care professional for vision problems."
)

_TWO_PI = 2.0 * math.pi

# Technique ids. "NONE" means keep the widget's existing pattern.
ACCOMMODATIVE_ROCK = "ACCOMMODATIVE_ROCK"
SACCADE_GRID = "SACCADE_GRID"
FIGURE8_PURSUIT = "FIGURE8_PURSUIT"
PERIPHERAL_RING = "PERIPHERAL_RING"
BLINK_BREAK = "BLINK_BREAK"
NONE = "NONE"

ALL_TECHNIQUES = (
    ACCOMMODATIVE_ROCK,
    SACCADE_GRID,
    FIGURE8_PURSUIT,
    PERIPHERAL_RING,
    BLINK_BREAK,
)

_INSTRUCTIONS = {
    ACCOMMODATIVE_ROCK: "Follow the dot: BIG = near, small = far. Breathe slowly.",
    SACCADE_GRID: "Flick eyes to the bright dot only — keep head still.",
    FIGURE8_PURSUIT: "Track the dot smoothly around the figure-8. No head moves.",
    PERIPHERAL_RING: "Stare at the centre dot; notice the outer dots fading.",
    BLINK_BREAK: "10 slow full blinks, then gaze 20 ft away for 20 s.",
    NONE: "",
}


def describe(technique):
    """One-line user instruction for a technique id ("" for NONE/unknown)."""
    return _INSTRUCTIONS.get(technique, "")


def accom_scale(t, period=10.0):
    """Near/far cycle for accommodative rock.

    Returns (scale, label) with scale in [0,1] (0=far/small, 1=near/large).
    Cosine ramp gives a ~5 s near hold and ~5 s far hold at period=10.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    phase = (t % period) / period
    scale = 0.5 - 0.5 * math.cos(_TWO_PI * phase)
    return scale, ("NEAR" if scale > 0.5 else "FAR")


def figure8_pos(t, ax, ay, speed=0.6):
    """Figure-8 (lemniscate) offset for smooth-pursuit tracking.

    x spans [-ax, ax]; y spans [-ay/2, ay/2]. Pure function of time.
    """
    w = t * speed
    return ax * math.sin(w), ay * math.sin(2.0 * w) / 2.0


# Saccade order jumps across the grid (opposite corners first) so each
# step is a large refixation, then settles toward centre / edges.
_SACCADE_ORDER = (
    (0, 0), (2, 2), (2, 0), (0, 2), (1, 1),
    (0, 1), (2, 1), (1, 0), (1, 2),
)


def saccade_target(step, grid=3):
    """Grid (col, row) for an integer step; cycles through _SACCADE_ORDER."""
    if grid != 3:
        raise ValueError("only a 3x3 grid is supported")
    return _SACCADE_ORDER[int(step) % len(_SACCADE_ORDER)]


def saccade_index(t, dwell=0.8):
    """Which saccade step is active at time t with dwell seconds per target."""
    if dwell <= 0:
        raise ValueError("dwell must be positive")
    return int(t / dwell)


def peripheral_twinkle(t, index, count=12, rate=2.0):
    """Brightness in [0,1] for one peripheral dot; neighbours desync via index."""
    if count <= 0:
        raise ValueError("count must be positive")
    return 0.5 + 0.5 * math.sin(t * rate + index * _TWO_PI / count)


def blink_due(last_prompt, now, interval=20.0):
    """True when a blink reminder should fire (interval seconds elapsed)."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    return (now - last_prompt) >= interval


def recommend_technique(state, ciliary, composite, blink_supp):
    """Pick one exercise id from strain inputs. Returns NONE to keep default.

    Priority: blink breaks first (dryness is fastest to relieve), then
    accommodative rock for tired focusing muscle, then state-specific drills.
    """
    if blink_supp >= 50.0:
        return BLINK_BREAK
    if ciliary >= 60.0:
        return ACCOMMODATIVE_ROCK
    if state == "CORRECTIVE":
        return SACCADE_GRID
    if state == "READING":
        return FIGURE8_PURSUIT
    if state == "IDLE":
        return PERIPHERAL_RING
    return NONE
