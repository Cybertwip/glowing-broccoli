"""input_health — pynput listener startup workaround + liveness supervision.

Root cause handled here: on macOS, pynput's keyboard and mouse listeners
both call ``HIServices.AXIsProcessTrusted()`` on first run. pyobjc resolves
that symbol lazily via an unlocked check-then-pop (``objc/_lazyimport.py``),
so two listeners started back-to-back race and the loser dies with
``KeyError: 'AXIsProcessTrusted'``. Touching the symbol once on the main
thread before starting any listener serializes the pop and removes the race.

Everything here is headless-safe and never raises: on non-mac platforms (or
without pyobjc) the warm-up is a no-op reported as skipped.
"""

# Warm-up outcomes.
TRUSTED = "trusted"        # symbol resolved, process is accessibility-trusted
UNTRUSTED = "untrusted"    # symbol resolved, process is NOT trusted
SKIPPED = "skipped"        # no HIServices (non-mac) — nothing to warm
FAILED = "failed"          # unexpected error while warming


def warm_darwin_trust_cache():
    """Resolve HIServices.AXIsProcessTrusted on the calling thread.

    Call once, before starting any pynput listener. Returns one of
    "trusted" / "untrusted" / "skipped: ..." / "failed: ...". Never raises.
    """
    try:
        import HIServices
    except Exception as exc:
        return f"{SKIPPED} (no HIServices: {exc})"
    try:
        trusted = bool(HIServices.AXIsProcessTrusted())
    except Exception as exc:
        return f"{FAILED} ({exc})"
    return TRUSTED if trusted else UNTRUSTED


class ListenerSupervisor:
    """Tracks thread-liveness of named listeners; reports each death once."""

    def __init__(self):
        self._known_dead = set()

    def poll(self, listeners):
        """listeners: {name: thread-like with .is_alive()}.

        Returns (alive_dict, newly_died_list). Never raises on dead threads;
        an is_alive() that raises is treated as dead.
        """
        alive = {}
        for name, thread in listeners.items():
            try:
                alive[name] = bool(thread.is_alive())
            except Exception:
                alive[name] = False
        newly = [n for n, a in alive.items()
                 if not a and n not in self._known_dead]
        self._known_dead.update(newly)
        return alive, newly


def format_input_warning(dead_names, trust_status):
    """One actionable line explaining dead input listeners."""
    names = "+".join(dead_names)
    if trust_status == UNTRUSTED:
        cause = ("macOS is not sending input events to this process. Fix: "
                 "System Settings → Privacy & Security → Accessibility → "
                 "add and enable your terminal app, then restart the widget.")
    elif trust_status == TRUSTED:
        cause = ("the process IS accessibility-trusted, so this is a "
                 "listener crash, not a permission problem — input proxies "
                 "are down until restart.")
    else:
        cause = f"listener startup state: {trust_status}."
    return (f"[input] {names} listener dead — {cause} "
            f"Camera signals (if enabled) keep working.")


def access_hint():
    return ("[input] process is not accessibility-trusted: keyboard/mouse "
            "monitoring will be silent until your terminal app is added under "
            "System Settings → Privacy & Security → Accessibility.")
