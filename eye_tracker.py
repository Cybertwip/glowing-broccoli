"""eye_tracker — optional camera blink + presence tracking.

Backend: OpenCV Haar cascades only (no new dependency beyond opencv-python,
no ML model download). Faces give presence (is the user at the screen?);
eyes-visible → missing → visible transitions while a face is present count
as blinks.

Privacy: frames are processed in memory and discarded. Nothing is displayed,
recorded, or transmitted — the module exposes only small numeric readings
(face present? how many blinks?). There is deliberately no preview API.

Robustness: cv2 is imported lazily inside the capture thread, so importing
this module never fails. Any failure (no opencv, no camera, permission
denied, read error) degrades to an invalid reading — never an exception —
so the caller can silently fall back to its mouse/keyboard proxies.
"""

import threading
import time

# Capture pacing and retry policy.
CAPTURE_HZ = 12.0
FRAME_WIDTH = 320
OPEN_RETRY_SECS = 5.0
STOP_JOIN_SECS = 2.0
WARMUP_ATTEMPTS = 10    # reads to wait out camera warmup after open
WARMUP_SLEEP = 0.1      # pause between warmup reads
READ_RETRIES = 3        # per-frame read retries before declaring failure
PROBE_INDICES = (0, 1, 2)  # device indices scanned by --check-camera

# Smoothing / blink tuning.
PRESENCE_TIMEOUT = 1.5    # face still "present" if seen within this window
BLINK_MAX_MISSING = 4     # frames of missing eyes that still count as a blink
BLINK_MIN_INTERVAL = 0.25  # seconds of debounce between blinks


class GazeReading:
    """Point-in-time snapshot. valid=False means "no camera data"."""

    __slots__ = ("valid", "face_present", "blinks_total")

    def __init__(self, valid=False, face_present=False, blinks_total=0):
        self.valid = valid
        self.face_present = face_present
        self.blinks_total = blinks_total

    def copy(self):
        return GazeReading(self.valid, self.face_present, self.blinks_total)


class BlinkDetector:
    """Frame-by-frame blink state machine (pure logic, no cv2).

    A blink is eyes-visible → missing for 1..max_missing frames →
    eyes-visible again, all while a face is continuously present.
    Losing the face resets the machine without emitting a blink.
    """

    def __init__(self, max_missing=BLINK_MAX_MISSING,
                 min_interval=BLINK_MIN_INTERVAL):
        if max_missing < 1:
            raise ValueError("max_missing must be >= 1")
        if min_interval <= 0:
            raise ValueError("min_interval must be positive")
        self._max_missing = max_missing
        self._min_interval = min_interval
        self._armed = False
        self._missing = 0
        self._last_blink = float("-inf")

    def update(self, eyes_visible, face_visible, now):
        """Feed one frame; returns True exactly when a blink completes."""
        if not face_visible:
            self._armed = False
            self._missing = 0
            return False
        if eyes_visible:
            if (self._armed and 0 < self._missing <= self._max_missing
                    and (now - self._last_blink) >= self._min_interval):
                self._last_blink = now
                self._armed = False
                self._missing = 0
                return True
            self._armed = True
            self._missing = 0
            return False
        if self._armed:
            self._missing += 1
            if self._missing > self._max_missing:
                self._armed = False  # looked away / lost eyes, not a blink
        return False


class PresenceLatch:
    """Smooths flickery face detection: present if seen within timeout."""

    def __init__(self, timeout=PRESENCE_TIMEOUT):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._timeout = timeout
        self._last_seen = float("-inf")

    def update(self, seen, now):
        if seen:
            self._last_seen = now
        return (now - self._last_seen) < self._timeout


class NullTracker:
    """Same interface as EyeTracker, but never produces data."""

    def __init__(self, reason="camera disabled"):
        self._reason = reason

    @property
    def reason(self):
        return self._reason

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
        return GazeReading(valid=False)


class EyeTracker:
    """Threaded OpenCV capture producing GazeReadings. Never raises."""

    def __init__(self, index=0):
        self._index = index
        self._lock = threading.Lock()
        self._reading = GazeReading(valid=False)
        self._reason = "not started"
        self._stop = threading.Event()
        self._thread = None
        self._warned_reason = None

    @property
    def reason(self):
        with self._lock:
            return self._reason

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="eye-tracker")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=STOP_JOIN_SECS)
            self._thread = None

    def read(self):
        with self._lock:
            return self._reading.copy()

    def _set(self, valid, face_present=False, blinks_total=0, reason=""):
        with self._lock:
            prev_valid = self._reading.valid
            self._reading = GazeReading(valid, face_present, blinks_total)
            if reason:
                self._reason = reason
        # Confirm recovery, and note each distinct failure once (silent fallback).
        if valid and not prev_valid:
            print(f"[camera] tracking started (index {self._index})")
        elif not valid and reason and reason != self._warned_reason:
            self._warned_reason = reason
            print(f"[camera] {reason} — falling back to input proxies")

    def _run(self):
        try:
            import cv2
        except ImportError:
            self._set(False, reason="opencv (cv2) not installed")
            return

        face_cascade, eye_cascade = self._load_cascades(cv2)
        if face_cascade is None or eye_cascade is None:
            self._set(False, reason="haar cascades unavailable")
            return

        detector = BlinkDetector()
        presence = PresenceLatch()
        blinks = 0
        cap = None
        last_open_attempt = 0.0
        pace = 1.0 / CAPTURE_HZ

        while not self._stop.is_set():
            t0 = time.time()
            if cap is None and t0 - last_open_attempt >= OPEN_RETRY_SECS:
                last_open_attempt = t0
                cap = self._try_open(cv2)
                if cap is None:
                    self._set(False, reason=(
                        f"index {self._index}: camera unavailable "
                        f"(denied/busy/missing/wrong index)"))
                    time.sleep(pace)
                    continue
                # Warmup: freshly opened cameras often yield empty frames
                # for ~1 s; wait them out instead of reporting failure.
                if _read_frame(cap, WARMUP_ATTEMPTS, WARMUP_SLEEP) is None:
                    self._release(cap)
                    cap = None
                    self._set(False, reason=(
                        f"index {self._index}: opened but no frames "
                        f"(another app holding the camera?)"))
                    continue
            if cap is None:
                time.sleep(pace)
                continue

            frame = _read_frame(cap, READ_RETRIES, 0.0)
            if frame is None:
                self._release(cap)
                cap = None
                self._set(False, reason=(
                    f"index {self._index}: camera read failed"))
                continue

            face_seen, eyes_seen = self._detect(cv2, frame,
                                               face_cascade, eye_cascade)
            if detector.update(eyes_seen, face_seen, t0):
                blinks += 1
            present = presence.update(face_seen, t0)
            self._set(True, face_present=present, blinks_total=blinks,
                      reason="tracking")

            spent = time.time() - t0
            time.sleep(max(0.0, pace - spent))

        if cap is not None:
            self._release(cap)

    @staticmethod
    def _load_cascades(cv2):
        try:
            base = cv2.data.haarcascades
            face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
            eye = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
            if face.empty() or eye.empty():
                return None, None
            return face, eye
        except Exception:
            return None, None

    @staticmethod
    def _release(cap):
        try:
            cap.release()
        except Exception:
            pass

    def _try_open(self, cv2):
        # Default backend first, then AVFoundation (macOS cameras sometimes
        # need it explicitly — open succeeds but reads fail otherwise).
        attempts = [(self._index,)]
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            attempts.append((self._index, cv2.CAP_AVFOUNDATION))
        for args in attempts:
            try:
                cap = cv2.VideoCapture(*args)
            except Exception:
                continue
            try:
                if not cap.isOpened():
                    continue
                # Small frames: detection is cheaper, still plenty for blinks.
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_WIDTH * 3 // 4)
                return cap
            except Exception:
                continue
        return None

    @staticmethod
    def _detect(cv2, frame, face_cascade, eye_cascade):
        """Returns (face_seen, eyes_seen). Frame is never stored or shown."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            return False, False
        h, w = gray.shape[:2]
        if w > FRAME_WIDTH:
            scale = FRAME_WIDTH / float(w)
            gray = cv2.resize(gray, (FRAME_WIDTH, int(h * scale)))
        try:
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        except Exception:
            return False, False
        if len(faces) == 0:
            return False, False
        # Largest face wins (closest person).
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        roi = gray[fy:fy + int(fh * 0.6), fx:fx + fw]  # upper face: eyes live here
        if roi.size == 0:
            return True, False
        try:
            eyes = eye_cascade.detectMultiScale(
                roi, scaleFactor=1.1, minNeighbors=4,
                minSize=(12, 12), maxSize=(fw // 2, fh // 3))
        except Exception:
            return True, False
        return True, len(eyes) >= 1


def _read_frame(cap, attempts, sleep_secs):
    """Read one non-empty frame with retries; None if all attempts fail."""
    for _ in range(max(1, attempts)):
        try:
            ok, frame = cap.read()
        except Exception:
            ok, frame = False, None
        if ok and frame is not None and getattr(frame, "size", 0):
            return frame
        if sleep_secs > 0:
            time.sleep(sleep_secs)
    return None


def probe_cameras(indices=None):
    """Probe webcam indices; returns a list of result dicts. Never raises.

    Each result has keys: index, backend ("default"/"avfoundation"/"none"),
    opened (bool), frame_ok (bool), shape (frame shape or None).
    Opens each device only long enough for a warmup read, then releases it.
    """
    results = []
    try:
        import cv2
    except ImportError:
        return [{"index": i, "backend": "none", "opened": False,
                 "frame_ok": False, "shape": None,
                 "error": "opencv (cv2) not installed"}
                for i in (indices if indices is not None else PROBE_INDICES)]
    for i in indices if indices is not None else PROBE_INDICES:
        entry = {"index": i, "backend": "none", "opened": False,
                 "frame_ok": False, "shape": None, "error": ""}
        backends = [(), ("avfoundation",)] if hasattr(
            cv2, "CAP_AVFOUNDATION") else [()]
        for be in backends:
            try:
                if be:
                    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                    name = "avfoundation"
                else:
                    cap = cv2.VideoCapture(i)
                    name = "default"
            except Exception as exc:
                entry["error"] = f"open raised {exc}"
                continue
            try:
                if not cap.isOpened():
                    entry["error"] = "open failed"
                    continue
                entry["opened"] = True
                entry["backend"] = name
                frame = _read_frame(cap, WARMUP_ATTEMPTS, WARMUP_SLEEP)
                if frame is None:
                    entry["error"] = "opened but no frames"
                    continue
                entry["frame_ok"] = True
                try:
                    entry["shape"] = tuple(frame.shape)
                except Exception:
                    entry["shape"] = None
                break
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
        results.append(entry)
    return results


def print_camera_report(results):
    """Print a human-readable probe report with next-step hints."""
    print("Camera probe (opens each device briefly; nothing is stored):")
    if results and results[0].get("error") == "opencv (cv2) not installed":
        print("  opencv (cv2) not installed — run: pip install opencv-python")
        return
    working = [r for r in results if r["frame_ok"]]
    for r in results:
        if r["frame_ok"]:
            status = f"OK ({r['backend']}, frame {r['shape']})"
        elif r["opened"]:
            status = f"opens but no frames [{r['error']}]"
        else:
            status = f"unavailable [{r['error'] or 'no device'}]"
        print(f"  index {r['index']}: {status}")
    if working:
        best = working[0]["index"]
        print(f"Run with: python3 main.py --enable-camera --camera-index {best}")
    else:
        print("No working camera found. Try:")
        print("  - macOS: allow the camera permission prompt (System Settings → "
              "Privacy & Security → Camera) and re-run")
        print("  - Close other apps holding the camera (Zoom/Teams/browsers)")
        print("  - Unplug/replug external webcams, then re-run --check-camera")


def create_tracker(enabled, index=0):
    """Factory: EyeTracker when enabled, NullTracker otherwise. Never raises."""
    if not enabled:
        return NullTracker(reason="camera disabled")
    try:
        return EyeTracker(index=index)
    except Exception as exc:  # pragma: no cover — defensive
        return NullTracker(reason=f"tracker init failed: {exc}")
