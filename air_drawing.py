
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  AIR DRAW PHYSICS  v10  ─  Two-Hand · Zero-Mirror · Velocity-Throw               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  INSTALL:  pip install opencv-python mediapipe pymunk pygame numpy               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  RIGHT HAND  (drawing)                                                           ║
║   ☝  INDEX ONLY extended      → DRAW  (ink follows INDEX_FINGER_TIP, no mirror)║
║        12 BRUSHES: pencil marker crayon pen neon fire ice lightning particle   ║
║        galaxy water smoke  -> hover the BRUSH tab (top-right) 0.5s, then a row ║
║   🤌  Pinch index+thumb       → RESIZE (pen width in real-time)                 ║
║   🤏  Pinch thumb+MIDDLE      → GRAB  (either hand: pick up, move, THROW)      ║
║        while holding:  roll wrist = ROTATE   |  BOTH hands pinch = SCALE       ║
║                        hand toward camera = PUSH  |  hand back = PULL          ║
║   👊  Fist                    → ERASE (circular rubber eraser)                 ║
║   ✌  Peace sign (hold 0.5 s)  → LAUNCH (drawings fall with physics)            ║
║                                                                                ║
║  LEFT HAND  (physics sandbox)                                                  ║
║   👊  Fist                    → PUSH  (push nearest object away)               ║
║   🖐  Open palm               → BLAST (area explosion at hand position)        ║
║                                                                                ║
║  FINGER UI  (no keyboard required)                                             ║
║   Hover over COLOR SWATCH  (top-right)   0.5 s → change color                ║
║   Hover over TOOLBAR BUTTON (bottom)     0.5 s → activate                    ║
║   Hold right open palm                   1.0 s → radial menu                 ║
║                                                                                ║
║  KEYBOARD SHORTCUTS  (fallback / power-user)                                  ║
║   SPACE  launch    C  clear    Z  undo    G  gravity    T  trails             ║
║   E  explode   S  screenshot   +/-  pen   1-8  color   D  debug  Q/ESC quit  ║
║   [ ]  scale held/hovered object     P  push (hold)     O  pull (hold)         ║
║   B  brush menu    TAB / Shift+TAB  next / prev brush    F1..F12  pick brush   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  ── WHY DRAWING FOLLOWS YOUR FINGER EXACTLY (NO MIRROR) ───────────────────── ║
║                                                                                ║
║  The camera frame is captured RAW — never flipped.                             ║
║  MediaPipe receives the RAW frame and returns lm.x / lm.y in [0, 1]          ║
║  normalised to the image it was given.                                         ║
║                                                                                ║
║  We map those to display pixels ONCE and ONLY ONCE:                            ║
║      disp_x = lm.x * WIN_W                                                    ║
║      disp_y = lm.y * WIN_H                                                    ║
║                                                                                ║
║  The pygame background is also built from the same raw frame (no h-flip).    ║
║  So every layer — camera image, hand skeleton, ink, physics bodies, cursor — ║
║  all share the exact same coordinate space.                                    ║
║  Result:  hand-right == screen-right.  Drawing "L" shows "L", never "⅃".     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import mediapipe as mp
import sys
import datetime
import math
import colorsys
import os
import random

import gc
import zlib
import re
import threading
import time
from collections import deque
import dataclasses
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Callable, Any
import os as _os
_os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
_os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2
import numpy as np
import pygame
import pymunk




WEBCAM_INDEX = 1
CAM_W,  CAM_H = 1280, 720
# ── CAMERA CLARITY (display only - hand tracking always uses the raw frame) ──────────────────────────────
CAM_SHARPEN = float(os.environ.get("AIRDRAW_SHARPEN", "0.6"))   # 0 = off, 0.6 = gentle, 1.2 = strong
CAM_UPSCALE = cv2.INTER_CUBIC                                   # used when the stream is smaller than the window
WIN_W,  WIN_H = 1280, 720
FPS_CAP = 60
THREADED_TRACKING = True      # hand tracking runs in its own thread: the UI never waits for MediaPipe
DETECT_CONF = 0.55
TRACK_CONF  = 0.45
MAX_HANDS = 2
EMA_FAST = 0.55
EMA_SLOW = 0.30
MIN_MOVE_PX = 2
DRAW_GRACE_S = 0.25        # a stroke waits this long through lost tracking / a misread gesture before it is closed
DENSIFY_STEP = 3.0
VEL_HISTORY  = 6
GRAVITY_PRESETS = [(0, 900), (0, -900), (0, 0)]
GRAVITY_LABELS = ["↓ DOWN", "↑ UP", "✦ ZERO-G"]
DEFAULT_GRAVITY = 0

PHYS_STEPS = 3
BODY_MASS = 1.5
ELASTICITY = 0.50
FRICTION = 0.65
WALL_W = 6
WALL_THICK = 90                # real collision thickness; the inner face stays exactly where WALL_W put it
MAX_BODY_SPEED = 3600.0        # px/s hard limit (a thrown / exploded drawing can never outrun the walls)
PHYS_MAX_SUBSTEP_PX = 36.0     # slow frame => more sub-steps so nothing moves further than this per step
PHYS_MAX_SUBSTEPS = 12
SEG_RADIUS = 4
SELECT_RADIUS = 90

DRAG_MAX_FORCE = 60_000
DRAG_ERR_BIAS = (1 - 0.15) ** 60 
EXPLODE_FORCE = 130_000
PUSH_FORCE = 45_000

DEFAULT_COLOR_KEY = "1"
DEFAULT_PEN_W = 5
MIN_PEN_W = 2
MAX_PEN_W = 28
INK_ALPHA = 240

PINCH_CLOSE = 0.04
PINCH_OPEN  = 0.20

PALETTE: Dict[str, Dict] = {
    "1": {"name": "Red",     "rgb": (220,  55,  55)},
    "2": {"name": "Green",   "rgb": ( 55, 210,  70)},
    "3": {"name": "Blue",    "rgb": ( 55, 110, 230)},
    "4": {"name": "Yellow",  "rgb": (230, 210,  40)},
    "5": {"name": "Cyan",    "rgb": ( 40, 220, 210)},
    "6": {"name": "Magenta", "rgb": (210,  50, 200)},
    "7": {"name": "White",   "rgb": (245, 245, 245)},
    "8": {"name": "Black",   "rgb": ( 20,  20,  20)},
}

LAUNCH_HOLD_S = 0.50                      # peace sign must be held this long (no accidental launches while drawing)
RADIAL_HOLD_S = 1.00
UI_HOVER_S    = 0.50

ERASE_RADIUS = 55
TRAIL_LEN = 26
TRAIL_ALPHA = 115
CURSOR_OUTER = 16
CURSOR_INNER = 5
GLOW_R = 30
FINGER_TRAIL_LEN = 12

P_LAUNCH = 22;   P_EXPLODE = 55
P_GRAV = 420;  P_SPMIN = 80; P_SPMAX  = 340
P_LMIN = 0.40; P_LMAX = 1.10; P_RMIN   = 2;    P_RMAX = 6

SCREENSHOT_DIR = "screenshots"
DEFAULT_BRUSH = "pen"          # brush active at start (pen = closest to the old plain ink)

# ── GESTURE GEOMETRY (all distances are normalised by palm size, so they are
#    independent of camera distance and hand rotation) ───────────────────────
USE_Z_IN_GESTURES = True       # use MediaPipe z so fingers pointing at the camera still count
# finger "reach" = dist(wrist, tip) / dist(wrist, knuckle).  ~2.0 straight, ~0.9 curled
INDEX_EXTENDED_THRESHOLD = 1.55     # index must reach at least this to draw
INDEX_MIN_PIP_ANGLE      = 135.0    # ...and be reasonably straight (degrees)
INDEX_FOLDED_THRESHOLD   = 1.35     # fist test
MIDDLE_FOLDED_THRESHOLD  = 1.40
RING_FOLDED_THRESHOLD    = 1.40
PINKY_FOLDED_THRESHOLD   = 1.40
FINGER_EXTENDED_THRESHOLD = 1.45    # open palm test (launch / blast)
THUMB_FOLDED_THRESHOLD   = 0.85     # dist(thumb tip, index knuckle) / palm
REQUIRE_THUMB_FOLDED_FOR_DRAW = False   # True = strict spec; False = thumb may stick out
GESTURE_HYSTERESIS = 0.12           # extra tolerance once a gesture is already active

GRAB_PINCH_THRESHOLD   = 0.34       # thumb-tip <-> middle-tip, enter GRAB
GRAB_RELEASE_THRESHOLD = 0.58       # ...leave GRAB (hysteresis)
GRAB_MIN_MIDDLE_REACH  = 1.10       # middle must not be curled into a fist
GRAB_HOVER_THRESHOLD   = 0.75       # show "selectable" highlight when this close
GRAB_SELECT_RADIUS     = 115        # px from stroke surface
GRAB_ACQUIRE_S         = 0.60       # after pinching, how long we may still latch onto an object
GRAB_SMOOTH            = 0.55       # EMA on the pinch point
GRAB_HOLD_GRACE        = 0.15       # s a held 3-D object survives a lost / blinking hand
MAX_THROW_SPEED        = 2800.0     # px/s clamp on release

RESIZE_ENTER = 0.55                 # thumb-tip <-> index-tip to enter RESIZE
RESIZE_EXIT  = 1.30
RESIZE_MIN_D = 0.15                 # maps to MIN_PEN_W
RESIZE_MAX_D = 1.10                 # maps to MAX_PEN_W

# ── PHASE 2: grabbed-object manipulation ────────────────────────────────────
ENABLE_GRAB_ROTATION = True    # roll your wrist (or turn two hands) to rotate the held object
ROT_DEADZONE_DEG     = 5.0     # ignore tiny wrist jitter before rotation "arms"
ROT_SMOOTH           = 0.35
ROT_GAIN             = 1.0
ROT_MAX_TORQUE       = 2e6
MAX_SPIN             = 12.0    # rad/s clamp for spin-on-release

ENABLE_GRAB_SCALE    = True    # two hands pinching: change distance to scale
SCALE_MIN, SCALE_MAX = 0.35, 3.0
SCALE_KEY_STEP       = 0.02    # per frame while [ or ] is held

ENABLE_PUSH_PULL     = True    # hand toward camera = PUSH others away, hand back = PULL them in
PUSHPULL_DEADBAND    = 0.18    # palm must grow/shrink by this fraction to engage
PUSHPULL_FULL        = 0.35    # ln-ratio at which strength reaches 100%
PUSHPULL_RADIUS      = 380     # px
PUSHPULL_FORCE       = 30_000  # impulse/sec at full strength, point-blank

# frames a new gesture must persist before it becomes active (anti-flicker)
GESTURE_ENTER_FRAMES = {"DRAW": 2, "GRAB": 2, "ERASE": 3, "PINCH": 3,
                        "LAUNCH": 4, "PALM": 3, "PUSH": 3, "BLAST": 2, "IDLE": 3}

class WebcamStream:
    def __init__(self, idx: int, w: int, h: int):
        self.cap = cv2.VideoCapture("http://10.243.219.195:4747/video")
        if not self.cap.isOpened():
            sys.exit(f"\n❌  Camera {idx} not found.  Edit WEBCAM_INDEX.\n")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = f
                    self._seq += 1

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def read_with_seq(self):
        """(sequence number, frame) of the newest camera frame - no copy, the capture thread never edits it."""
        with self._lock:
            return None if self._frame is None else (self._seq, self._frame)

    def stop(self):
        self._stop.set()
        self.cap.release()

#DUAL-STAGE EMA SMOOTHER
class DualEMA:
    def __init__(self, af: float = EMA_FAST, as_: float = EMA_SLOW):
        self._af, self._as = af, as_
        self._x1 = self._y1 = None
        self._x2 = self._y2 = None

    def update(self, x: float, y: float) -> Tuple[float, float]:
        # speed of the (lightly filtered) fingertip: tremor barely registers -> heavy smoothing; real movement -> light smoothing, no lag
        if self._x1 is None:
            self._x1, self._y1 = float(x), float(y); self._x2, self._y2 = self._x1, self._y1
            return self._x2, self._y2
        step = 0.45 * math.hypot(x - self._x1, y - self._y1)
        k = max(0.0, min(1.0, (step - 4.0) / 20.0))
        k = k * k * (3 - 2 * k)
        af = 0.45 + (0.88 - 0.45) * k
        as_ = 0.24 + (0.65 - 0.24) * k
        self._x1 = af * x + (1 - af) * self._x1
        self._y1 = af * y + (1 - af) * self._y1
        self._x2 = as_ * self._x1 + (1 - as_) * self._x2
        self._y2 = as_ * self._y1 + (1 - as_) * self._y2
        return self._x2, self._y2

    def reset(self):
        self._x1 = self._y1 = self._x2 = self._y2 = None

class VelocityTracker:
    """
    Stores the last VEL_HISTORY (timestamp, x, y) tuples.
    .velocity() returns (vx, vy) in px/s averaged over the stored window.
    Applied to a body on grab release so it inherits the throw momentum.
    """
    def __init__(self, history: int = VEL_HISTORY):
        self._h: Deque[Tuple[float, float, float]] = deque(maxlen=history)

    def update(self, x: float, y: float):
        self._h.append((time.perf_counter(), x, y))

    def velocity(self) -> Tuple[float, float]:
        if len(self._h) < 2:
            return (0.0, 0.0)
        t0, x0, y0 = self._h[0]
        t1, x1, y1 = self._h[-1]
        dt = t1 - t0
        if dt < 1e-6:
            return (0.0, 0.0)
        return ((x1 - x0) / dt, (y1 - y0) / dt)

    def reset(self):
        self._h.clear()

@dataclass
class HandFeatures:
    """Orientation-independent geometry of one hand (debug + classification)."""
    reach: Dict[str, float]          # index/middle/ring/pinky  wrist-reach ratio
    index_angle: float               # PIP angle of index (deg)
    tm: float                        # thumb tip <-> middle tip  / palm
    ti: float                        # thumb tip <-> index tip   / palm
    thumb_in: float                  # thumb tip <-> index knuckle / palm
    mid_reach: float                 # wrist <-> middle tip / palm
    palm: float
    raw: str = "IDLE"
    size: float = 0.0                # 2-D palm size (depth cue: bigger = closer)


@dataclass
class HandData:
    label: str
    gesture: str
    tip: Optional[Tuple[int, int]]       # INDEX_FINGER_TIP (screen px)
    mid: Optional[Tuple[int, int]]
    thumb: Optional[Tuple[int, int]]
    pinch_d: Optional[float]             # raw thumb-index distance (legacy)
    landmarks: object
    grab_pt: Optional[Tuple[int, int]] = None   # midpoint thumb tip / middle tip
    feats: Optional[HandFeatures] = None
    angle: float = 0.0                           # wrist->middle-knuckle direction, screen space (rad)

    @property
    def size(self) -> float:
        return self.feats.size if self.feats else 0.0

    @property
    def tm(self) -> Optional[float]:
        return self.feats.tm if self.feats else None


class GestureEngine:
    """
    Pure-geometry gesture classifier + per-hand debouncer.
    Needs only objects with .x .y .z  (so it can be unit-tested without a camera).

    Priority:  GRAB > ERASE(fist) > RESIZE(pinch) > DRAW > LAUNCH > IDLE
    """
    MCP = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
    TIP = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}

    def __init__(self):
        self._stable: Dict[str, str] = {}
        self._cand:   Dict[str, str] = {}
        self._count:  Dict[str, int] = {}

    # ---- geometry ----
    @staticmethod
    def features(lm, aspect: float = 16 / 9) -> HandFeatures:
        zs = aspect if USE_Z_IN_GESTURES else 0.0
        P = lambda i: (lm[i].x * aspect, lm[i].y, lm[i].z * zs)
        d = lambda a, b: math.dist(P(a), P(b))
        palm = max(d(0, 9), 1e-6)
        reach = {k: d(0, GestureEngine.TIP[k]) / max(d(0, GestureEngine.MCP[k]), 1e-6)
                 for k in GestureEngine.MCP}
        # angle at index PIP between (mcp-pip) and (tip-pip)
        a, b, c = P(5), P(6), P(8)
        v1 = [a[i] - b[i] for i in range(3)]
        v2 = [c[i] - b[i] for i in range(3)]
        n1 = math.sqrt(sum(x * x for x in v1)) or 1e-6
        n2 = math.sqrt(sum(x * x for x in v2)) or 1e-6
        cosv = max(-1.0, min(1.0, sum(x * y for x, y in zip(v1, v2)) / (n1 * n2)))
        d2 = lambda a, b: math.hypot((lm[a].x - lm[b].x) * aspect, lm[a].y - lm[b].y)
        size = 0.5 * (d2(0, 9) + 1.2 * d2(5, 17))
        return HandFeatures(
            reach=reach, index_angle=math.degrees(math.acos(cosv)),
            tm=d(4, 12) / palm, ti=d(4, 8) / palm, thumb_in=d(4, 5) / palm,
            mid_reach=d(0, 12) / palm, palm=palm, size=size)

    # ---- raw (un-debounced) classification ----
    def classify_raw(self, label: str, f: HandFeatures, prev: str) -> str:
        h = GESTURE_HYSTERESIS
        R = f.reach
        right = (label == "Right")

        # 1. GRAB  (thumb + middle pinch; never a fist)
        if prev == "GRAB":
            grab = f.tm <= GRAB_RELEASE_THRESHOLD and f.mid_reach >= GRAB_MIN_MIDDLE_REACH - 0.1
        else:
            grab = (f.tm <= GRAB_PINCH_THRESHOLD
                    and f.mid_reach >= GRAB_MIN_MIDDLE_REACH
                    and f.tm <= f.ti * 1.05)
        if grab:
            return "GRAB"

        # 2. ERASE / PUSH  (fist)
        e = h if prev in ("ERASE", "PUSH") else 0.0
        if (R["index"] <= INDEX_FOLDED_THRESHOLD + e and R["middle"] <= MIDDLE_FOLDED_THRESHOLD + e
                and R["ring"] <= RING_FOLDED_THRESHOLD + e and R["pinky"] <= PINKY_FOLDED_THRESHOLD + e):
            return "ERASE" if right else "PUSH"

        # 3. RESIZE  (thumb-index pinch, other three fingers folded; right hand only)
        dh = h if prev == "DRAW" else 0.0
        others_folded = (R["middle"] <= MIDDLE_FOLDED_THRESHOLD + dh
                         and R["ring"] <= RING_FOLDED_THRESHOLD + dh
                         and R["pinky"] <= PINKY_FOLDED_THRESHOLD + dh)
        if right and others_folded and f.ti <= (RESIZE_EXIT if prev == "PINCH" else RESIZE_ENTER):
            return "PINCH"

        # 4. DRAW  (index ONLY)
        idx_ext = (R["index"] >= INDEX_EXTENDED_THRESHOLD - dh
                   and f.index_angle >= INDEX_MIN_PIP_ANGLE - 10 * (dh > 0))
        thumb_ok = (not REQUIRE_THUMB_FOLDED_FOR_DRAW) or f.thumb_in <= THUMB_FOLDED_THRESHOLD + dh
        if idx_ext and others_folded and thumb_ok:
            return "DRAW"

        # 4b. LAUNCH  (right hand PEACE sign: index + middle up, ring + pinky folded)
        mth = FINGER_EXTENDED_THRESHOLD + (0.30 if prev == "DRAW" else 0.12)        # middle finger clearly up (and even more so mid-stroke)
        if (right and R["index"] >= INDEX_EXTENDED_THRESHOLD and R["middle"] >= mth
                and R["ring"] <= RING_FOLDED_THRESHOLD + dh and R["pinky"] <= PINKY_FOLDED_THRESHOLD + dh):
            return "LAUNCH"

        # 5. open palm: right = PALM (radial menu only, no launch), left = BLAST
        if all(R[k] >= FINGER_EXTENDED_THRESHOLD for k in R):
            return "PALM" if right else "BLAST"
        return "IDLE"

    # ---- debounce ----
    def update(self, label: str, lm, aspect: float = 16 / 9) -> Tuple[str, HandFeatures]:
        f = self.features(lm, aspect)
        prev = self._stable.get(label, "IDLE")
        raw = self.classify_raw(label, f, prev)
        f.raw = raw
        if raw == prev:
            self._cand[label], self._count[label] = raw, 0
            return prev, f
        if self._cand.get(label) == raw:
            self._count[label] += 1
        else:
            self._cand[label], self._count[label] = raw, 1
        if self._count[label] >= GESTURE_ENTER_FRAMES.get(raw, 2):
            self._stable[label] = raw
            self._count[label] = 0
            return raw, f
        return prev, f

    def reset(self, label: Optional[str] = None):
        for dct in (self._stable, self._cand, self._count):
            if label is None: dct.clear()
            else: dct.pop(label, None)


class GestureClassifier:
    """MediaPipe wrapper: landmarks -> HandData (screen px, zero-mirror)."""

    def __init__(self):
        mh = mp.solutions.hands
        self._hands = mh.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=DETECT_CONF,
            min_tracking_confidence=TRACK_CONF,
        )
        self._mh     = mh
        self._mpdraw = mp.solutions.drawing_utils
        self.engine  = GestureEngine()
        self._launch_t: Dict[str, float] = {"Right": 0.0, "Left": 0.0}
        self._radial_t: Dict[str, float] = {"Right": 0.0, "Left": 0.0}

    def _timers(self, label: str, gesture: str, t: float):
        lw, rw = ("LAUNCH", "PALM") if label == "Right" else ("BLAST", "BLAST")
        if gesture == lw:
            if not self._launch_t[label]: self._launch_t[label] = t
        else:
            self._launch_t[label] = 0.0
        if gesture == rw:
            if not self._radial_t[label]: self._radial_t[label] = t
        else:
            self._radial_t[label] = 0.0

    def process(self, rgb_frame: np.ndarray) -> List[HandData]:
        result = self._hands.process(rgb_frame)
        out: List[HandData] = []
        fh, fw = rgb_frame.shape[:2]
        aspect = fw / max(fh, 1)

        if not result.multi_hand_landmarks:
            self._launch_t = {"Right": 0.0, "Left": 0.0}
            self._radial_t = {"Right": 0.0, "Left": 0.0}
            self.engine.reset()
            return out

        t = time.perf_counter()
        seen = set()

        for lm_obj, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            lm    = lm_obj.landmark
            label = handedness.classification[0].label
            # MediaPipe assumes a mirrored selfie image; we feed the raw frame -> swap
            label = "Right" if label == "Left" else "Left"
            seen.add(label)

            def px(i: int) -> Tuple[int, int]:
                return (int(lm[i].x * WIN_W), int((1.0 - lm[i].y) * WIN_H))

            tip   = px(mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP)   # draw point
            mid   = px(12)
            thumb = px(4)
            grab_pt = ((thumb[0] + mid[0]) // 2, (thumb[1] + mid[1]) // 2)
            pinch_d = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)

            gesture, feats = self.engine.update(label, lm, aspect)
            self._timers(label, gesture, t)
            w0, k9 = px(0), px(9)
            ang = math.atan2(k9[1] - w0[1], k9[0] - w0[0])
            out.append(HandData(label, gesture, tip, mid, thumb, pinch_d, lm_obj, grab_pt, feats, ang))

        for lab in ("Right", "Left"):
            if lab not in seen:
                self.engine.reset(lab)
                self._launch_t[lab] = self._radial_t[lab] = 0.0
        return out

    def launch_ready(self, hand: str = "Right") -> bool:
        t0 = self._launch_t.get(hand, 0.0)
        return t0 > 0 and (time.perf_counter() - t0) >= LAUNCH_HOLD_S

    def radial_ready(self, hand: str = "Right") -> bool:
        t0 = self._radial_t.get(hand, 0.0)
        return t0 > 0 and (time.perf_counter() - t0) >= RADIAL_HOLD_S

    def reset_launch(self, hand: str = "Right"):
        self._launch_t[hand] = time.perf_counter() + 99999.0

    def reset_radial(self, hand: str = "Right"):
        self._radial_t[hand] = time.perf_counter() + 99999.0

    def draw_skeleton(self, bgr_frame: np.ndarray, lm_obj, label: str):
        """Overlay the hand skeleton on the BGR camera frame (in-place)."""
        col_dot = (90, 210, 90) if label == "Right" else (90, 160, 230)
        self._mpdraw.draw_landmarks(
            bgr_frame, lm_obj, self._mh.HAND_CONNECTIONS,
            self._mpdraw.DrawingSpec(color=col_dot, thickness=2, circle_radius=3),
            self._mpdraw.DrawingSpec(color=(200, 200, 200), thickness=1),
        )

    def close(self):
        self._hands.close()

@dataclass
class PhysBody:
    body: pymunk.Body
    shapes: List[pymunk.Shape]
    color: Tuple[int, int, int]
    width: int
    trail: Deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    frozen: bool  = False
    selected: bool  = False
    hover: bool = False
    uid: int = 0
    scale: float = 1.0
    extra: str = ""
    brush: str = DEFAULT_BRUSH
    sprite: object = None                 # cached brush rendering (local coords)
    sprite_origin: Tuple[int, int] = (0, 0)
    sprite_scale: float = 0.0
    sprite_t: float = -1e9
    rot_q: int = -9999
    rot_surf: object = None
    effect: Optional[str] = None          # Phase 4 particle effect key
    ring: bool = False                    # effect on a circle -> energy ring
    shape: str = ""                       # recognised shape kind of the stroke it came from
    opacity: float = 1.0                  # Phase 5 editable properties
    glow: float = 0.0
    visible: bool = True
    mass_mult: float = 1.0
    elast: float = ELASTICITY
    fric: float = FRICTION
    gscale: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  BRUSH ENGINE  (Phase 3)
#    Brush        -> how a stroke LOOKS (layered passes) and what it EMITS
#    StrokeLayers -> per-pass opaque layers, composited with per-pass alpha
#                    (so overlapping segments never accumulate into a solid blob)
#    FXSystem     -> capped particle engine (additive glow / soft / shard / star)
#    BrushPanel   -> finger / mouse / keyboard brush menu
# ═══════════════════════════════════════════════════════════════════════════════
FX_MAX              = 1500    # hard cap on live particles (brushes + effects)
FX_AMBIENT_BUDGET   = 14      # max ambient emissions per frame (launched/pending strokes)
SPRITE_REBUILD_BUDGET = 2     # animated launched strokes re-rendered per frame
BRUSH_ORDER = ["pencil", "marker", "crayon", "pen", "neon", "fire",
               "ice", "lightning", "particle", "galaxy", "water", "smoke"]


def _mix(a, b, t):
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))

def _scl(c, f):
    return (max(0, min(255, int(c[0] * f))), max(0, min(255, int(c[1] * f))), max(0, min(255, int(c[2] * f))))

def _saturate(c):
    """Push a colour to full neon saturation (greys / black become cool white-blue)."""
    h, s, v = colorsys.rgb_to_hsv(c[0] / 255, c[1] / 255, c[2] / 255)
    if v < 0.3 or s < 0.12:
        return (190, 200, 255) if v < 0.3 else (235, 240, 255)
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))

def _hue_shift(c, dh):
    h, s, v = colorsys.rgb_to_hsv(c[0] / 255, c[1] / 255, c[2] / 255)
    r, g, b = colorsys.hsv_to_rgb((h + dh) % 1.0, max(s, 0.55), max(v, 0.85))
    return (int(r * 255), int(g * 255), int(b * 255))


@dataclass
class PassSpec:
    wm: float                      # width multiplier
    rgb: Tuple[int, int, int]
    alpha: int                     # composite alpha of the whole pass (0-255)
    line: bool = True              # False = dots-only pass


class StrokeCtx:
    """Per-stroke running state (so a stroke can be painted point by point)."""
    __slots__ = ("rng", "passes", "rgb", "w", "brush_key", "last", "dist",
                 "phase", "aj", "ap", "ad", "wp")

    def __init__(self, rng, passes, rgb, w, brush_key, start, phase=0.0):
        self.rng, self.passes, self.rgb, self.w, self.brush_key = rng, passes, rgb, w, brush_key
        self.last = start
        self.dist = 0.0
        self.phase = phase
        self.aj = self.ap = self.wp = None
        self.ad = 0.0


class Brush:
    key = "pen"; name = "Pen"
    animated = False; anim_hz = 0.0          # animated launched strokes get re-rendered at anim_hz
    emits = False; amb_rate = 0.0            # ambient emissions per second per ~200px of stroke

    def passes(self, rgb) -> List[PassSpec]:
        return [PassSpec(1.0, rgb, 255)]

    def width_var(self, ctx: StrokeCtx) -> float:
        return 1.0

    def new_ctx(self, rgb, w, start, seed=None, phase=0.0) -> StrokeCtx:
        return StrokeCtx(random.Random(seed), self.passes(rgb), rgb, w, self.key, start, phase)

    def feed(self, ctx: StrokeCtx, p, w):
        """Next stroke point -> drawing primitives."""
        d = math.hypot(p[0] - ctx.last[0], p[1] - ctx.last[1])
        ctx.dist += d
        prims = self.segment(ctx, ctx.last, p, w)
        ctx.last = p
        return prims

    def segment(self, ctx, p1, p2, w):
        wv = self.width_var(ctx)
        return [("L", i, p1, p2, max(1, int(round(w * ps.wm * wv))))
                for i, ps in enumerate(ctx.passes) if ps.line]

    def emit(self, fx: "FXSystem", p, rgb, w, rng, now):
        pass


class Pencil(Brush):
    key, name = "pencil", "Pencil"
    def passes(self, rgb):
        return [PassSpec(0.55, _mix(rgb, (70, 70, 70), 0.22), 175),
                PassSpec(0.0, _scl(rgb, 0.55), 170, line=False)]
    def width_var(self, ctx):
        return 0.85 + 0.25 * math.sin(ctx.dist * 0.07) * math.sin(ctx.dist * 0.023 + 1.0)
    def segment(self, ctx, p1, p2, w):
        prims = super().segment(ctx, p1, p2, w)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        for _ in range(1 + (w > 4)):                       # paper grain
            t, o = ctx.rng.random(), ctx.rng.uniform(-0.55, 0.55) * w
            prims.append(("D", 1, (p1[0] + dx * t + nx * o, p1[1] + dy * t + ny * o), 1,
                          _scl(ctx.rgb, ctx.rng.uniform(0.45, 0.9))))
        return prims


class Marker(Brush):
    key, name = "marker", "Marker"
    def passes(self, rgb):
        return [PassSpec(2.5, rgb, 55), PassSpec(1.9, rgb, 150)]      # soft edge + translucent body


class Crayon(Brush):
    key, name = "crayon", "Crayon"
    def passes(self, rgb):
        return [PassSpec(1.25, rgb, 150), PassSpec(0.0, rgb, 255, line=False)]
    def width_var(self, ctx):
        return 0.88 + 0.24 * math.sin(ctx.dist * 0.11 + 2.0)
    def segment(self, ctx, p1, p2, w):
        prims = super().segment(ctx, p1, p2, w)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        rng = ctx.rng
        for _ in range(max(2, int(w * 1.0))):              # waxy flecks, some outside the edge
            t, o = rng.random(), rng.uniform(-0.95, 0.95) * w * 0.9
            prims.append(("D", 1, (p1[0] + dx * t + nx * o, p1[1] + dy * t + ny * o),
                          rng.choice((1, 1, 2, 2, 3 if w > 8 else 2)),
                          _scl(ctx.rgb, rng.uniform(0.72, 1.18))))
        return prims


class Pen(Brush):
    key, name = "pen", "Pen"
    def passes(self, rgb):
        return [PassSpec(1.3, rgb, 70), PassSpec(1.0, rgb, 255)]       # crisp core, 1px soft rim


class Neon(Brush):
    key, name = "neon", "Neon"
    def passes(self, rgb):
        s = _saturate(rgb)
        return [PassSpec(3.6, s, 36), PassSpec(2.3, s, 85),
                PassSpec(1.3, s, 255), PassSpec(0.45, _mix(s, (255, 255, 255), 0.92), 255)]


class Fire(Brush):
    key, name = "fire", "Fire"
    emits, amb_rate = True, 26.0
    def passes(self, rgb):
        hot, mid = _mix(rgb, (255, 235, 160), 0.7), _mix(rgb, (255, 110, 20), 0.55)
        return [PassSpec(2.3, mid, 45), PassSpec(1.2, mid, 190), PassSpec(0.5, hot, 255)]
    def emit(self, fx, p, rgb, w, rng, now):
        hot, ember = _mix(rgb, (255, 235, 150), 0.65), _mix(rgb, (150, 25, 0), 0.6)
        k = 0.8 + w / 16.0
        for _ in range(2):                                              # flame body: big soft additive blobs
            fx.spawn(x=p[0] + rng.gauss(0, w * 0.6), y=p[1] + rng.gauss(0, w * 0.3),
                     vx=rng.gauss(0, 18), vy=-rng.uniform(50, 140), ay=-30,
                     size=rng.uniform(5.0, 9.0) * k, size_end=1.0,
                     life=rng.uniform(0.5, 1.1), c0=hot, c1=ember, a0=0.8, turb=60, kind="glow")
        if rng.random() < 0.5:                                          # embers
            fx.spawn(x=p[0] + rng.gauss(0, w), y=p[1], vx=rng.gauss(0, 30), vy=-rng.uniform(90, 200),
                     ay=-20, size=rng.uniform(1.5, 2.5), size_end=0.5, life=rng.uniform(0.6, 1.2),
                     c0=(255, 230, 150), c1=(255, 90, 20), a0=1.0, turb=80, kind="glow")


class Ice(Brush):
    key, name = "ice", "Ice"
    emits, amb_rate = True, 12.0
    def passes(self, rgb):
        return [PassSpec(2.6, (110, 190, 255), 45),
                PassSpec(1.1, _mix(rgb, (150, 215, 255), 0.75), 210),
                PassSpec(0.4, (235, 250, 255), 255)]
    def emit(self, fx, p, rgb, w, rng, now):
        if rng.random() < 0.55:                                          # crystal shards
            fx.spawn(x=p[0] + rng.gauss(0, w), y=p[1] + rng.gauss(0, w), vx=rng.gauss(0, 14),
                     vy=rng.uniform(10, 45), ay=18, size=rng.uniform(1.4, 3.0), size_end=0.5,
                     life=rng.uniform(0.8, 1.7), c0=(170, 225, 255), c1=(90, 160, 240),
                     rot=rng.uniform(0, 6.28), vrot=rng.uniform(-4, 4), kind="shard")
        fx.spawn(x=p[0] + rng.gauss(0, w * 1.4), y=p[1] + rng.gauss(0, w), vx=rng.gauss(0, 10),   # snow
                 vy=rng.uniform(14, 40), size=rng.uniform(1.4, 2.6), size_end=1.0,
                 life=rng.uniform(1.0, 2.0), c0=(245, 250, 255), c1=(200, 225, 255),
                 a0=0.9, turb=18, kind="soft")
        if rng.random() < 0.25:                                          # cold glow
            fx.spawn(x=p[0], y=p[1], vx=0, vy=-8, size=rng.uniform(6, 10), size_end=2,
                     life=0.6, c0=(60, 130, 220), c1=(10, 30, 80), a0=0.5, kind="glow")


class Lightning(Brush):
    key, name = "lightning", "Lightning"
    animated, anim_hz = True, 12.0
    emits, amb_rate = True, 16.0
    def passes(self, rgb):
        return [PassSpec(3.0, _mix(rgb, (120, 150, 255), 0.6), 45),
                PassSpec(1.0, _mix(rgb, (190, 215, 255), 0.6), 255),
                PassSpec(0.4, (255, 255, 255), 255)]
    @staticmethod
    def _bolt(ctx, a, b, w, wk):
        return [("L", i, a, b, max(1, int(w * ps.wm * wk))) for i, ps in enumerate(ctx.passes)]
    def segment(self, ctx, p1, p2, w):
        if ctx.aj is None:
            ctx.aj, ctx.ap, ctx.ad = p1, p1, 0.0
        if ctx.dist - ctx.ad < max(11.0, w * 1.6):
            return []
        dx, dy = p2[0] - ctx.ap[0], p2[1] - ctx.ap[1]
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        amp = max(4.0, w * 0.9)
        off = max(-2.2 * amp, min(2.2 * amp, ctx.rng.gauss(0, amp)))
        q = (p2[0] + nx * off, p2[1] + ny * off)
        prims = self._bolt(ctx, ctx.aj, q, w, 1.0)
        if ctx.rng.random() < 0.22:                                      # random short branch
            ang = math.atan2(dy, dx) + ctx.rng.choice((-1, 1)) * ctx.rng.uniform(0.5, 1.2)
            pos = q
            for _ in range(ctx.rng.randint(2, 4)):
                ang += ctx.rng.uniform(-0.5, 0.5)
                ln = ctx.rng.uniform(6, 13)
                nxt = (pos[0] + math.cos(ang) * ln, pos[1] + math.sin(ang) * ln)
                prims += self._bolt(ctx, pos, nxt, w, 0.55)
                pos = nxt
        ctx.aj, ctx.ap, ctx.ad = q, p2, ctx.dist
        return prims
    def emit(self, fx, p, rgb, w, rng, now):
        for _ in range(1 + (rng.random() < 0.5)):
            fx.spawn(x=p[0] + rng.gauss(0, w), y=p[1] + rng.gauss(0, w),
                     vx=rng.gauss(0, 150), vy=rng.gauss(0, 150), drag=4.0,
                     size=rng.uniform(1.5, 3.2), size_end=0.4, life=rng.uniform(0.12, 0.3),
                     c0=(235, 245, 255), c1=_mix(rgb, (80, 120, 255), 0.6), a0=1.0, kind="glow")


class ParticleBrush(Brush):
    key, name = "particle", "Particle"
    emits, amb_rate = True, 18.0
    def passes(self, rgb):
        return [PassSpec(1.7, rgb, 35), PassSpec(0.6, rgb, 130)]
    def emit(self, fx, p, rgb, w, rng, now):
        for _ in range(3):
            a, s = rng.uniform(0, math.tau), rng.uniform(30, 110)
            col = _hue_shift(rgb, rng.uniform(-0.07, 0.07))
            fx.spawn(x=p[0], y=p[1], vx=math.cos(a) * s, vy=math.sin(a) * s,
                     size=rng.uniform(2.0, 4.2), size_end=0.5, life=rng.uniform(0.6, 1.4),
                     c0=col, c1=_scl(col, 0.3), a0=0.95, drag=0.8, kind="glow")


class Galaxy(Brush):
    key, name = "galaxy", "Galaxy"
    emits, amb_rate = True, 14.0
    STAR = ((255, 255, 255), (170, 220, 255), (220, 180, 255))
    def passes(self, rgb):
        return [PassSpec(2.9, (110, 60, 220), 42),
                PassSpec(1.2, _mix(rgb, (70, 120, 255), 0.7), 120),
                PassSpec(0.0, (255, 255, 255), 255, line=False)]
    def segment(self, ctx, p1, p2, w):
        prims = super().segment(ctx, p1, p2, w)
        if ctx.rng.random() < 0.4:
            o = ctx.rng.uniform(-1.7, 1.7) * w
            prims.append(("D", 2, (p2[0] + ctx.rng.uniform(-1, 1) * w, p2[1] + o), ctx.rng.choice((1, 1, 2)),
                          ctx.rng.choice(self.STAR)))
        return prims
    def emit(self, fx, p, rgb, w, rng, now):
        pal = ((180, 120, 255), (110, 170, 255), (90, 230, 255), (255, 255, 255))
        fx.spawn(x=p[0], y=p[1], orbit=(p[0], p[1], rng.uniform(w * 1.2, w * 3.0 + 6),
                                        rng.uniform(0, 6.28), rng.choice((-1, 1)) * rng.uniform(2.0, 5.0)),
                 size=rng.uniform(1.5, 2.6), size_end=0.6, life=rng.uniform(0.8, 1.8),
                 c0=rng.choice(pal), c1=(60, 40, 140), a0=0.95, kind="glow")
        if rng.random() < 0.5:
            fx.spawn(x=p[0] + rng.uniform(-2, 2) * w, y=p[1] + rng.uniform(-2, 2) * w,
                     vx=rng.gauss(0, 6), vy=rng.gauss(0, 6), size=rng.uniform(1.0, 2.0),
                     size_end=1.0, life=rng.uniform(1.0, 2.2), c0=(255, 255, 255), c1=(150, 170, 255),
                     a0=1.0, kind="star")
        if rng.random() < 0.15:                                          # nebula puff
            fx.spawn(x=p[0], y=p[1], vx=rng.gauss(0, 8), vy=rng.gauss(0, 8),
                     size=rng.uniform(9, 14), size_end=14, life=1.4,
                     c0=rng.choice(((90, 40, 170), (40, 70, 170), (30, 110, 150))),
                     c1=(10, 5, 40), a0=0.22, kind="glow")


class Water(Brush):
    key, name = "water", "Water"
    animated, anim_hz = True, 10.0
    emits, amb_rate = True, 8.0
    def passes(self, rgb):
        return [PassSpec(2.0, (70, 150, 255), 55),
                PassSpec(1.1, _mix(rgb, (70, 150, 255), 0.75), 150),
                PassSpec(0.4, (210, 240, 255), 210)]
    def segment(self, ctx, p1, p2, w):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dy) or 1.0
        off = max(1.5, w * 0.45) * math.sin(ctx.dist * 0.11 + ctx.phase)      # flowing wave
        q = (p2[0] - dy / L * off, p2[1] + dx / L * off)
        if ctx.wp is None:
            ctx.wp = p1
        prims = [("L", i, ctx.wp, q, max(1, int(w * ps.wm))) for i, ps in enumerate(ctx.passes)]
        ctx.wp = q
        return prims
    def emit(self, fx, p, rgb, w, rng, now):
        if rng.random() < 0.65:
            fx.spawn(x=p[0] + rng.gauss(0, w * 0.5), y=p[1], vx=rng.gauss(0, 25), vy=-rng.uniform(30, 110),
                     ay=520, size=rng.uniform(2.0, 3.6), size_end=1.5, life=rng.uniform(0.5, 0.9),
                     c0=(160, 215, 255), c1=(90, 160, 255), a0=0.85, kind="soft", death="splash")


class Smoke(Brush):
    key, name = "smoke", "Smoke"
    emits, amb_rate = True, 10.0
    def passes(self, rgb):
        g = _mix(rgb, (140, 140, 145), 0.6)
        return [PassSpec(2.2, g, 32), PassSpec(1.1, g, 70)]
    def emit(self, fx, p, rgb, w, rng, now):
        if rng.random() < 0.7:
            g = _mix(rgb, (170, 170, 175), 0.65)
            fx.spawn(x=p[0] + rng.gauss(0, w * 0.6), y=p[1], vx=rng.gauss(0, 8), vy=-rng.uniform(15, 45),
                     size=rng.uniform(4, 7), size_end=rng.uniform(14, 20), life=rng.uniform(1.2, 2.4),
                     c0=g, c1=_scl(g, 0.55), a0=0.34, drag=0.5, turb=25, kind="soft")


BRUSHES: Dict[str, Brush] = {b.key: b for b in
                             (Pencil(), Marker(), Crayon(), Pen(), Neon(), Fire(), Ice(),
                              Lightning(), ParticleBrush(), Galaxy(), Water(), Smoke())}


class StrokeLayers:
    """One opaque layer per brush pass. Each pass is drawn at 100% inside its own layer and the
    layer is composited with the pass alpha -> uniform translucency / glow, no blobs.
    Only the dirty rectangle is ever composited or cleared (a live stroke is small, the layer is the whole window)."""

    def __init__(self, size: Tuple[int, int], off: Tuple[float, float] = (0, 0), n: int = 4):
        self.size, self.off = size, off
        self.layers = [pygame.Surface(size, pygame.SRCALPHA) for _ in range(n)]
        self.used = [False] * n
        self._lp = [None] * n                     # last round-cap centre per layer
        self._lr = [0] * n                        # ... and its radius
        self._bb = None                           # dirty rectangle [x0, y0, x1, y1] in layer coordinates

    def _grow(self, x0, y0, x1, y1):
        b = self._bb
        if b is None:
            self._bb = [x0, y0, x1, y1]
        else:
            if x0 < b[0]: b[0] = x0
            if y0 < b[1]: b[1] = y0
            if x1 > b[2]: b[2] = x1
            if y1 > b[3]: b[3] = y1

    def _area(self):
        b = self._bb
        if b is None: return None
        w, h = self.size
        x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])
        return pygame.Rect(x0, y0, x1 - x0, y1 - y0) if (x1 > x0 and y1 > y0) else None

    def clear(self):
        area = self._area()
        for i, L in enumerate(self.layers):
            if self.used[i]:
                if area is not None: L.fill((0, 0, 0, 0), area)
                else: L.fill((0, 0, 0, 0))
                self.used[i] = False
                self._lp[i], self._lr[i] = None, 0
        self._bb = None

    def draw(self, prims, passes: List[PassSpec]):
        ox, oy = self.off
        for pr in prims:
            pi = pr[1]
            if pi >= len(passes) or pi >= len(self.layers):
                continue
            L, ps = self.layers[pi], passes[pi]
            self.used[pi] = True
            if pr[0] == "L":
                a = (int(pr[2][0] - ox), int(pr[2][1] - oy))
                b = (int(pr[3][0] - ox), int(pr[3][1] - oy))
                w = pr[4]
                col = (*ps.rgb, 255)
                pygame.draw.line(L, col, a, b, w)
                m = w // 2 + 2
                self._grow(min(a[0], b[0]) - m, min(a[1], b[1]) - m, max(a[0], b[0]) + m + 1, max(a[1], b[1]) + m + 1)
                if w > 2:
                    r = w // 2
                    if not (self._lp[pi] == a and self._lr[pi] >= r):      # same centre, not bigger => already covered
                        pygame.draw.circle(L, col, a, r)
                    pygame.draw.circle(L, col, b, r)
                    self._lp[pi], self._lr[pi] = b, r
            else:
                c = pr[4] or ps.rgb
                ctr = (int(pr[2][0] - ox), int(pr[2][1] - oy)); rad = max(1, int(pr[3]))
                pygame.draw.circle(L, (*c, 255), ctr, rad)
                self._grow(ctr[0] - rad - 2, ctr[1] - rad - 2, ctr[0] + rad + 3, ctr[1] + rad + 3)

    def render(self, dest: pygame.Surface, passes: List[PassSpec], at=(0, 0)):
        area = self._area()
        if area is None: return
        pos = (at[0] + area.x, at[1] + area.y)
        for i, L in enumerate(self.layers):
            if self.used[i] and i < len(passes):
                L.set_alpha(passes[i].alpha)
                dest.blit(L, pos, area)

    flatten_into = render


@dataclass
class FXParticle:
    x: float; y: float
    vx: float = 0.0; vy: float = 0.0
    ax: float = 0.0; ay: float = 0.0
    size: float = 3.0; size_end: float = 1.0
    life: float = 1.0; max_life: float = 1.0
    c0: Tuple[int, int, int] = (255, 255, 255)
    c1: Tuple[int, int, int] = (255, 255, 255)
    a0: float = 1.0
    drag: float = 0.0
    turb: float = 0.0
    kind: str = "glow"                      # glow (additive) | soft (alpha) | shard | star
    rot: float = 0.0; vrot: float = 0.0
    orbit: Optional[Tuple[float, float, float, float, float]] = None   # cx, cy, r, ang, omega
    death: str = ""
    seed: float = 0.0
    pts: Optional[List[Tuple[float, float]]] = None      # polyline for kind='bolt'


_GLOW_CACHE: Dict[Tuple, pygame.Surface] = {}
_SOFT_CACHE: Dict[Tuple, pygame.Surface] = {}


def _glow_sprite(rad: int, col: Tuple[int, int, int]) -> pygame.Surface:
    k = (rad, col)
    s = _GLOW_CACHE.get(k)
    if s is None:
        if len(_GLOW_CACHE) > 2500: _GLOW_CACHE.clear()
        s = pygame.Surface((rad * 2 + 1, rad * 2 + 1))
        for i in range(rad, 0, -1):
            f = ((rad - i + 1) / rad) ** 1.6
            pygame.draw.circle(s, _scl(col, f), (rad, rad), i)
        _GLOW_CACHE[k] = s
    return s


def _soft_sprite(rad: int, col: Tuple[int, int, int]) -> pygame.Surface:
    k = (rad, col)
    s = _SOFT_CACHE.get(k)
    if s is None:
        if len(_SOFT_CACHE) > 1500: _SOFT_CACHE.clear()
        s = pygame.Surface((rad * 2 + 1, rad * 2 + 1), pygame.SRCALPHA)
        for i in range(rad, 0, -1):
            pygame.draw.circle(s, (*col, int(255 * ((rad - i + 1) / rad) ** 1.2)), (rad, rad), i)
        _SOFT_CACHE[k] = s
    return s


_BUBBLE_CACHE: Dict[int, pygame.Surface] = {}


def _bubble_sprite(rad: int) -> pygame.Surface:
    s = _BUBBLE_CACHE.get(rad)
    if s is None:
        s = pygame.Surface((rad * 2 + 1, rad * 2 + 1), pygame.SRCALPHA)
        pygame.draw.circle(s, (190, 225, 255, 45), (rad, rad), rad)
        pygame.draw.circle(s, (200, 235, 255, 210), (rad, rad), rad, 1)
        pygame.draw.circle(s, (255, 255, 255, 230), (rad - rad // 3, rad - rad // 3), max(1, rad // 4))
        _BUBBLE_CACHE[rad] = s
    return s


class FXSystem:
    """Capped particle engine used by the brushes (Phase 4 builds the effect modes on this)."""

    def __init__(self, max_n: int = FX_MAX):
        self.max_n = max_n
        self.p: List[FXParticle] = []
        self._t = 0.0
        self.alpha_mul = 1.0                       # layer opacity (Effects)

    def __len__(self): return len(self.p)

    def clear(self): self.p.clear()

    def spawn(self, **kw) -> bool:
        if len(self.p) >= self.max_n:
            return False
        life = kw.pop("life", 1.0)
        part = FXParticle(life=life, max_life=life, seed=random.random() * 6.28, **kw)
        self.p.append(part)
        return True

    def update(self, dt: float, now: Optional[float] = None):
        self._t += dt
        t = self._t
        alive: List[FXParticle] = []
        for q in self.p:
            q.life -= dt
            if q.life <= 0:
                if q.death == "splash":
                    for _ in range(3):
                        self.spawn(x=q.x, y=q.y, vx=random.gauss(0, 40), vy=-random.uniform(10, 50),
                                   ay=400, size=1.5, size_end=0.8, life=random.uniform(0.25, 0.4),
                                   c0=q.c0, c1=q.c1, a0=0.8, kind="soft")
                continue
            if q.orbit is not None:
                cx, cy, r, ang, om = q.orbit
                ang += om * dt
                q.orbit = (cx, cy, r, ang, om)
                q.x, q.y = cx + math.cos(ang) * r, cy + math.sin(ang) * r
            else:
                if q.turb:
                    q.vx += q.turb * math.sin(t * 6.0 + q.y * 0.04 + q.seed) * dt * 3.0
                q.vx += q.ax * dt
                q.vy += q.ay * dt
                if q.drag:
                    k = max(0.0, 1.0 - q.drag * dt)
                    q.vx *= k; q.vy *= k
                q.x += q.vx * dt
                q.y += q.vy * dt
            q.rot += q.vrot * dt
            alive.append(q)
        self.p = alive

    def render(self, surface: pygame.Surface):
        t = self._t
        W, H = surface.get_size()
        batch: list = []
        fb = getattr(surface, "fblits", None)

        def flush():
            if batch:
                if fb is not None:
                    fb(batch, pygame.BLEND_RGB_ADD)
                else:
                    for spr, pos in batch: surface.blit(spr, pos, special_flags=pygame.BLEND_RGB_ADD)
                batch.clear()

        for q in self.p:
            u = q.life / q.max_life                     # 1 -> 0
            x, y = int(q.x), int(q.y)
            if x < -30 or y < -30 or x > W + 30 or y > H + 30:
                continue
            size = q.size_end + (q.size - q.size_end) * u
            col = _mix(q.c1, q.c0, u)
            f = self.alpha_mul * q.a0 * (u ** 0.7 if q.kind != "soft" else math.sin(min(1.0, (1 - u) * 6 + 0.1) * 1.5708) * u)
            if q.kind == "glow" or q.kind == "star":
                if q.kind == "star":
                    f *= 0.55 + 0.45 * math.sin(t * 9 + q.seed * 3)
                c = (int(col[0] * f) // 24 * 24, int(col[1] * f) // 24 * 24, int(col[2] * f) // 24 * 24)
                if max(c) < 24: continue
                rad = max(1, min(18, int(size)))
                batch.append((_glow_sprite(rad, c), (x - rad, y - rad)))     # drawn in order, many blits per call
            elif q.kind == "soft":
                flush()
                rad = max(1, min(24, int(size)))
                c = (col[0] // 32 * 32 + 16, col[1] // 32 * 32 + 16, col[2] // 32 * 32 + 16)
                s = _soft_sprite(rad, c)
                s.set_alpha(max(0, min(255, int(255 * f))))
                surface.blit(s, (x - rad, y - rad))
            elif q.kind == "spark":
                flush()
                sp = math.hypot(q.vx, q.vy) or 1.0
                ln = min(16.0, 3.0 + sp * 0.045) * max(0.25, f)            # fade by SHRINKING, never by darkening
                if f > 0.12:
                    pygame.draw.line(surface, (int(col[0]), int(col[1]), int(col[2])), (x, y),
                                     (int(q.x - q.vx / sp * ln), int(q.y - q.vy / sp * ln)), 2 if u > 0.5 else 1)
            elif q.kind == "confetti":
                flush()
                wd = max(0.25, abs(math.cos(t * 6.0 + q.seed))) * size * 1.8
                ht = size * 0.9 * (1.0 if u > 0.2 else u / 0.2)
                ca, sa = math.cos(q.rot), math.sin(q.rot)
                pygame.draw.polygon(surface, q.c0, [(x + ca * px - sa * py, y + sa * px + ca * py)
                                                    for px, py in ((-wd, -ht), (wd, -ht), (wd, ht), (-wd, ht))])
            elif q.kind == "bubble":
                flush()
                rad = max(3, min(14, int(size)))
                s = _bubble_sprite(rad)
                s.set_alpha(max(0, min(255, int(255 * f))))
                surface.blit(s, (x - rad, y - rad))
            elif q.kind == "bolt":
                flush()
                if q.pts and len(q.pts) >= 2 and int(t * 40 + q.seed) % 3:   # flicker (needs 2+ points)
                    cc = (int(col[0] * f), int(col[1] * f), int(col[2] * f))
                    ip = [(int(a), int(b)) for a, b in q.pts]
                    pygame.draw.lines(surface, cc, False, ip, 4)
                    pygame.draw.lines(surface, (255, 255, 255), False, ip, 1)
            else:                                       # shard: rotating crystal
                flush()
                sz = max(1.0, size * (1.0 if u > 0.3 else u / 0.3))
                ca, sa = math.cos(q.rot), math.sin(q.rot)
                pts = [(x + ca * px - sa * py, y + sa * px + ca * py)
                       for px, py in ((0, -sz * 1.7), (sz * 0.7, 0), (0, sz * 1.7), (-sz * 0.7, 0))]
                pygame.draw.polygon(surface, col, pts)
                pygame.draw.polygon(surface, (255, 255, 255), pts, 1)
        flush()


def build_stroke_sprite(brush: Brush, pts, rgb, w: int, seed: int, phase: float = 0.0):
    """Render a stroke (local coordinates) into an SRCALPHA sprite. Returns (surface, (ox, oy))."""
    pad = int(w * 3.8) + 28
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ox, oy = int(math.floor(min(xs))) - pad, int(math.floor(min(ys))) - pad
    bw = min(2600, int(math.ceil(max(xs))) + pad - ox + 1)
    bh = min(2600, int(math.ceil(max(ys))) + pad - oy + 1)
    ctx = brush.new_ctx(rgb, w, pts[0], seed, phase)
    lay = StrokeLayers((bw, bh), (ox, oy), n=len(ctx.passes))
    for p in pts[1:]:
        lay.draw(brush.feed(ctx, p, w), ctx.passes)
    spr = pygame.Surface((bw, bh), pygame.SRCALPHA)
    lay.flatten_into(spr, ctx.passes)
    return spr, (ox, oy)


_REBUILD_LEFT = [SPRITE_REBUILD_BUDGET]
_PREVIEW_CACHE: Dict[Tuple, pygame.Surface] = {}


def brush_preview(key: str, rgb, size=(92, 22)) -> pygame.Surface:
    if (rgb[0] * 3 + rgb[1] * 6 + rgb[2]) < 600:            # keep dark inks visible on the dark panel
        rgb = (205, 205, 215)
    ck = (key, rgb, size)
    s = _PREVIEW_CACHE.get(ck)
    if s is None:
        sw, sh = size
        pts = [(8 + i * (sw - 16) / 40.0, sh / 2 + 5.5 * math.sin(i / 40.0 * 7.5)) for i in range(41)]
        spr, (ox, oy) = build_stroke_sprite(BRUSHES[key], pts, rgb, 4, 3, 0.0)
        out = pygame.Surface(size, pygame.SRCALPHA)
        out.blit(spr, (ox, oy))                   # sprite origin is bbox-minus-pad (negative)
        _PREVIEW_CACHE[ck] = s = out
    return s


class BrushPanel:
    """Futuristic brush menu. Hover a fingertip over the tab to open, over a row to select (dwell),
    or use  B / TAB / F1-F12 / mouse."""
    TAB = pygame.Rect(974, 64, 100, 32)
    ROW, PAD, HEAD = 32, 8, 6
    PANEL = pygame.Rect(974, 102, 240, 12 * 32 + 8 * 2 + 6)

    def __init__(self):
        pygame.font.init()
        self._f = pygame.font.SysFont("monospace", 14, bold=True)
        self._s = pygame.font.SysFont("monospace", 11)
        self.open = False
        self._hk: Optional[str] = None
        self._latched: Optional[str] = None
        self._ht = 0.0
        self._idle = 0.0
        self._rows: Dict[str, pygame.Rect] = {}
        y = self.PANEL.y + self.PAD
        for k in BRUSH_ORDER:
            self._rows[k] = pygame.Rect(self.PANEL.x + 6, y, self.PANEL.w - 12, self.ROW - 3)
            y += self.ROW

    def toggle(self):
        self.open = not self.open
        self._hk, self._ht, self._idle = None, 0.0, 0.0
        self._latched = "_tab" if self._latched == "_tab" else None

    def blocks(self, pt) -> bool:
        """True when a drawing fingertip is over the menu (so it must not leave ink)."""
        return bool(pt) and (self.TAB.collidepoint(pt) or (self.open and self.PANEL.collidepoint(pt)))

    def _hit(self, pt) -> Optional[str]:
        if not pt: return None
        if self.TAB.collidepoint(pt): return "_tab"
        if self.open:
            for k, r in self._rows.items():
                if r.collidepoint(pt): return k
        return None

    def update(self, tip, dt: float) -> Optional[str]:
        hov = self._hit(tip)
        fired = None
        if hov != self._latched:
            self._latched = None                      # finger left the thing that just fired
        if hov and hov == self._hk and hov != self._latched:
            self._ht += dt
            if self._ht >= UI_HOVER_S:
                self._ht = 0.0
                self._latched = hov                   # fire once; must leave before it can fire again
                if hov == "_tab":
                    self.toggle()
                else:
                    fired = hov
                    self.open = False
        elif hov != self._hk:
            self._hk, self._ht = hov, 0.0
        if self.open and not hov:
            self._idle += dt
            if self._idle > 6.0: self.open = False
        else:
            self._idle = 0.0
        return fired

    def click(self, pos) -> Optional[str]:
        hit = self._hit(pos)
        if hit == "_tab":
            self.toggle()
        elif hit:
            self.open = False
            return hit
        return None

    def render(self, surface: pygame.Surface, active: str, rgb):
        CY = (60, 220, 255)
        t = pygame.Surface(self.TAB.size, pygame.SRCALPHA)
        pygame.draw.rect(t, (8, 14, 24, 200), t.get_rect(), border_radius=10)
        pygame.draw.rect(t, (*CY, 230), t.get_rect(), 2, border_radius=10)
        if self._hk == "_tab":
            pygame.draw.rect(t, (*CY, 110), (0, 0, int(self.TAB.w * min(1, self._ht / UI_HOVER_S)), self.TAB.h), border_radius=10)
        surface.blit(t, self.TAB.topleft)
        lb = self._f.render(f"BRUSH: {BRUSHES[active].name.upper()}", True, (225, 245, 255))
        surface.blit(lb, (self.TAB.x + 12, self.TAB.y + (self.TAB.h - lb.get_height()) // 2))
        if not self.open: return
        p = pygame.Surface(self.PANEL.size, pygame.SRCALPHA)
        pygame.draw.rect(p, (8, 14, 24, 205), p.get_rect(), border_radius=12)
        pygame.draw.rect(p, (*CY, 200), p.get_rect(), 2, border_radius=12)
        surface.blit(p, self.PANEL.topleft)
        for i, k in enumerate(BRUSH_ORDER):
            r = self._rows[k]
            row = pygame.Surface(r.size, pygame.SRCALPHA)
            if k == active:
                pygame.draw.rect(row, (*CY, 70), row.get_rect(), border_radius=8)
                pygame.draw.rect(row, (*CY, 255), row.get_rect(), 2, border_radius=8)
            if self._hk == k:
                pygame.draw.rect(row, (255, 255, 255, 90),
                                 (0, 0, int(r.w * min(1, self._ht / UI_HOVER_S)), r.h), border_radius=8)
            surface.blit(row, r.topleft)
            surface.blit(brush_preview(k, rgb), (r.x + 4, r.y + 2))
            nm = self._f.render(BRUSHES[k].name, True, (235, 245, 255))
            surface.blit(nm, (r.x + 100, r.y + 3))
            fk = self._s.render(f"F{i + 1}", True, (140, 170, 190))
            surface.blit(fk, (r.right - fk.get_width() - 4, r.y + 15))


_LABEL_FONT = None


def _limit_body_speed(body, gravity, damping, dt):
    gs = getattr(body, "gscale", 1.0)                                  # per-body "gravity response" (Phase 5)
    if gs != 1.0: gravity = (gravity[0] * gs, gravity[1] * gs)
    pymunk.Body.update_velocity(body, gravity, damping, dt)
    v = body.velocity
    sp = v.length
    if sp > MAX_BODY_SPEED:
        body.velocity = v * (MAX_BODY_SPEED / sp)


class PhysicsWorld:
    def __init__(self, w: int, h: int):
        self.W, self.H = w, h
        self.space        = pymunk.Space()
        self.space.gravity = GRAVITY_PRESETS[DEFAULT_GRAVITY]
        self.space.damping = 0.995
        self.bodies: List[PhysBody] = []
        self._grav_idx = DEFAULT_GRAVITY
        self._wind = 0.0
        self._mb: Optional[pymunk.Body] = None
        self._dj: Optional[pymunk.PivotJoint] = None
        self._gj: Optional[pymunk.GearJoint] = None
        self._sel: Optional[PhysBody] = None
        self._next_uid = 1
        self.alpha_mul = 1.0                       # layer opacity (Physics Objects)
        self._build_walls()

    def _build_walls(self):
        sb = self.space.static_body
        o = WALL_THICK - WALL_W                      # push the segments outward so the inner faces stay put
        for a, b in [
            ((-o,          self.H + o), (self.W + o, self.H + o)),
            ((-o,          -o        ), (-o,         self.H + o)),
            ((self.W + o,  -o        ), (self.W + o, self.H + o)),
            ((-o,          -o        ), (self.W + o, -o        )),
        ]:
            seg = pymunk.Segment(sb, a, b, WALL_THICK)
            seg.elasticity = ELASTICITY
            seg.friction   = FRICTION
            self.space.add(seg)
    def add_stroke(
        self,
        pts:   List[Tuple[int, int]],
        color: Tuple[int, int, int],
        width: int,
        brush: str = DEFAULT_BRUSH,
    ) -> Tuple[float, float]:
        if len(pts) < 2:
            return (0.0, 0.0)
        arr    = np.asarray(pts, dtype=float)
        cx, cy = arr[:, 0].mean(), arr[:, 1].mean()

        moment = pymunk.moment_for_segment(
            BODY_MASS,
            (arr[0,  0] - cx, arr[0,  1] - cy),
            (arr[-1, 0] - cx, arr[-1, 1] - cy),
            SEG_RADIUS,
        )
        body = pymunk.Body(BODY_MASS, max(moment, 1.0))
        body.velocity_func = _limit_body_speed
        body.position = (cx, cy)
        body.angular_velocity = random.uniform(-2.2, 2.2)

        shapes: List[pymunk.Shape] = []
        n = len(pts) - 1
        for i in range(n):
            a = (pts[i][0]   - cx, pts[i][1]   - cy)
            b = (pts[i+1][0] - cx, pts[i+1][1] - cy)
            seg = pymunk.Segment(body, a, b, SEG_RADIUS)
            seg.mass = BODY_MASS / n
            seg.elasticity = ELASTICITY
            seg.friction = FRICTION
            shapes.append(seg)

        self.space.add(body, *shapes)
        self.bodies.append(PhysBody(body, shapes, color, width, uid=self._next_uid, brush=brush))
        self._next_uid += 1
        return (cx, cy)
    def cycle_gravity(self) -> str:
        self._grav_idx = (self._grav_idx + 1) % len(GRAVITY_PRESETS)
        self.space.gravity = GRAVITY_PRESETS[self._grav_idx]
        return GRAVITY_LABELS[self._grav_idx]

    @property
    def gravity_idx(self) -> int:
        return self._grav_idx

    #wind
    def set_wind(self, w: float):
        self._wind = w

    #explosion
    def explode(self, cx: float, cy: float):
        """Outward impulse proportional to 1/distance from (cx, cy)."""
        for pb in self.bodies:
            if pb.frozen: continue
            bx, by = pb.body.position
            dx, dy = bx - cx, by - cy
            d = max(math.hypot(dx, dy), 1.0)
            m = EXPLODE_FORCE / d
            pb.body.apply_impulse_at_world_point((dx/d*m, dy/d*m), (bx, by))

    def push_near(self, tip: Tuple[int, int]):
        """Apply a single outward push impulse to the nearest body."""
        pb = self.find_nearest(tip, SELECT_RADIUS * 1.5)
        if pb is None or pb.frozen: return
        bx, by = pb.body.position
        dx, dy = bx - tip[0], by - tip[1]
        d = max(math.hypot(dx, dy), 1.0)
        pb.body.apply_impulse_at_world_point(
            (dx / d * PUSH_FORCE, dy / d * PUSH_FORCE), (bx, by))

    def find_nearest(
        self,
        tip: Tuple[int, int],
        radius: float = SELECT_RADIUS,
    ) -> Optional[PhysBody]:
        """Nearest body by distance to its *stroke surface* (not its centroid),
        so you can pick a circle by touching its outline."""
        p = pymunk.Vec2d(float(tip[0]), float(tip[1]))
        best, bd = None, radius
        for pb in self.bodies:
            if not pb.visible: continue                         # hidden drawings can't be picked
            for sh in pb.shapes:
                d = max(sh.point_query(p).distance, 0.0)
                if d < bd:
                    best, bd = pb, d
        return best

    def set_hover(self, pb: Optional[PhysBody]):
        for b in self.bodies:
            b.hover = (b is pb)

    def object_bounds(self, pb: PhysBody) -> Tuple[float, float, float]:
        pts = []
        for sh in pb.shapes:
            if isinstance(sh, pymunk.Segment):
                for q in (sh.a, sh.b):
                    w = pb.body.local_to_world(q)
                    pts.append((w.x, w.y))
        if not pts:
            bx, by = pb.body.position
            return bx, by, 30.0
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        r = max(math.hypot(x - cx, y - cy) for x, y in pts) + SEG_RADIUS + 12
        return cx, cy, r

    def start_drag(self, tip: Tuple[int, int],
                   radius: float = SELECT_RADIUS) -> Optional[PhysBody]:
        if self._dj: return None                      # already holding something
        pb = self.find_nearest(tip, radius)
        if pb is None or pb.frozen: return None
        pb.selected = True
        pb.body.velocity = (pb.body.velocity[0] * 0.25, pb.body.velocity[1] * 0.25)
        pb.body.angular_velocity *= 0.25
        self._sel = pb
        mb = pymunk.Body(body_type=pymunk.Body.STATIC)
        mb.position = tip
        j = pymunk.PivotJoint(mb, pb.body, tip)
        j.max_force  = DRAG_MAX_FORCE
        j.error_bias = DRAG_ERR_BIAS
        self.space.add(mb, j)
        if ENABLE_GRAB_ROTATION:
            g = pymunk.GearJoint(mb, pb.body, pb.body.angle - mb.angle, 1.0)
            g.max_force = ROT_MAX_TORQUE
            g.error_bias = (1 - 0.12) ** 60
            self.space.add(g)
            self._gj = g
        self._mb = mb
        self._dj = j
        return pb

    def rotate_drag(self, angle: float):
        """Rotate the held body to (grab-start angle + angle)."""
        if self._mb is not None and self._gj is not None:
            self._mb.angle = angle

    def scale_body(self, pb: PhysBody, k: float):
        """Uniformly scale a stroke about its body origin (in place, cheap)."""
        for sh in pb.shapes:
            if isinstance(sh, pymunk.Segment):
                sh.unsafe_set_endpoints(sh.a * k, sh.b * k)
        pb.scale *= k
        if pb is self._sel and self._dj is not None:
            self._dj.anchor_b = self._dj.anchor_b * k   # keep the pinched point under the fingers

    def radial_force(self, cx: float, cy: float, sign: int, strength: float,
                     dt: float, exclude: Optional[PhysBody] = None):
        """sign=+1 push everything away from (cx,cy), -1 pull toward it."""
        for pb in self.bodies:
            if pb is exclude or pb.frozen: continue
            bx, by = pb.body.position
            dx, dy = bx - cx, by - cy
            d = max(math.hypot(dx, dy), 1.0)
            if d > PUSHPULL_RADIUS or (sign < 0 and d < 45): continue
            imp = PUSHPULL_FORCE * strength * math.sqrt(1 - d / PUSHPULL_RADIUS) * dt * sign
            pb.body.apply_impulse_at_world_point((dx / d * imp, dy / d * imp), (bx, by))

    def update_drag(self, tip: Tuple[int, int]):
        if self._mb:
            self._mb.position = tip

    def end_drag(self, throw_vel: Tuple[float, float] = (0.0, 0.0), spin: float = 0.0):
        """Release the grabbed body.
           If throw_vel magnitude > 50 px/s, apply it directly to the body so
           the throw feels natural and physically accurate."""
        if self._gj:
            self.space.remove(self._gj)
        if self._dj:
            self.space.remove(self._dj, self._mb)
        if self._sel:
            if spin:
                self._sel.body.angular_velocity = max(-MAX_SPIN, min(MAX_SPIN, spin))
            vx, vy = throw_vel
            sp = math.hypot(vx, vy)
            if sp > MAX_THROW_SPEED:
                vx, vy = vx * MAX_THROW_SPEED / sp, vy * MAX_THROW_SPEED / sp
            if abs(vx) + abs(vy) > 50:
                self._sel.body.velocity = (vx, vy)
            self._sel.selected = False
        self._mb = self._dj = self._gj = self._sel = None

    # ── Phase 5: edit helpers ──
    def apply_props(self, pb: PhysBody):
        """Push the editable physics properties (mass / bounce / friction / gravity) onto the pymunk objects."""
        n = max(1, len(pb.shapes))
        for sh in pb.shapes:
            sh.mass = BODY_MASS * pb.mass_mult / n
            sh.elasticity = pb.elast
            sh.friction = pb.fric
        pb.body.gscale = pb.gscale

    def set_frozen(self, pb: PhysBody, frozen: bool):
        pb.frozen = frozen
        if frozen:
            pb.body.velocity = (0.0, 0.0); pb.body.angular_velocity = 0.0
        pb.body.body_type = pymunk.Body.STATIC if frozen else pymunk.Body.DYNAMIC
        if not frozen: self.apply_props(pb)

    def remove_body(self, pb: PhysBody):
        if pb is self._sel: self.end_drag()
        if pb in self.bodies:
            self.space.remove(pb.body, *pb.shapes)
            self.bodies.remove(pb)

    def freeze_selected(self):
        if not self._sel: return
        self._sel.frozen = not self._sel.frozen
        bt = pymunk.Body.STATIC if self._sel.frozen else pymunk.Body.DYNAMIC
        self._sel.body.body_type = bt
        if not self._sel.frozen:
            self._sel.body.mass = BODY_MASS

    def delete_selected(self):
        if not self._sel: return
        self.space.remove(self._sel.body, *self._sel.shapes)
        self.bodies.remove(self._sel)
        if self._gj:
            self.space.remove(self._gj)
        if self._dj:
            self.space.remove(self._dj, self._mb)
        self._mb = self._dj = self._gj = self._sel = None

    def duplicate_selected(self):
        if not self._sel: return
        old  = self._sel
        pts: List[Tuple[int, int]] = []
        for sh in old.shapes:
            if isinstance(sh, pymunk.Segment):
                a = old.body.local_to_world(sh.a)
                b = old.body.local_to_world(sh.b)
                pts += [(int(a.x)+40, int(a.y)+40),
                        (int(b.x)+40, int(b.y)+40)]
        if pts:
            self.add_stroke(pts, old.color, old.width, old.brush)

    #step
    def step(self, dt: float):
        if self._wind != 0.0:
            for pb in self.bodies:
                if not pb.frozen:
                    pb.body.apply_force_at_local_point(
                        (self._wind * BODY_MASS, 0.0), (0.0, 0.0))
        n = max(PHYS_STEPS, min(PHYS_MAX_SUBSTEPS, int(math.ceil(dt * MAX_BODY_SPEED / PHYS_MAX_SUBSTEP_PX))))
        sub = dt / n
        for _ in range(n):
            self.space.step(sub)
        for pb in self.bodies:
            bx, by = pb.body.position
            if not (math.isfinite(bx) and math.isfinite(by)) or not (-150 < bx < self.W + 150 and -150 < by < self.H + 150):
                pb.body.position = (min(max(bx if math.isfinite(bx) else self.W / 2, 40), self.W - 40),
                                    min(max(by if math.isfinite(by) else self.H / 2, 40), self.H - 40))   # rescue: never lose a drawing
                pb.body.velocity = (0.0, 0.0)
                bx, by = pb.body.position
            pb.trail.append((int(bx), int(by)))

    def clear(self):
        self.end_drag()
        for pb in self.bodies:
            self.space.remove(pb.body, *pb.shapes)
        self.bodies.clear()

    def render(self, surface: pygame.Surface, trails: bool):
        _REBUILD_LEFT[0] = SPRITE_REBUILD_BUDGET
        for pb in self.bodies:
            if not pb.visible: continue
            # Optional fading centroid trail
            if trails and len(pb.trail) > 1:
                pts = list(pb.trail)
                pad = pb.width + 2                      # small surface around the trail (was full-screen per body)
                tx0 = min(p[0] for p in pts) - pad
                ty0 = min(p[1] for p in pts) - pad
                tw = max(p[0] for p in pts) + pad - tx0 + 1
                th = max(p[1] for p in pts) + pad - ty0 + 1
                if tw * th > 4_000_000:                  # corrupt / runaway trail: skip instead of allocating a monster surface
                    pb.trail.clear(); continue
                tmp = pygame.Surface((tw, th), pygame.SRCALPHA)
                for i in range(1, len(pts)):
                    a = int(TRAIL_ALPHA * i / len(pts))
                    pygame.draw.line(tmp, (*pb.color, a),
                                     (pts[i-1][0] - tx0, pts[i-1][1] - ty0),
                                     (pts[i][0] - tx0, pts[i][1] - ty0), max(pb.width-1, 1))
                surface.blit(tmp, (tx0, ty0))

            lw = max(pb.width, 2)
            if pb.selected or pb.frozen:                      # underlay (selection / frozen)
                ucol = (255, 255, 80) if pb.selected else (80, 140, 255)
                for sh in pb.shapes:
                    if not isinstance(sh, pymunk.Segment): continue
                    aw = pb.body.local_to_world(sh.a)
                    bw = pb.body.local_to_world(sh.b)
                    pygame.draw.line(surface, ucol, (int(aw.x), int(aw.y)), (int(bw.x), int(bw.y)),
                                     lw * 2 + (8 if pb.selected else 6))
            if not self._render_brush(surface, pb):
                for sh in pb.shapes:                          # safe fallback: plain line
                    if not isinstance(sh, pymunk.Segment): continue
                    aw = pb.body.local_to_world(sh.a)
                    bw = pb.body.local_to_world(sh.b)
                    ax, ay, bx, by = int(aw.x), int(aw.y), int(bw.x), int(bw.y)
                    pygame.draw.line(surface, pb.color, (ax, ay), (bx, by), lw * 2)
                    pygame.draw.circle(surface, pb.color, (ax, ay), lw)
                    pygame.draw.circle(surface, pb.color, (bx, by), lw)
            if pb.hover or pb.selected:
                self._render_highlight(surface, pb)

    def _render_brush(self, surface: pygame.Surface, pb: PhysBody) -> bool:
        """Blit the cached brush sprite of a launched stroke, rotated with the body.
        Static brushes are rendered once; animated ones (lightning, water) are refreshed at
        anim_hz, a couple per frame at most. Returns False -> caller draws the plain fallback."""
        try:
            br = BRUSHES.get(pb.brush) or BRUSHES[DEFAULT_BRUSH]
            now = time.perf_counter()
            fresh = pb.sprite is None or abs(pb.sprite_scale - pb.scale) > 1e-4
            if not fresh and br.animated and now - pb.sprite_t > 1.0 / br.anim_hz and _REBUILD_LEFT[0] > 0:
                fresh = True
                _REBUILD_LEFT[0] -= 1
            if fresh:
                segs = [sh for sh in pb.shapes if isinstance(sh, pymunk.Segment)]
                if not segs: return False
                pts = [(segs[0].a.x, segs[0].a.y)] + [(sh.b.x, sh.b.y) for sh in segs]
                w = max(1, int(pb.width * min(2.0, max(0.6, pb.scale ** 0.5))))
                seed = pb.uid * 7919 + (int(now * br.anim_hz) if br.animated else 0)
                pb.sprite, pb.sprite_origin = build_stroke_sprite(br, pts, pb.color, w, seed, now * 6.0)
                if pb.glow > 0:
                    pb.sprite, gp = add_glow(pb.sprite, pb.glow)
                    pb.sprite_origin = (pb.sprite_origin[0] - gp, pb.sprite_origin[1] - gp)
                pb.sprite_scale, pb.sprite_t, pb.rot_q = pb.scale, now, -9999
            spr = pb.sprite
            q = int(round(math.degrees(pb.body.angle) / 2.0))
            if q != pb.rot_q:
                pb.rot_surf = pygame.transform.rotate(spr, -q * 2.0)
                pb.rot_q = q
            ox, oy = pb.sprite_origin
            cx, cy = ox + spr.get_width() / 2.0, oy + spr.get_height() / 2.0
            ang = math.radians(q * 2.0)
            ca, sa = math.cos(ang), math.sin(ang)
            bx, by = pb.body.position
            wx, wy = bx + cx * ca - cy * sa, by + cx * sa + cy * ca
            rs = pb.rot_surf
            a = pb.opacity * self.alpha_mul
            rs.set_alpha(255 if a >= 0.999 else int(255 * max(0.05, a)))      # 255, NEVER None: None strips the per-pixel alpha
            surface.blit(rs, (int(wx - rs.get_width() / 2), int(wy - rs.get_height() / 2)))
            return True
        except Exception:
            return False

    def _render_highlight(self, surface: pygame.Surface, pb: PhysBody):
        """Pulsing dotted holo-ring + label around a selectable / grabbed body."""
        global _LABEL_FONT
        if _LABEL_FONT is None:
            _LABEL_FONT = pygame.font.SysFont("monospace", 15, bold=True)
        t = time.perf_counter()
        cx, cy, r = self.object_bounds(pb)
        r += 3 * math.sin(t * 5.0)
        col = (255, 235, 90) if pb.selected else (90, 220, 255)
        glow = pygame.Surface((int(r * 2 + 8), int(r * 2 + 8)), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*col, 28), (glow.get_width() // 2, glow.get_height() // 2), int(r))
        surface.blit(glow, (cx - glow.get_width() // 2, cy - glow.get_height() // 2))
        n = 30
        spin = t * (2.2 if pb.selected else 1.0)
        for k in range(n):
            a = spin + k * math.tau / n
            pygame.draw.circle(surface, col,
                               (int(cx + math.cos(a) * r), int(cy + math.sin(a) * r)),
                               3 if k % 3 == 0 else 2)
        txt = f"OBJECT SELECTED #{pb.uid}" if pb.selected else "GRAB"
        if pb.selected and pb.extra:
            txt += "  " + pb.extra
        img = _LABEL_FONT.render(txt, True, col)
        surface.blit(img, (int(cx - img.get_width() / 2), int(cy - r - 22)))

@dataclass
class Particle:
    x:float; y:float; vx:float; vy:float
    life:float; max_life:float
    color:Tuple[int,int,int]; radius:float


class ParticleSystem:
    def __init__(self):
        self._p: List[Particle] = []

    def burst(self, x: float, y: float,
              color: Tuple[int,int,int], count: int = P_LAUNCH):
        for _ in range(count):
            a = random.uniform(0, math.tau)
            s = random.uniform(P_SPMIN, P_SPMAX)
            l = random.uniform(P_LMIN, P_LMAX)
            self._p.append(Particle(
                x, y, math.cos(a)*s, math.sin(a)*s,
                l, l, color, random.uniform(P_RMIN, P_RMAX),
            ))

    def update(self, dt: float):
        alive = []
        for p in self._p:
            p.x   += p.vx * dt
            p.y   += p.vy * dt
            p.vy  += P_GRAV * dt
            p.life -= dt
            if p.life > 0: alive.append(p)
        self._p = alive

    def render(self, surface: pygame.Surface):
        for p in self._p:
            f = p.life / p.max_life
            r = max(int(p.radius * f), 1)
            tmp = pygame.Surface((r*2+2, r*2+2), pygame.SRCALPHA)
            pygame.draw.circle(tmp, (*p.color, int(255*f)), (r+1, r+1), r)
            surface.blit(tmp, (int(p.x)-r-1, int(p.y)-r-1))

@dataclass
class Toast:
    text:     str
    life:     float = 2.0
    max_life: float = 2.0
    color:    Tuple[int,int,int] = (255, 240, 100)


class Notifier:
    def __init__(self):
        pygame.font.init()
        self._font   = pygame.font.SysFont("monospace", 22, bold=True)
        self._toasts: List[Toast] = []

    def show(self, text: str, color: Tuple[int,int,int] = (255, 240, 100)):
        self._toasts.append(Toast(text, color=color))
        print(f"   ℹ  {text}")

    def update(self, dt: float):
        for t in self._toasts: t.life -= dt
        self._toasts = [t for t in self._toasts if t.life > 0]

    def render(self, surface: pygame.Surface):
        y = WIN_H - 130
        for t in reversed(self._toasts[-4:]):
            frac  = t.life / t.max_life
            alpha = int(255 * min(frac * 3, 1.0))
            s     = self._font.render(t.text, True, t.color)
            s.set_alpha(alpha)
            surface.blit(s, (WIN_W//2 - s.get_width()//2, y))
            y -= 36

class UndoStack:
    def __init__(self, max_len: int = 12):
        self._s: Deque[bytes] = deque(maxlen=max_len)

    def push(self, canvas: pygame.Surface):
        self._s.append(pygame.image.tostring(canvas, "RGBA"))

    def clear(self):
        self._s.clear()

    def pop(self, canvas: pygame.Surface) -> bool:
        if not self._s: return False
        img = pygame.image.frombytes(self._s.pop(), (WIN_W, WIN_H), "RGBA")
        canvas.fill((0, 0, 0, 0))
        canvas.blit(img, (0, 0))
        return True

class ColorPalette:
    SW = 44; PAD = 6; TOP = 14

    def __init__(self):
        self._hk:   Optional[str]   = None
        self._ht:   float            = 0.0
        self._anim: Dict[str, float] = {}
        keys  = list(PALETTE.keys())
        total = len(keys) * (self.SW + self.PAD) - self.PAD
        x0    = WIN_W - total - 14
        self._rects: Dict[str, pygame.Rect] = {}
        for k in keys:
            self._rects[k] = pygame.Rect(x0, self.TOP, self.SW, self.SW)
            x0 += self.SW + self.PAD

    def update(self, tip: Optional[Tuple[int,int]], dt: float) -> Optional[str]:
        hov = None
        if tip:
            for k, r in self._rects.items():
                if r.collidepoint(tip): hov = k; break

        fired = None
        if hov and hov == self._hk:
            self._ht += dt
            self._anim[hov] = min(self._ht / UI_HOVER_S, 1.0)
            if self._ht >= UI_HOVER_S:
                fired = hov
                self._ht = 0.0
        else:
            if self._hk: self._anim[self._hk] = 0.0
            self._hk = hov
            self._ht = 0.0

        for k in list(self._anim):
            if k != self._hk:
                self._anim[k] = max(self._anim.get(k, 0.0) - dt*3.0, 0.0)
        return fired

    def render(self, surface: pygame.Surface, active: str):
        for k, r in self._rects.items():
            col  = PALETTE[k]["rgb"]
            anim = self._anim.get(k, 0.0)
            pygame.draw.rect(surface, col, r, border_radius=8)
            if anim > 0:
                fh = int(r.h * anim)
                fr = pygame.Rect(r.x, r.bottom - fh, r.w, fh)
                brighter = tuple(min(c + 60, 255) for c in col)
                pygame.draw.rect(surface, brighter, fr, border_radius=8)
            bc = (255, 255, 255) if k == active else (80, 80, 80)
            bw = 3 if k == active else 1
            pygame.draw.rect(surface, bc, r, bw, border_radius=8)


@dataclass
class TBtn:
    key: str
    label: str
    color: Tuple[int, int, int]


class Toolbar:
    H = 48; PAD = 8; BW = 72
    BTNS = [
        TBtn("launch",     "LAUNCH",  (230, 180,  50)),
        TBtn("clear",      "CLEAR",   (220,  80,  80)),
        TBtn("undo",       "UNDO",     (100, 180, 230)),
        TBtn("redo",       "REDO",     (120, 170, 235)),
        TBtn("gravity",    "GRAVITY", (130, 100, 230)),
        TBtn("trails",     "TRAILS",  ( 80, 210, 130)),
        TBtn("brushes",    "BRUSH",   ( 60, 200, 240)),
        TBtn("wind_l",     "WIND",    ( 40, 200, 200)),
        TBtn("wind_r",     "WIND",    ( 40, 200, 200)),
        TBtn("screenshot", "SAVE",   (200, 200,  80)),
        TBtn("explode",    "BLAST",  (230, 120,  50)),
        TBtn("to3d",       "TO 3D",  (160, 110, 255)),
        TBtn("mode3d",     "2D/3D",  ( 90, 150, 255)),
        TBtn("effects",    "FX",     (190, 130, 255)),
        TBtn("mode",       "MODE",   (110, 220, 255)),
        TBtn("theme",      "THEME",  (255, 110, 210)),
    ]

    def __init__(self):
        total = len(self.BTNS) * (self.BW + self.PAD) - self.PAD
        x0    = (WIN_W - total) // 2
        y0    = WIN_H - self.H - 8
        self._rects: Dict[str, pygame.Rect] = {}
        for b in self.BTNS:
            self._rects[b.key] = pygame.Rect(x0, y0, self.BW, self.H)
            x0 += self.BW + self.PAD
        self._hk:   Optional[str]   = None
        self._ht:   float            = 0.0
        self._anim: Dict[str, float] = {}
        self._font = pygame.font.SysFont("monospace", 13, bold=True)
        self.mode_label = "MODE 2D"
        self.cur_mode = "DRAW"

    def update(self, tip: Optional[Tuple[int,int]], dt: float) -> Optional[str]:
        hov = None
        if tip:
            for k, r in self._rects.items():
                if r.collidepoint(tip): hov = k; break

        fired = None
        if hov and hov == self._hk:
            self._ht += dt
            self._anim[hov] = min(self._ht / UI_HOVER_S, 1.0)
            if self._ht >= UI_HOVER_S:
                fired = hov
                self._ht = 0.0
        else:
            if self._hk: self._anim[self._hk] = 0.0
            self._hk = hov
            self._ht = 0.0

        for k in list(self._anim):
            if k != self._hk:
                self._anim[k] = max(self._anim.get(k, 0.0) - dt*3.0, 0.0)
        return fired

    def render(self, surface: pygame.Surface):
        for b in self.BTNS:
            r = self._rects[b.key]
            anim = self._anim.get(b.key, 0.0)
            bg = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
            bg.fill((*b.color, int(180 + 60 * anim)))
            if anim > 0:
                fw = int(r.w * anim)
                lighter = tuple(min(c + 40, 255) for c in b.color)
                pygame.draw.rect(bg, (*lighter, 200), (0, 0, fw, r.h))
            pygame.draw.rect(bg, (*b.color, 255), (0, 0, r.w, r.h), 2, border_radius=8)
            surface.blit(bg, r.topleft)
            lb = self._font.render(self.mode_label if b.key == "mode3d" else (getattr(self, "cur_mode", "MODE") if b.key == "mode" else b.label), True, (255, 255, 255))
            surface.blit(lb, (r.x + (r.w - lb.get_width())//2,
                               r.y + (r.h - lb.get_height())//2))

RADIAL_OPT = [
    {"key": "launch",     "icon": "🚀", "label": "Launch"},
    {"key": "clear",      "icon": "🗑", "label": "Clear"},
    {"key": "gravity",    "icon": "🌍", "label": "Gravity"},
    {"key": "screenshot", "icon": "📸", "label": "Save"},
    {"key": "explode",    "icon": "💥", "label": "Explode"},
    {"key": "trails",     "icon": "✨", "label": "Trails"},
    {"key": "brushes",    "icon": "🖌", "label": "Brushes"},
    {"key": "to3d",       "icon": "🧊", "label": "To 3D"},
    {"key": "mode3d",     "icon": "🔮", "label": "2D/3D"},
    {"key": "effects",    "icon": "✨", "label": "Effects"},
    {"key": "mode",       "icon": "🎛", "label": "Mode"},
    {"key": "theme",      "icon": "🌌", "label": "Theme"},
    {"key": "hud",        "icon": "📟", "label": "HUD"},
]

class RadialMenu:
    RAD = 156; BTN_R = 34; DUR = 0.25

    def __init__(self):
        pygame.font.init()
        self._fn  = pygame.font.SysFont("monospace", 14, bold=True)
        self._fi  = pygame.font.SysFont("segoeui", 22)
        self._vis = False
        self._cx  = WIN_W // 2
        self._cy  = WIN_H // 2
        self._anim = 0.0
        self._hov: Optional[str] = None

    @property
    def visible(self) -> bool: return self._vis

    def open(self, cx: int, cy: int):
        self._vis  = True
        self._cx, self._cy = cx, cy
        self._anim = 0.0

    def close(self):
        self._vis = False
        self._hov = None

    def _pos(self, i: int, n: int) -> Tuple[int, int]:
        ang = math.pi * 1.5 + i * math.tau / n
        r   = self.RAD * self._anim
        return (int(self._cx + math.cos(ang) * r),
                int(self._cy + math.sin(ang) * r))

    def update(self, tip: Optional[Tuple[int,int]], dt: float) -> Optional[str]:
        if not self._vis: return None
        self._anim = min(self._anim + dt / self.DUR, 1.0)
        self._hov  = None
        fired      = None
        if tip and self._anim > 0.8:
            for i, opt in enumerate(RADIAL_OPT):
                px, py = self._pos(i, len(RADIAL_OPT))
                if math.hypot(tip[0]-px, tip[1]-py) < self.BTN_R:
                    self._hov = opt["key"]
                    fired     = opt["key"]
                    break
        return fired

    def render(self, surface: pygame.Surface):
        if not self._vis: return
        n = len(RADIAL_OPT)
        dim = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, int(100 * self._anim)))
        surface.blit(dim, (0, 0))
        pygame.draw.circle(surface, (40, 40, 40),    (self._cx, self._cy), 28)
        pygame.draw.circle(surface, (200, 200, 200), (self._cx, self._cy), 28, 2)
        for i, opt in enumerate(RADIAL_OPT):
            px, py = self._pos(i, n)
            pygame.draw.line(surface, (80, 80, 80), (self._cx, self._cy), (px, py), 1)
            hov    = (self._hov == opt["key"])
            bgcol  = (160, 140, 60) if hov else (90, 90, 90)
            pygame.draw.circle(surface, bgcol,         (px, py), self.BTN_R)
            pygame.draw.circle(surface, (200, 200, 200), (px, py), self.BTN_R, 2)
            ic = self._fi.render(opt["icon"],  True, (255, 255, 255))
            lb = self._fn.render(opt["label"], True, (220, 220, 220))
            surface.blit(ic, (px - ic.get_width()//2, py - ic.get_height()//2 - 6))
            surface.blit(lb, (px - lb.get_width()//2, py + 18))

_PANEL_CACHE: Dict[Tuple, pygame.Surface] = {}


class HUD:
    def __init__(self):
        pygame.font.init()
        self._big   = pygame.font.SysFont("monospace", 26, bold=True)
        self._med   = pygame.font.SysFont("monospace", 18, bold=True)
        self._small = pygame.font.SysFont("monospace", 15)

    @staticmethod
    def _panel(surface, x, y, w, h, col=(0,0,0), alpha=155):
        k = (w, h, col, alpha)
        s = _PANEL_CACHE.get(k)
        if s is None:
            if len(_PANEL_CACHE) > 200: _PANEL_CACHE.clear()
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            s.fill((*col, alpha))
            _PANEL_CACHE[k] = s
        surface.blit(s, (x, y))

    def render(self, surface: pygame.Surface,
               hands:       List[HandData],
               color_key:   str,
               pen_w:       int,
               fps:         float,
               n_bodies:    int,
               grav_idx:    int,
               show_trails: bool,
               wind:        float,
               extra=None):

        W, H = surface.get_size()

        MODES = {
            "DRAW":   ("✏  DRAW",    (80,  225,  80)),
            "LAUNCH": ("✌ LAUNCH",  (230, 185,  50)),
            "PALM":   ("🖐 MENU",    (150, 200, 240)),
            "GRAB":   ("GRAB",       ( 80, 180, 230)),
            "ERASE":  ("👊 ERASE",  (230, 100,  60)),
            "PINCH":  ("🤌 RESIZE", (200, 100, 230)),
            "PUSH":   ("👊 PUSH",   (230, 140,  60)),
            "BLAST":  ("🖐 BLAST",  (230,  80,  50)),
            "IDLE":   ("   IDLE",   (150, 150, 150)),
        }
        y_badge = 10
        for hd in hands:
            txt, col = MODES.get(hd.gesture, MODES["IDLE"])
            prefix   = "R" if hd.label == "Right" else "L"
            self._panel(surface, 10, y_badge, 330, 40)
            surface.blit(
                self._big.render(f"{prefix}: {txt}", True, col),
                (18, y_badge + 6))
            y_badge += 50

        # Info strips
        g_cols = [(100,160,230),(230,160,100),(200,100,230)]
        self._panel(surface, 10, y_badge, 265, 28)
        surface.blit(
            self._small.render(f"[G] gravity: {GRAVITY_LABELS[grav_idx]}",
                               True, g_cols[grav_idx]),
            (16, y_badge + 5))
        y_badge += 32

        self._panel(surface, 10, y_badge, 220, 28)
        tc = (80, 220, 80) if show_trails else (150, 150, 150)
        surface.blit(
            self._small.render(f"[T] trails: {'ON' if show_trails else 'OFF'}",
                               True, tc),
            (16, y_badge + 5))
        y_badge += 32

        if wind != 0.0:
            self._panel(surface, 10, y_badge, 220, 28)
            surface.blit(
                self._small.render(
                    f"wind: {'←' if wind < 0 else '→'} {abs(int(wind))}",
                    True, (40, 220, 210)),
                (16, y_badge + 5))
            y_badge += 32

        self._panel(surface, 10, y_badge, 180, 28)
        surface.blit(
            self._small.render(f"pen: {pen_w}px  [+/-]", True, (210,210,210)),
            (16, y_badge + 5))

        if extra:
            y_badge += 32
            for txt, col in extra:
                self._panel(surface, 10, y_badge, 265, 28)
                surface.blit(self._small.render(txt, True, col), (16, y_badge + 5))
                y_badge += 32

        self._panel(surface, 0, H-28, W, 28)
        bar = (f" FPS:{fps:3.0f} Obj:{n_bodies} | SPACE launch  C clear  Z undo  E blast  G grav  S save  D debug"
               "  B brush  X 2D/3D  V to3D  I edit  L layers  H history  Y redo  Q quit")
        surface.blit(self._small.render(bar, True, (160,160,160)), (4, H-23))

class GlowCursor:
    def __init__(self):
        self._trails: Dict[str, Deque] = {
            "Right": deque(maxlen=FINGER_TRAIL_LEN),
            "Left":  deque(maxlen=FINGER_TRAIL_LEN),
        }

    def update(self, label: str,
               tip:   Optional[Tuple[int,int]],
               color: Tuple[int,int,int]):
        if tip:
            self._trails[label].append((tip, color))

    def render(self, surface: pygame.Surface,
               label: str,
               tip: Optional[Tuple[int,int]],
               color: Tuple[int,int,int],
               gesture: str):
        if not tip: return

        trail = self._trails.get(label, deque())
        n = len(trail)
        for i, (pt, col) in enumerate(trail):
            frac = (i + 1) / max(n, 1)
            alpha = int(180 * frac)
            r = max(int(CURSOR_INNER * frac), 1)
            tmp = pygame.Surface((r*2+2, r*2+2), pygame.SRCALPHA)
            pygame.draw.circle(tmp, (*col, alpha), (r+1, r+1), r)
            surface.blit(tmp, (pt[0]-r-1, pt[1]-r-1))

        glow = pygame.Surface((GLOW_R*2, GLOW_R*2), pygame.SRCALPHA)
        for rr in range(GLOW_R, 0, -4):
            a = int(60 * rr / GLOW_R)
            pygame.draw.circle(glow, (*color, a), (GLOW_R, GLOW_R), rr)
        surface.blit(glow, (tip[0]-GLOW_R, tip[1]-GLOW_R))

        ring_col = {
            "DRAW":   color,
            "GRAB":   ( 80, 180, 230),
            "ERASE":  (230,  80,  80),
            "PINCH":  (200, 100, 230),
            "LAUNCH": (230, 185,  50),
            "PALM":   (150, 200, 240),
            "PUSH":   (230, 140,  60),
            "BLAST":  (230,  80,  50),
            "IDLE":   (150, 150, 150),
        }.get(gesture, (200, 200, 200))

        pygame.draw.circle(surface, ring_col, tip, CURSOR_OUTER, 2)
        pygame.draw.circle(surface, ring_col, tip, CURSOR_INNER)

def _wrap_angle(a: float) -> float:
    return (a + math.pi) % math.tau - math.pi


class GrabController:
    """Thumb+middle pinch -> select, hold, move, ROTATE, SCALE, PUSH/PULL, throw.

      move    : pinch point drives a pivot joint (real physics, no teleporting)
      rotate  : roll the wrist (single hand)  or  turn the line between two pinching hands
      scale   : pinch with BOTH hands, move them apart / together         ([ ] keys as fallback)
      push    : while holding, move hand toward the camera  -> other objects pushed away (P key)
      pull    : while holding, move hand away from camera   -> other objects pulled in    (O key)
      throw   : open the pinch while moving -> linear velocity + spin
    Works with either hand; one object at a time; the hand that grabbed owns it."""

    def __init__(self, phys: "PhysicsWorld"):
        self.phys = phys
        self.vel = VelocityTracker()
        self.active = False
        self.hand: Optional[str] = None
        self.body: Optional[PhysBody] = None
        self.hover: Optional[PhysBody] = None
        self.info: List[str] = []
        self.key_scale = 0          # -1 / 0 / +1   ([ and ])
        self.key_force = 0          # +1 push (P)  -1 pull (O)
        self.field = 0
        self.field_strength = 0.0
        self._sm: Optional[Tuple[float, float]] = None
        self._was = {"Right": False, "Left": False}
        self._acq = {"Right": 0.0, "Left": 0.0}
        self._reset_manip()

    def _reset_manip(self):
        self._ang_last: Optional[float] = None
        self._ang_acc = 0.0
        self._ang_s = 0.0
        self._rot_armed = False
        self._rot_off = 0.0
        self._rot_eff = 0.0
        self._rot_hist: Deque[Tuple[float, float]] = deque(maxlen=6)
        self._size_base: Optional[float] = None
        self._size_s = 0.0
        self._depth = 1.0
        self._two = False
        self._d0 = 1.0
        self._s0 = 1.0
        self._d_s = 1.0
        self.field = 0
        self.field_strength = 0.0
        self.info = []

    def _smooth(self, p: Tuple[int, int]) -> Tuple[int, int]:
        if self._sm is None:
            self._sm = (float(p[0]), float(p[1]))
        else:
            a = GRAB_SMOOTH
            self._sm = (a * p[0] + (1 - a) * self._sm[0], a * p[1] + (1 - a) * self._sm[1])
        return int(self._sm[0]), int(self._sm[1])

    def reset(self):
        """Drop any grab without throwing (e.g. after CLEAR)."""
        if self.body is not None:
            self.body.extra = ""
        self.active, self.hand, self.body, self._sm = False, None, None, None
        self.vel.reset()
        self._reset_manip()

    def release(self) -> str:
        vx, vy = self.vel.velocity()
        spin = 0.0
        if self._rot_armed and len(self._rot_hist) >= 2:
            (t0, a0), (t1, a1) = self._rot_hist[0], self._rot_hist[-1]
            if t1 - t0 > 1e-3:
                spin = (a1 - a0) / (t1 - t0)
        self.phys.end_drag((vx, vy), spin)
        self.reset()
        return "release"

    # ---- scale helper (screen-size guarded) ----
    def _apply_scale(self, target: float):
        pb = self.body
        target = max(SCALE_MIN, min(SCALE_MAX, target))
        k = target / pb.scale
        if abs(k - 1.0) < 0.002:
            return
        if k > 1.0:
            _, _, r = self.phys.object_bounds(pb)
            if r * k > 0.45 * min(WIN_W, WIN_H):
                return
        self.phys.scale_body(pb, k)

    def _manipulate(self, h: HandData, other: Optional[HandData], dt: float, now: float):
        pb = self.body

        # ---------- scale (two hands) + choose rotation source ----------
        two = ENABLE_GRAB_SCALE and other is not None and other.grab_pt is not None
        if two:
            d = math.hypot(other.grab_pt[0] - h.grab_pt[0], other.grab_pt[1] - h.grab_pt[1])
            if not self._two:
                self._d0, self._s0, self._d_s = max(d, 40.0), pb.scale, d
                self._ang_last = None
            self._d_s = 0.5 * d + 0.5 * self._d_s
            self._apply_scale(self._s0 * self._d_s / self._d0)
            src_angle = math.atan2(other.grab_pt[1] - h.grab_pt[1], other.grab_pt[0] - h.grab_pt[0])
        else:
            if self._two:
                self._ang_last = None
            src_angle = h.angle
        self._two = two
        if self.key_scale:
            self._apply_scale(pb.scale * (1.0 + SCALE_KEY_STEP * self.key_scale))

        # ---------- rotation ----------
        if ENABLE_GRAB_ROTATION:
            if self._ang_last is not None:
                self._ang_acc += _wrap_angle(src_angle - self._ang_last)
            self._ang_last = src_angle
            self._ang_s += ROT_SMOOTH * (self._ang_acc - self._ang_s)
            if not self._rot_armed and abs(self._ang_s) > math.radians(ROT_DEADZONE_DEG):
                self._rot_armed, self._rot_off = True, self._ang_s
            self._rot_eff = (self._ang_s - self._rot_off) * ROT_GAIN if self._rot_armed else 0.0
            self.phys.rotate_drag(self._rot_eff)
            self._rot_hist.append((now, self._rot_eff))

        # ---------- push / pull from hand depth ----------
        f, st = 0, 0.0
        sz = h.size
        if sz > 0:
            if self._size_base is None:
                self._size_base, self._size_s = sz, sz
            self._size_s = 0.3 * sz + 0.7 * self._size_s
            self._depth = self._size_s / max(self._size_base, 1e-6)
        if ENABLE_PUSH_PULL and not two:
            lr, db = math.log(max(self._depth, 1e-3)), math.log(1.0 + PUSHPULL_DEADBAND)
            if lr > db:    f, st = +1, min(1.0, (lr - db) / PUSHPULL_FULL)
            elif lr < -db: f, st = -1, min(1.0, (-lr - db) / PUSHPULL_FULL)
        if self.key_force:
            f, st = self.key_force, 1.0
        self.field, self.field_strength = f, st
        if f:
            cx, cy, _ = self.phys.object_bounds(pb)
            self.phys.radial_force(cx, cy, f, st, dt, exclude=pb)

        # ---------- feedback ----------
        deg = math.degrees(self._rot_eff)
        bits = []
        if self._rot_armed: bits.append(f"rot {deg:+.0f}")
        if abs(pb.scale - 1.0) > 0.02: bits.append(f"x{pb.scale:.2f}")
        if f: bits.append("PUSH" if f > 0 else "PULL")
        if two: bits.append("2-HAND")
        pb.extra = " ".join(bits)
        self.info = [f"ROT   : {deg:+6.1f} deg {'(armed)' if self._rot_armed else '(deadzone)'}",
                     f"SCALE : x{pb.scale:.2f}  {'2-hand' if two else ''}",
                     f"DEPTH : x{self._depth:.2f}  {'PUSH' if f > 0 else 'PULL' if f < 0 else '-'} {st:.2f}"]

    def update(self, hands: List[HandData], now: Optional[float] = None,
               dt: float = 1 / 60) -> Optional[str]:
        """Returns 'grab' / 'release' on transitions, else None."""
        now = time.perf_counter() if now is None else now
        grabbers = {h.label: h for h in hands if h.gesture == "GRAB" and h.grab_pt}
        for lab in ("Right", "Left"):
            if lab in grabbers and not self._was[lab]:
                self._acq[lab] = now + GRAB_ACQUIRE_S
            self._was[lab] = lab in grabbers

        event = None
        if self.active:
            h = grabbers.get(self.hand)
            if h is None or self.phys._sel is not self.body:
                event = self.release()
            else:
                pt = self._smooth(h.grab_pt)
                self.vel.update(float(pt[0]), float(pt[1]))
                self.phys.update_drag(pt)
                other = grabbers.get("Left" if self.hand == "Right" else "Right")
                self._manipulate(h, other, dt, now)

        if not self.active:
            for lab in ("Right", "Left"):
                h = grabbers.get(lab)
                if h and now <= self._acq[lab]:
                    pb = self.phys.start_drag(h.grab_pt, GRAB_SELECT_RADIUS)
                    if pb:
                        self.active, self.hand, self.body = True, lab, pb
                        self._sm = None
                        self.vel.reset()
                        self._reset_manip()
                        pt = self._smooth(h.grab_pt)
                        self.vel.update(float(pt[0]), float(pt[1]))
                        event = "grab"
                        break

        self.hover = None
        if not self.active:
            for h in hands:
                if (h.grab_pt and h.tm is not None and h.tm < GRAB_HOVER_THRESHOLD
                        and h.gesture in ("IDLE", "GRAB")):
                    pb = self.phys.find_nearest(h.grab_pt, GRAB_SELECT_RADIUS)
                    if pb:
                        self.hover = pb
                        break
            # keyboard scale fallback on the hovered object
            if self.hover is not None and self.key_scale:
                self.body = self.hover
                self._apply_scale(self.hover.scale * (1.0 + SCALE_KEY_STEP * self.key_scale))
                self.body = None
        self.phys.set_hover(self.hover)
        return event

    def render(self, surface: pygame.Surface):
        """Expanding rings (PUSH) / collapsing rings (PULL) around the held object."""
        if not (self.active and self.body is not None and self.field):
            return
        cx, cy, _ = self.phys.object_bounds(self.body)
        R = PUSHPULL_RADIUS
        layer = pygame.Surface((R * 2, R * 2), pygame.SRCALPHA)
        col = (255, 150, 70) if self.field > 0 else (110, 190, 255)
        t = time.perf_counter()
        for k in range(3):
            ph = (t * 0.9 + k / 3.0) % 1.0
            rad = R * (ph if self.field > 0 else 1.0 - ph)
            a = int(170 * (1 - ph) * (0.35 + 0.65 * self.field_strength))
            if rad > 6:
                pygame.draw.circle(layer, (*col, a), (R, R), int(rad), 3)
        surface.blit(layer, (int(cx - R), int(cy - R)))


class DebugOverlay:
    """[D] toggles. Shows the numbers behind every gesture decision."""
    def __init__(self):
        pygame.font.init()
        self._f = pygame.font.SysFont("monospace", 14)
        self._b = pygame.font.SysFont("monospace", 15, bold=True)

    @staticmethod
    def _state(v: float, ext_t: float, fold_t: float) -> str:
        return "EXTENDED" if v >= ext_t else ("FOLDED" if v <= fold_t else "partial")

    def render(self, surface: pygame.Surface, hands: List[HandData],
               grab: "GrabController", fps: float, extra=None):
        if extra:
            w, h = 340, 8 + 18 * len(extra)
            pn = pygame.Surface((w, h), pygame.SRCALPHA); pn.fill((0, 0, 0, 170))
            x0 = (surface.get_width() - w) // 2
            surface.blit(pn, (x0, 70))
            for i, t in enumerate(extra):
                surface.blit((self._b if i == 0 else self._f).render(t, True, (200, 170, 255)), (x0 + 8, 75 + 18 * i))
        for hd in hands:
            f = hd.feats
            if f is None: continue
            R = f.reach
            lines = [
                (f"{hd.label.upper()} HAND", (255, 255, 255)),
                (f"INDEX : {self._state(R['index'], INDEX_EXTENDED_THRESHOLD, INDEX_FOLDED_THRESHOLD):8s} {R['index']:.2f} {f.index_angle:3.0f}deg", (230, 230, 230)),
                (f"MIDDLE: {self._state(R['middle'], FINGER_EXTENDED_THRESHOLD, MIDDLE_FOLDED_THRESHOLD):8s} {R['middle']:.2f}", (230, 230, 230)),
                (f"RING  : {self._state(R['ring'], FINGER_EXTENDED_THRESHOLD, RING_FOLDED_THRESHOLD):8s} {R['ring']:.2f}", (230, 230, 230)),
                (f"PINKY : {self._state(R['pinky'], FINGER_EXTENDED_THRESHOLD, PINKY_FOLDED_THRESHOLD):8s} {R['pinky']:.2f}", (230, 230, 230)),
                (f"THUMB : {'FOLDED' if f.thumb_in <= THUMB_FOLDED_THRESHOLD else 'OUT':8s} {f.thumb_in:.2f}", (230, 230, 230)),
                (f"PINCH T-M: {f.tm:.3f} (grab<{GRAB_PINCH_THRESHOLD:.2f})", (120, 220, 255)),
                (f"PINCH T-I: {f.ti:.3f} (size<{RESIZE_ENTER:.2f})", (200, 150, 255)),
                (f"RAW: {f.raw}   STABLE: {hd.gesture}", (255, 230, 90)),
            ]
            if hd.gesture == "GRAB" or (grab.active and grab.hand == hd.label):
                lines.append((f"OBJECT = #{grab.body.uid}" if grab.body else "OBJECT = NONE", (120, 255, 160)))
                if grab.active and grab.hand == hd.label:
                    lines += [(t, (255, 200, 120)) for t in grab.info]
            elif grab.hover:
                lines.append((f"HOVER  = #{grab.hover.uid}", (120, 220, 255)))
            else:
                lines.append(("OBJECT = NONE", (150, 150, 150)))
            w, h = 300, 8 + 18 * len(lines)
            x = 10 if hd.label == "Right" else surface.get_width() - w - 230
            y = 270 if hd.label == "Right" else 70
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            s.fill((0, 0, 0, 170))
            surface.blit(s, (x, y))
            for i, (t, c) in enumerate(lines):
                surface.blit((self._b if i == 0 else self._f).render(t, True, c), (x + 8, y + 5 + 18 * i))
            if hd.grab_pt:
                pygame.draw.circle(surface, (120, 220, 255), hd.grab_pt, 6, 1)
            if hd.tip:
                pygame.draw.circle(surface, (255, 255, 255), hd.tip, 9, 1)
        if not hands:
            surface.blit(self._b.render("DEBUG: no hands detected", True, (255, 120, 120)), (10, 270))


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 6 — 3D MODE
#  World space = screen pixels (x right, y down) + z toward the viewer.  A point
#  with z=0 projects exactly onto its 2-D position, so 2-D drawings morph into
#  3-D objects without jumping.  Depth is *approximate interaction depth*
#  (hand-size based), not metric depth.
# ══════════════════════════════════════════════════════════════════════════════
ENABLE_3D         = True
AUTO_3D_DEFAULT   = False      # True = every finished stroke is converted automatically
DEPTH_METHOD      = "hand_size"   # "hand_size" (hand toward camera = closer)  |  "keyboard" (PgUp/PgDn)
DEPTH_SENSITIVITY = 900.0      # px of z per +100 % hand-size change
DEPTH_SMOOTH      = 0.25       # EMA on hand size
DEPTH_DEADBAND    = 0.03       # ignore +-3 % size noise
Z_RANGE           = 380.0      # z is clamped to +-Z_RANGE
FOCAL             = 900.0      # perspective focal length (px)
MAX_SOLIDS        = 30
MORPH_S           = 0.80       # 2-D -> 3-D inflate animation length
LAUNCH_RAMP_S     = 1.00       # gravity fades in over this long after a 3-D launch
IDLE_SPIN         = 0.0        # rad/s auto-rotation of not-yet-launched solids (0 = off)
SOLID_FACE_ALPHA  = 105        # translucency of the shaded faces (0 = pure wireframe)
SOLID_BOUNCE      = 0.50
SOLID_AIR_DRAG    = 0.35
SOLID_ANG_DRAG    = 0.55
SOLID_REBUILD_MS  = 9.0        # per-frame budget for re-rendering moving solids
GRID3D            = True       # perspective floor grid in 3-D mode
SCALE3D_MIN, SCALE3D_MAX = 0.30, 3.5
ORBIT_GAIN        = 0.006      # rad per px of left-fist drag
CONVERT_CHIP_S    = 7.0
CATCH_RADIUS      = 120        # a pinching hand catches a thrown (flying) solid within this many px
HANDOFF_RADIUS    = 160        # second hand must pinch this close to the object for a hand-off (else it's a throw)
CATCH_MIN_SPEED   = 120.0      # px/s - resting solids are never "magnetised", only flying ones
SCALE3D_DEADBAND  = 0.10       # two-hand distance must change >10 % before scaling starts (hand-offs don't resize)
FREE_FINGER_SPIN  = 0.012      # rad per px: the free hand's pointing finger spins the held solid like a trackball
TILT_X, TILT_Y    = 0.50, -0.62   # pleasing 3/4 view the solid tilts into while inflating
_LIGHT = np.array([-0.40, -0.60, 0.70]) / np.linalg.norm([-0.40, -0.60, 0.70])


# ───────────────────────────── math helpers ─────────────────────────────
def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)

def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)

def _axis_angle(axis, ang: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n < 1e-9 or abs(ang) < 1e-9:
        return np.eye(3)
    x, y, z = np.asarray(axis, float) / n
    c, s = math.cos(ang), math.sin(ang); C = 1 - c
    return np.array([[c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]], float)

def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt

def _smootherstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (6 * t - 15) + 10)


@dataclass
class Point3D:
    """One drawing point.  x,y = screen-pixel world coords, z = approximate depth (+ = closer)."""
    x: float; y: float; z: float
    def arr(self) -> np.ndarray: return np.array([self.x, self.y, self.z], float)
    @staticmethod
    def of(a) -> "Point3D": return Point3D(float(a[0]), float(a[1]), float(a[2]))


class View3D:
    """Orbit camera.  Rotates the world about the screen centre, then perspective-projects."""
    C = np.array([WIN_W / 2.0, WIN_H / 2.0, 0.0])

    def __init__(self):
        self.yaw = self.pitch = 0.0
        self.t_yaw = self.t_pitch = 0.0
        self._recalc()

    def _recalc(self):
        self.R = _rot_x(self.pitch) @ _rot_y(self.yaw)

    @property
    def key(self) -> Tuple[float, float]:
        return (round(self.yaw, 4), round(self.pitch, 4))

    def orbit(self, dyaw: float, dpitch: float):
        self.t_yaw += dyaw
        self.t_pitch = max(-1.25, min(1.25, self.t_pitch + dpitch))

    def reset(self):
        self.t_yaw = self.t_pitch = 0.0

    def update(self, dt: float):
        a = 1.0 - math.exp(-dt * 12.0)
        self.yaw += (self.t_yaw - self.yaw) * a
        self.pitch += (self.t_pitch - self.pitch) * a
        if abs(self.t_yaw - self.yaw) < 1e-4: self.yaw = self.t_yaw
        if abs(self.t_pitch - self.pitch) < 1e-4: self.pitch = self.t_pitch
        self._recalc()

    def to_view(self, P: np.ndarray) -> np.ndarray:
        return (P - self.C) @ self.R.T + self.C

    def to_world(self, V: np.ndarray) -> np.ndarray:
        return (V - self.C) @ self.R + self.C

    @staticmethod
    def persp(zv):
        return FOCAL / np.maximum(FOCAL - zv, 0.2 * FOCAL)

    def project(self, Vv: np.ndarray):
        k = self.persp(Vv[:, 2])
        return (View3D.C[0] + (Vv[:, 0] - View3D.C[0]) * k,
                View3D.C[1] + (Vv[:, 1] - View3D.C[1]) * k, k)

    def unproject(self, sx: float, sy: float, zv: float) -> np.ndarray:
        k = float(self.persp(np.array([zv]))[0])
        return np.array([View3D.C[0] + (sx - View3D.C[0]) / k,
                         View3D.C[1] + (sy - View3D.C[1]) / k, zv])


# ───────────────────────────── the 3-D object ─────────────────────────────
class Solid3D:
    """A 3-D wire/shaded object (a converted drawing, or a stroke drawn in 3-D mode)."""
    _uid = 0
    CULL = ("sphere", "ellipsoid", "cube", "box", "prism", "pyramid", "cylinder", "star3d")

    def __init__(self, kind, V, chains, faces, rgb, brush, width, pos, name=""):
        Solid3D._uid += 1
        self.uid = Solid3D._uid
        self.kind, self.name = kind, name
        self.V = np.asarray(V, float).reshape(-1, 3)
        self.chains = [np.asarray(c, int) for c in chains]
        self.faces = [tuple(int(i) for i in f) for f in faces]
        self.cull = kind in Solid3D.CULL
        self.rgb, self.brush, self.width = rgb, brush, width
        self.pos = np.array(pos, float)
        self.R = np.eye(3)
        self.scale = 1.0
        self.vel = np.zeros(3); self.ang = np.zeros(3)
        self.active = False; self.held = False; self.launch_t = 0.0
        self.morph = 1.0; self.tilt = (0.0, 0.0)
        self.hover = self.selected = False
        self.building = False
        self.pts: List[np.ndarray] = []
        self.extra = ""
        self.effect: Optional[str] = None       # Phase 4 particle effect key
        self.opacity = 1.0; self.glow = 0.0; self.visible = True          # Phase 5 editable properties
        self.mass = 1.0; self.bounce = SOLID_BOUNCE; self.fric = 1.0; self.gscale = 1.0; self.frozen = False
        self.src = None
        self.born = time.perf_counter()
        self._spr = None; self._spr_key = None; self._spr_org = (0, 0); self._spr_c = (0.0, 0.0)
        self._spr_t = -1e9; self._lod = False
        self._psx = self._psy = None
        self.prep()

    def prep(self):
        self.radius = float(np.max(np.linalg.norm(self.V, axis=1))) if len(self.V) else 1.0
        N = []
        for f in self.faces:
            P = self.V[list(f)]
            nx = ny = nz = 0.0
            for i in range(len(P)):
                a, b = P[i], P[(i + 1) % len(P)]
                nx += (a[1] - b[1]) * (a[2] + b[2])
                ny += (a[2] - b[2]) * (a[0] + b[0])
                nz += (a[0] - b[0]) * (a[1] + b[1])
            v = np.array([nx, ny, nz]); l = np.linalg.norm(v)
            v = v / l if l > 1e-9 else np.array([0.0, 0.0, 1.0])
            if np.dot(v, P.mean(axis=0)) < 0: v = -v          # outward (solids are star-shaped about the origin)
            N.append(v)
        self.N = np.array(N).reshape(-1, 3)

    def to_points(self) -> List[Point3D]:
        """Vertices in world space as Point3D (for export / inspection)."""
        return [Point3D.of(p) for p in self.world_verts()]

    def morph_R(self) -> np.ndarray:
        if self.tilt == (0.0, 0.0): return self.R
        e = _smootherstep(self.morph)          # at morph=1 this equals the baked result, so no visible jump
        return _rot_x(self.tilt[0] * e) @ _rot_y(self.tilt[1] * e) @ self.R

    def local_verts(self) -> np.ndarray:
        if self.morph >= 1.0: return self.V
        V = self.V.copy(); V[:, 2] *= _smootherstep(self.morph)
        return V

    def world_verts(self) -> np.ndarray:
        return (self.local_verts() @ self.morph_R().T) * self.scale + self.pos

    def finish_morph(self):
        self.R = _rot_x(self.tilt[0]) @ _rot_y(self.tilt[1]) @ self.R
        self.tilt = (0.0, 0.0)
        self.morph = 1.0


# ───────────────────────────── shape recognition ─────────────────────────────
@dataclass
class ShapeResult:
    kind: str                      # circle ellipse square rectangle quad triangle star line closed open
    outline: np.ndarray            # idealised outline, absolute screen px (M,2)
    center: Tuple[float, float]
    params: Dict
    @property
    def shape_name(self) -> str: return SHAPE_NAMES[self.kind][0]
    @property
    def solid_name(self) -> str: return SHAPE_NAMES[self.kind][1]

SHAPE_NAMES = {
    "circle": ("Circle", "Sphere"), "ellipse": ("Ellipse", "Ellipsoid"),
    "square": ("Square", "Cube"), "rectangle": ("Rectangle", "Box"), "quad": ("Quad", "Prism"),
    "triangle": ("Triangle", "Pyramid"), "star": ("Star", "3D Star"), "line": ("Line", "Cylinder"),
    "closed": ("Shape", "Inflated Shape"), "open": ("Curve", "Tube"),
}


class ShapeRecognizer:
    """Deterministic geometry recognition (no ML).  Runs once per finished stroke, never per frame.
    Every stroke maps to *something*: unknown closed shapes inflate, unknown open curves become tubes."""

    @staticmethod
    def resample(P, n: int, closed: bool = False) -> np.ndarray:
        P = np.asarray(P, float)
        Q = np.vstack([P, P[:1]]) if closed else P
        seg = np.hypot(*np.diff(Q, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 1e-9: return np.repeat(P[:1], n, axis=0)
        t = np.linspace(0.0, s[-1], n, endpoint=not closed)
        return np.stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])], axis=1)

    @staticmethod
    def _smooth(P, closed: bool, passes: int = 2) -> np.ndarray:
        P = np.asarray(P, float)
        for _ in range(passes):
            if closed: P = (np.roll(P, 1, 0) + 2 * P + np.roll(P, -1, 0)) / 4.0
            else:
                Q = P.copy(); Q[1:-1] = (P[:-2] + 2 * P[1:-1] + P[2:]) / 4.0; P = Q
        return P

    @staticmethod
    def _rdp(P, eps: float) -> np.ndarray:
        n = len(P); keep = np.zeros(n, bool); keep[0] = keep[-1] = True
        st = [(0, n - 1)]
        while st:
            a, b = st.pop()
            if b <= a + 1: continue
            A, B = P[a], P[b]; AB = B - A; L = math.hypot(*AB); seg = P[a + 1:b]
            d = (np.hypot(*(seg - A).T) if L < 1e-9
                 else np.abs(AB[0] * (seg[:, 1] - A[1]) - AB[1] * (seg[:, 0] - A[0])) / L)
            i = int(np.argmax(d))
            if d[i] > eps:
                keep[a + 1 + i] = True; st += [(a, a + 1 + i), (a + 1 + i, b)]
        return np.nonzero(keep)[0]

    @staticmethod
    def _corners(R, eps: float, min_turn_deg: float = 30.0) -> np.ndarray:
        M = len(R)
        far = int(np.argmax(np.hypot(*(R - R[0]).T)))
        a = ShapeRecognizer._rdp(R[:far + 1], eps)
        b = ShapeRecognizer._rdp(np.vstack([R[far:], R[:1]]), eps) + far
        idx = sorted(set(a.tolist()) | set((b % M).tolist()))
        C = R[idx]
        while len(C) > 3:                                    # drop near-collinear vertices
            v1 = C - np.roll(C, 1, 0); v2 = np.roll(C, -1, 0) - C
            ang = np.abs(np.arctan2(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0],
                                    v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]))
            i = int(np.argmin(ang))
            if math.degrees(ang[i]) < min_turn_deg: C = np.delete(C, i, 0)
            else: break
        return C

    @staticmethod
    def _trim_overlap(P) -> np.ndarray:
        """If the stroke overshoots its own start (typical for circles), cut it at the first return."""
        D = float(np.hypot(*(P.max(0) - P.min(0))))
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        d0 = np.hypot(*(P - P[0]).T)
        cand = np.nonzero((s > 0.55 * s[-1]) & (d0 < max(0.10 * D, 6.0)))[0]
        return P[:cand[0] + 1] if len(cand) and cand[0] >= 5 else P

    @staticmethod
    def _area(P) -> float:
        x, y = P[:, 0], P[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def recognize(pts) -> Optional[ShapeResult]:
        """None for specks / tiny scribbles (< 24 px)."""
        P = np.asarray(pts, float)
        if len(P) < 6: return None
        L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
        D = float(np.hypot(*(P.max(0) - P.min(0))))
        if D < 24.0 or L < 40.0: return None
        P = ShapeRecognizer._trim_overlap(P)
        L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
        D = float(np.hypot(*(P.max(0) - P.min(0))))
        gap = float(np.hypot(*(P[0] - P[-1])))
        R = ShapeRecognizer.resample(P, 96, closed=True)
        area = ShapeRecognizer._area(R)
        bbox = float(np.prod(np.maximum(P.max(0) - P.min(0), 1.0)))
        closed = gap < min(0.28 * L, 0.35 * D) and area > 0.06 * bbox
        if not closed:
            chord = float(np.hypot(*(P[-1] - P[0])))
            if chord / L > 0.93 or len(ShapeRecognizer._rdp(P, 0.05 * L)) == 2:
                q = np.array([P[0], P[-1]])
                return ShapeResult("line", q, tuple(q.mean(0)), {"length": chord})
            O = ShapeRecognizer._smooth(ShapeRecognizer.resample(P, 36), False)
            return ShapeResult("open", O, tuple(O.mean(0)), {"length": L})
        return ShapeRecognizer._closed(R, L, area)

    @staticmethod
    def _closed(R, L: float, area: float) -> ShapeResult:
        c = R.mean(0); cen = (float(c[0]), float(c[1]))
        C = ShapeRecognizer._corners(R, 0.03 * L)
        n = len(C)
        # triangle
        if n == 3:
            return ShapeResult("triangle", C, tuple(C.mean(0)), {})
        # rectangle (any rotation)
        if n == 4:
            ang = []
            for i in range(4):
                a, b = C[i - 1] - C[i], C[(i + 1) % 4] - C[i]
                ang.append(math.degrees(math.acos(max(-1, min(1, np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))))))
            if all(abs(a - 90) < 24 for a in ang):
                E = np.roll(C, -1, 0) - C
                phi = float(np.angle(np.sum(np.exp(4j * np.arctan2(E[:, 1], E[:, 0]))))) / 4.0
                ax = np.array([math.cos(phi), math.sin(phi)]); ay = np.array([-ax[1], ax[0]])
                q = (C - C.mean(0))
                pu, pv = q @ ax, q @ ay
                w, h = pu.max() - pu.min(), pv.max() - pv.min()
                cu, cv = (pu.max() + pu.min()) / 2, (pv.max() + pv.min()) / 2
                mid = C.mean(0) + ax * cu + ay * cv
                hw, hh = w / 2, h / 2
                rect = np.array([mid - ax * hw - ay * hh, mid + ax * hw - ay * hh,
                                 mid + ax * hw + ay * hh, mid - ax * hw + ay * hh])
                sq = max(w, h) / max(min(w, h), 1e-6) < 1.15
                if sq:                                         # snap to a perfect square
                    side = (w + h) / 2; hw = hh = side / 2
                    rect = np.array([mid - ax * hw - ay * hh, mid + ax * hw - ay * hh,
                                     mid + ax * hw + ay * hh, mid - ax * hw + ay * hh])
                return ShapeResult("square" if sq else "rectangle", rect, tuple(mid),
                                   {"w": float(w), "h": float(h), "side": float((w + h) / 2)})
        # circle / ellipse
        q = R - c
        ev, evec = np.linalg.eigh(np.cov(q.T))
        a_len, b_len = math.sqrt(2 * max(ev[1], 1e-9)), math.sqrt(2 * max(ev[0], 1e-9))
        uv = q @ evec
        rho = np.hypot(uv[:, 1] / a_len, uv[:, 0] / b_len)
        if float(np.std(rho)) / float(np.mean(rho)) < 0.095 and a_len / b_len < 3.5:
            t = np.linspace(0, math.tau, 32, endpoint=False)
            ax = evec[:, 1]; ay = evec[:, 0]
            if a_len / b_len < 1.18:
                r = (a_len + b_len) / 2
                out = c + np.stack([np.cos(t), np.sin(t)], 1) * r
                return ShapeResult("circle", out, cen, {"r": float(r)})
            out = c + np.outer(np.cos(t) * a_len, ax) + np.outer(np.sin(t) * b_len, ay)
            return ShapeResult("ellipse", out, cen, {"a": float(a_len), "b": float(b_len)})
        # star
        star = ShapeRecognizer._star(R, c)
        if star is not None:
            return ShapeResult("star", star, cen, {"points": len(star) // 2})
        if n == 4:                                              # generic quadrilateral
            return ShapeResult("quad", C, tuple(C.mean(0)), {})
        O = ShapeRecognizer._smooth(ShapeRecognizer.resample(R, 32, closed=True), True)
        return ShapeResult("closed", O, tuple(O.mean(0)), {"area": area})

    @staticmethod
    def _star(R, c) -> Optional[np.ndarray]:
        Rs = ShapeRecognizer.resample(R, 192, closed=True)
        d = Rs - c
        r = np.hypot(d[:, 0], d[:, 1]); th = np.arctan2(d[:, 1], d[:, 0])
        k = np.ones(9) / 9.0
        rp = np.convolve(np.concatenate([r[-8:], r, r[:8]]), k, mode="same")[8:-8]
        win = 12; mean = rp.mean()
        peaks = [i for i in range(len(rp)) if rp[i] > 1.04 * mean and
                 rp[i] == max(rp[(i + j) % len(rp)] for j in range(-win, win + 1))]
        merged = []
        for i in peaks:
            if not merged or i - merged[-1] > win: merged.append(i)
        if len(merged) >= 2 and merged[0] + len(rp) - merged[-1] <= win: merged.pop()
        kpk = len(merged)
        if not 4 <= kpk <= 8: return None
        valleys = []
        for a, b in zip(merged, merged[1:] + [merged[0] + len(rp)]):
            seg = [rp[i % len(rp)] for i in range(a, b + 1)]
            valleys.append(min(seg))
        ro = float(np.mean([rp[i] for i in merged])); ri = float(np.mean(valleys))
        if ri / ro > 0.74: return None
        ang0 = float(th[merged[0]])
        pts = []
        for j in range(kpk):
            a = ang0 + math.tau * j / kpk; a2 = a + math.pi / kpk
            pts.append(c + ro * np.array([math.cos(a), math.sin(a)]))
            pts.append(c + ri * np.array([math.cos(a2), math.sin(a2)]))
        return np.array(pts)

    @staticmethod
    def generic(pts) -> Optional[ShapeResult]:
        """Fallback for strokes recognize() ignores: tiny closed -> blob, otherwise tube."""
        P = np.asarray(pts, float)
        if len(P) < 3: return None
        L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
        if L < 6: return None
        O = ShapeRecognizer._smooth(ShapeRecognizer.resample(P, 36), False)
        return ShapeResult("open", O, tuple(O.mean(0)), {"length": L})


# ───────────────────────────── mesh builders (local frame) ─────────────────────────────
def _inflate(outline, depth: float, K: int = 8):
    """Round solid: stacked, shrinking copies of the outline (circle -> sphere)."""
    M = len(outline); th = np.linspace(-math.pi / 2, math.pi / 2, K + 1)
    V, rings = [], []
    for k in range(1, K):
        c, z = math.cos(th[k]), depth * math.sin(th[k]); base = len(V)
        for p in outline: V.append((p[0] * c, p[1] * c, z))
        rings.append(list(range(base, base + M)))
    iB = len(V); V.append((0, 0, -depth)); iF = len(V); V.append((0, 0, depth))
    F = []
    for r in range(len(rings) - 1):
        a, b = rings[r], rings[r + 1]
        for i in range(M): F.append((a[i], a[(i + 1) % M], b[(i + 1) % M], b[i]))
    for i in range(M):
        F.append((iB, rings[0][(i + 1) % M], rings[0][i]))
        F.append((iF, rings[-1][i], rings[-1][(i + 1) % M]))
    ch = [ring + [ring[0]] for ring in rings]
    for i in range(0, M, max(1, M // 8)): ch.append([iB] + [ring[i] for ring in rings] + [iF])
    return V, ch, F

def _prism(poly, depth: float):
    M = len(poly); h = depth / 2
    V = [(x, y, -h) for x, y in poly] + [(x, y, h) for x, y in poly]
    F = [tuple(range(M)), tuple(range(2 * M - 1, M - 1, -1))]
    F += [(i, (i + 1) % M, M + (i + 1) % M, M + i) for i in range(M)]
    ch = [list(range(M)) + [0], list(range(M, 2 * M)) + [M]] + [[i, M + i] for i in range(M)]
    return V, ch, F

def _pyramid(tri, height: float):
    V = [(x, y, -height / 2) for x, y in tri] + [(0, 0, height / 2)]
    F = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    return V, [[0, 1, 2, 0], [0, 3], [1, 3], [2, 3]], F

def _star3d(star, depth: float):
    M = len(star)
    V = [(x, y, 0) for x, y in star] + [(0, 0, depth), (0, 0, -depth)]
    f, b = M, M + 1
    F = []
    for i in range(M): F += [(i, (i + 1) % M, f), ((i + 1) % M, i, b)]
    ch = [list(range(M)) + [0]]
    for i in range(0, M, 2): ch += [[f, i], [b, i]]
    return V, ch, F

def _cylinder(p0, p1, radius: float, seg: int = 14):
    a = np.array([p1[0] - p0[0], p1[1] - p0[1], 0.0]); Lc = float(np.linalg.norm(a)) or 1.0
    u = a / Lc; n = np.array([-u[1], u[0], 0.0]); b = np.array([0.0, 0.0, 1.0])
    mid = np.array([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, 0.0])
    ends = [np.array([p0[0], p0[1], 0.0]) - mid, np.array([p1[0], p1[1], 0.0]) - mid]
    V = []
    for e in ends:
        for j in range(seg):
            t = math.tau * j / seg; V.append(tuple(e + radius * (math.cos(t) * n + math.sin(t) * b)))
    F = [tuple(range(seg)), tuple(range(2 * seg - 1, seg - 1, -1))]
    F += [(j, (j + 1) % seg, seg + (j + 1) % seg, seg + j) for j in range(seg)]
    ch = [list(range(seg)) + [0], list(range(seg, 2 * seg)) + [seg]] + [[j, seg + j] for j in range(0, seg, 2)]
    return V, ch, F

def _tube(path, radius: float, seg: int = 8):
    P = np.asarray(path, float); M = len(P)
    T = np.gradient(P, axis=0); T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    V = []
    for i in range(M):
        n = np.array([-T[i, 1], T[i, 0], 0.0]); b = np.array([0.0, 0.0, 1.0])
        for j in range(seg):
            t = math.tau * j / seg
            V.append((P[i, 0] + radius * (math.cos(t) * n[0] + math.sin(t) * b[0]),
                      P[i, 1] + radius * (math.cos(t) * n[1] + math.sin(t) * b[1]),
                      radius * math.sin(t)))
    F = [(i * seg + j, i * seg + (j + 1) % seg, (i + 1) * seg + (j + 1) % seg, (i + 1) * seg + j)
         for i in range(M - 1) for j in range(seg)]
    ch = [[i * seg + j for i in range(M)] for j in range(0, seg, 2)]
    for i in range(0, M, 7): ch.append([i * seg + j for j in range(seg)] + [i * seg])
    return V, ch, F


def build_solid(res: ShapeResult, rgb, brush: str, width: int, view: View3D, z: float = 0.0) -> Solid3D:
    """Idealised 3-D version of a recognised (or generic) 2-D shape, placed where it was drawn
    and oriented to face the current view, so the morph starts exactly on the drawn outline."""
    k = res.kind; cx, cy = res.center
    O = np.asarray(res.outline, float) - np.array([cx, cy])
    if k in ("circle", "ellipse", "closed"):
        O = ShapeRecognizer.resample(O, 32, closed=True) if k == "closed" else O
        if k == "circle": depth = res.params["r"]
        elif k == "ellipse": depth = (res.params["a"] + res.params["b"]) / 2
        else: depth = 0.7 * math.sqrt(max(res.params.get("area", 1.0), 1.0) / math.pi)
        V, ch, F = _inflate(O, max(depth, 6.0))
        kind = {"circle": "sphere", "ellipse": "ellipsoid", "closed": "blob"}[k]
    elif k in ("square", "rectangle"):
        V, ch, F = _prism(O, res.params["side"] if k == "square" else min(res.params["w"], res.params["h"]))
        kind = "cube" if k == "square" else "box"
    elif k == "quad":
        side = float(np.mean(np.hypot(*(O - np.roll(O, 1, 0)).T)))
        V, ch, F = _prism(O, 0.6 * side); kind = "prism"
    elif k == "triangle":
        side = float(np.mean(np.hypot(*(O - np.roll(O, 1, 0)).T)))
        V, ch, F = _pyramid(O, 0.85 * side); kind = "pyramid"
    elif k == "star":
        V, ch, F = _star3d(O, 0.55 * float(np.max(np.hypot(*O.T)))); kind = "star3d"
    elif k == "line":
        V, ch, F = _cylinder(O[0], O[1], max(8.0, width * 1.6)); kind = "cylinder"
    else:
        V, ch, F = _tube(O, max(5.0, width * 1.6)); kind = "tube"
    V = np.asarray(V, float)
    s = Solid3D(kind, V, ch, F, rgb, brush, width, (0, 0, 0), name=res.solid_name)
    Vv = view.unproject(cx, cy, z)
    s.pos = view.to_world(Vv[None, :])[0]
    s.R = view.R.T.copy()
    return s


def _holo_ring(surface, cx, cy, r, col, label, spin, font):
    n = 28
    for k in range(n):
        a = spin + k * math.tau / n
        pygame.draw.circle(surface, col, (int(cx + math.cos(a) * r), int(cy + math.sin(a) * r)), 3 if k % 3 == 0 else 2)
    img = font.render(label, True, col)
    surface.blit(img, (int(cx - img.get_width() / 2), int(cy - r - 22)))


class Scene3D:
    """Holds every Solid3D: smooth launch/morph, lightweight 3-D physics, perspective rendering."""

    def __init__(self, phys: "PhysicsWorld"):
        self.phys = phys
        self.solids: List[Solid3D] = []
        self.view = View3D()
        self._ms_left = SOLID_REBUILD_MS
        self._grid = None; self._grid_key = None
        self.alpha_mul = 1.0                       # layer opacity (Physics Objects)
        pygame.font.init()
        self._font = pygame.font.SysFont("monospace", 15, bold=True)

    # ---------- bookkeeping ----------
    def add(self, s: Solid3D, morph: bool = True) -> Solid3D:
        if morph and s.kind != "stroke":
            s.morph = 0.0; s.tilt = (TILT_X, TILT_Y)
        while len(self.solids) >= MAX_SOLIDS:
            old = next((o for o in self.solids if not o.held), None)
            if old is None: break
            self.solids.remove(old)
        self.solids.append(s)
        return s

    def remove(self, s: Solid3D):
        if s in self.solids: self.solids.remove(s)

    def remove_uid(self, uid: int) -> bool:
        for s in self.solids:
            if s.uid == uid: self.solids.remove(s); return True
        return False

    def remove_last(self) -> Optional[Solid3D]:
        cand = [s for s in self.solids if not s.held and not s.building]
        if not cand: return None
        s = max(cand, key=lambda o: o.uid); self.solids.remove(s); return s

    def clear(self):
        self.solids.clear()

    def launch(self) -> int:
        """Smooth launch: physics wakes up with gravity fading in (no sudden jolt)."""
        n = 0
        for s in self.solids:
            if s.building or s.held or s.active or s.frozen: continue
            s.active = True; s.launch_t = 0.0; n += 1
        return n

    # ---------- strokes drawn directly in 3-D mode ----------
    def begin_stroke(self, rgb, brush: str, width: int) -> Solid3D:
        s = Solid3D("stroke", np.zeros((0, 3)), [], [], rgb, brush, width, (0, 0, 0))
        s.building = True
        self.solids.append(s)
        return s

    def add_point(self, s: Solid3D, sx: float, sy: float, z: float) -> bool:
        P = self.view.to_world(self.view.unproject(sx, sy, z)[None, :])[0]
        if s.pts and np.linalg.norm(P - s.pts[-1]) < 1.5: return False
        s.pts.append(P); s.V = np.array(s.pts)
        s.chains = [np.arange(len(s.pts))]
        return True

    def finish_stroke(self, s: Solid3D) -> bool:
        if len(s.pts) < 2 or float(np.sum(np.linalg.norm(np.diff(np.array(s.pts), axis=0), axis=1))) < 10.0:
            self.remove(s); return False
        V = np.array(s.pts); c = V.mean(axis=0)
        s.V = V - c; s.pos = c.copy(); s.chains = [np.arange(len(V))]
        s.building = False; s.pts = []; s.prep()
        return True

    # ---------- projection / picking ----------
    def project_solid(self, s: Solid3D):
        Vv = self.view.to_view(s.world_verts())
        sx, sy, k = self.view.project(Vv)
        return sx, sy, k, Vv[:, 2]

    def center_screen(self, s: Solid3D):
        cv = self.view.to_view(s.pos[None, :])
        sx, sy, k = self.view.project(cv)
        return float(sx[0]), float(sy[0]), float(k[0]), float(cv[0, 2])

    def pick(self, pt, radius: float):
        best, bd = None, float(radius)
        for s in self.solids:
            if s.building or len(s.V) == 0 or not s.visible: continue
            sx, sy, _, _ = self.project_solid(s)
            d = float(np.sqrt((sx - pt[0]) ** 2 + (sy - pt[1]) ** 2).min())
            if d < bd: best, bd = s, d
        return best, bd

    def pick_flying(self, pt, radius: float):
        """Nearest solid that is in flight (thrown) - used for catching."""
        best, bd = None, float(radius)
        for s in self.solids:
            if s.building or s.held or not s.active or len(s.V) == 0 or not s.visible: continue
            if float(np.linalg.norm(s.vel)) < CATCH_MIN_SPEED: continue
            sx, sy, _, _ = self.project_solid(s)
            d = float(np.sqrt((sx - pt[0]) ** 2 + (sy - pt[1]) ** 2).min())
            if d < bd: best, bd = s, d
        return best

    # ---------- physics ----------
    def explode(self, cx: float, cy: float):
        for s in self.solids:
            if s.held or s.building or s.frozen: continue
            sx, sy, _, _ = self.center_screen(s)
            dx, dy = sx - cx, sy - cy; d = max(math.hypot(dx, dy), 1.0)
            m = min(2500.0, EXPLODE_FORCE / d / 3.0 / max(0.25, s.mass))
            s.vel += np.array([dx / d * m, dy / d * m, (random.random() - 0.5) * m * 0.4])
            s.ang += np.array([random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3)])
            if not s.active: s.active = True; s.launch_t = LAUNCH_RAMP_S

    def step(self, dt: float):
        self._ms_left = SOLID_REBUILD_MS
        dt = min(dt, 1 / 30.0)
        g = np.array([self.phys.space.gravity[0], self.phys.space.gravity[1], 0.0])
        for s in self.solids:
            if s.morph < 1.0:
                s.morph = min(1.0, s.morph + dt / MORPH_S)
                if s.morph >= 1.0 and s.kind != "stroke": s.finish_morph()
        live = [s for s in self.solids if not s.building and not s.held and s.active and not s.frozen and s.morph >= 1.0]
        for s in live:
            s.launch_t += dt
            gs = _smootherstep(s.launch_t / LAUNCH_RAMP_S)
            s.vel += g * (gs * s.gscale) * dt
            s.vel *= max(0.0, 1.0 - SOLID_AIR_DRAG * dt)
            s.pos += s.vel * dt
            w = float(np.linalg.norm(s.ang))
            if w > 1e-6:
                s.R = _orthonormalize(_axis_angle(s.ang, w * dt) @ s.R)
                s.ang *= max(0.0, 1.0 - SOLID_ANG_DRAG * dt)
            self._bounds(s)
        for s in self.solids:                                  # optional idle spin for unlaunched solids
            if IDLE_SPIN and not s.active and not s.held and not s.building and s.morph >= 1.0:
                s.R = _axis_angle(np.array([0.0, 1.0, 0.0]), IDLE_SPIN * dt) @ s.R
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                self._collide(live[i], live[j])

    def _bounds(self, s: Solid3D):
        W = s.world_verts()
        lo, hi = W.min(axis=0), W.max(axis=0)
        lim_lo = np.array([4.0, 4.0, -Z_RANGE]); lim_hi = np.array([WIN_W - 4.0, WIN_H - 4.0, Z_RANGE])
        for ax in range(3):
            push = 0.0
            if lo[ax] < lim_lo[ax]: push = lim_lo[ax] - lo[ax]
            elif hi[ax] > lim_hi[ax]: push = lim_hi[ax] - hi[ax]
            if push:
                s.pos[ax] += push
                if (push > 0 and s.vel[ax] < 0) or (push < 0 and s.vel[ax] > 0):
                    s.vel[ax] = -s.vel[ax] * s.bounce
                    if abs(s.vel[ax]) < 25.0: s.vel[ax] = 0.0
                    for o in range(3):
                        if o != ax: s.vel[o] *= max(0.0, 1.0 - 0.015 * s.fric)      # contact friction
                    s.ang *= 0.93

    def _collide(self, a: Solid3D, b: Solid3D):
        d = b.pos - a.pos; dist = float(np.linalg.norm(d))
        ra, rb = a.radius * a.scale * 0.85, b.radius * b.scale * 0.85
        if dist >= ra + rb or dist < 1e-6: return
        n = d / dist; over = ra + rb - dist
        a.pos -= n * over * 0.5; b.pos += n * over * 0.5
        vn = float(np.dot(b.vel - a.vel, n))
        if vn < 0:
            e = 0.5 * (a.bounce + b.bounce)
            j = -(1 + e) * vn / (1.0 / a.mass + 1.0 / b.mass)             # heavier solids push lighter ones around
            a.vel -= n * (j / a.mass); b.vel += n * (j / b.mass)
            t = (b.vel - a.vel) - n * np.dot(b.vel - a.vel, n)
            a.ang += np.cross(n, t) * 0.004; b.ang += np.cross(n, t) * 0.004

    # ---------- ambient particle emission for fire / ice / galaxy ... brushes ----------
    def emit_fx(self, fx: "FXSystem", rng: random.Random, dt: float):
        budget = max(2, FX_AMBIENT_BUDGET // 2)
        for s in self.solids:
            br = BRUSHES.get(s.brush)
            if budget <= 0: break
            if not br or not br.emits or s._psx is None or len(s._psx) == 0: continue
            if rng.random() < br.amb_rate * min(3.0, max(0.5, s.radius / 60.0)) * dt:
                i = rng.randrange(len(s._psx))
                br.emit(fx, (float(s._psx[i]), float(s._psy[i])), s.rgb, s.width, rng, 0.0)
                budget -= 1

    # ---------- rendering ----------
    def render_grid(self, surface: pygame.Surface):
        key = self.view.key
        if self._grid is None or self._grid_key != key:
            lay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA); yf = WIN_H - 3.0
            def seg(a, b):
                V = self.view.to_view(np.array([a, b], float)); sx, sy, k = self.view.project(V)
                al = int(40 + 100 * max(0.0, min(1.0, float(k.mean()) / 1.7)))
                p = [(int(max(-4000, min(4000, sx[i]))), int(max(-4000, min(4000, sy[i])))) for i in (0, 1)]
                pygame.draw.line(lay, (70, 210, 240, al), p[0], p[1], 1)
            for z in np.arange(-Z_RANGE, Z_RANGE + 1, 95.0): seg((0, yf, z), (WIN_W, yf, z))
            for x in np.arange(0, WIN_W + 1, 107.0): seg((x, yf, -Z_RANGE), (x, yf, Z_RANGE))
            self._grid, self._grid_key = lay, key
        surface.blit(self._grid, (0, 0))

    def render(self, surface: pygame.Surface, now: Optional[float] = None, budget_ms: Optional[float] = None):
        now = time.perf_counter() if now is None else now
        if budget_ms is not None: self._ms_left = budget_ms        # adaptive: only what the rest of the frame left over
        order = []
        for s in self.solids:
            if len(s.V) == 0 or not s.visible: continue
            order.append((self.center_screen(s)[3], s))
        order.sort(key=lambda t: t[0])
        for _, s in order:
            try:
                self._render_solid(surface, s, now)
            except Exception:
                pass                                           # never let one bad solid crash the frame

    def _render_solid(self, surface, s: Solid3D, now: float):
        view = self.view
        Vv = view.to_view(s.world_verts())
        sx, sy, k = view.project(Vv); zv = Vv[:, 2]
        s._psx, s._psy = sx, sy
        cv = view.to_view(s.pos[None, :]); csx, csy, ck = (float(a[0]) for a in view.project(cv))
        Rtot = view.R @ s.morph_R()
        br = BRUSHES.get(s.brush) or BRUSHES[DEFAULT_BRUSH]
        bucket = int(now * br.anim_hz) if br.animated else 0
        moving = s.morph < 1.0 or s.held or float(np.linalg.norm(s.ang)) > 0.25
        Rk = np.round(Rtot / 0.035) * 0.035 if (moving and s.kind != "stroke") else np.round(Rtot, 3)   # ~2 deg steps while spinning
        key = (round(float(cv[0, 2]), 1), tuple(np.round(Rk, 3).ravel().tolist()),
               round(s.scale, 3), round(s.morph, 3), bucket, s.width, s.rgb, s.brush, len(s.V), round(s.glow, 2))
        stale = s._spr is not None and s._spr_key != key
        settle = s._spr is not None and s._lod and not moving                  # motion stopped -> redraw at full detail
        if s._spr is None or ((stale or settle) and (self._ms_left > 0 or s.building)):
            t0 = time.perf_counter()
            lod = s.kind != "stroke" and moving                                    # spinning / morphing -> lighter wireframe
            self._build_sprite(s, sx, sy, k, zv, Rtot, br, csx, csy, key, now, lod)
            s._spr_t, s._lod = now, lod
            self._ms_left -= (time.perf_counter() - t0) * 1000.0
        if s._spr is None: return
        dx, dy = csx - s._spr_c[0], csy - s._spr_c[1]
        s._spr.set_alpha(int(255 * max(0.45, min(1.0, 0.5 + 0.5 * ck)) * max(0.05, min(1.0, s.opacity * self.alpha_mul))))
        surface.blit(s._spr, (int(s._spr_org[0] + dx), int(s._spr_org[1] + dy)))
        if s.hover or s.selected:
            r = s.radius * s.scale * ck + 10 + 3 * math.sin(now * 5.0)
            col = (255, 235, 90) if s.selected else (90, 220, 255)
            txt = ("OBJECT SELECTED #%d  %s" % (s.uid, s.extra)).strip() if s.selected else "GRAB  " + s.name
            _holo_ring(surface, csx, csy, r, col, txt, now * (2.2 if s.selected else 1.0), self._font)

    def _build_sprite(self, s, sx, sy, k, zv, Rtot, br, csx, csy, key, now, lod=False):
        wpx = max(1.0, s.width * 0.7 * min(2.0, max(0.6, s.scale ** 0.5)))
        pad = int(wpx * 3.8 * float(k.max())) + 18
        M = 120
        x0 = max(int(math.floor(sx.min())) - pad, -M); y0 = max(int(math.floor(sy.min())) - pad, -M)
        x1 = min(int(math.ceil(sx.max())) + pad, WIN_W + M); y1 = min(int(math.ceil(sy.max())) + pad, WIN_H + M)
        bw, bh = x1 - x0, y1 - y0
        if bw < 4 or bh < 4:
            s._spr = None; return
        spr = pygame.Surface((bw, bh), pygame.SRCALPHA)
        if s.faces and SOLID_FACE_ALPHA > 0 and len(s.faces) <= 400 and not s.building:
            Nv = s.N @ Rtot.T
            dotl = Nv @ _LIGHT
            bright = 0.28 + 0.95 * (np.maximum(0.0, dotl) if s.cull else np.abs(dotl))
            vis = (Nv[:, 2] >= -0.02).tolist() if s.cull else None
            SX = (sx - x0).tolist(); SY = (sy - y0).tolist(); zl = zv.tolist(); bl = bright.tolist()
            order = sorted(((sum(zl[i] for i in f) / len(f), fi) for fi, f in enumerate(s.faces)
                            if vis is None or vis[fi]))
            base = _mix(s.rgb, (255, 255, 255), 0.12)
            fl = pygame.Surface((bw, bh), pygame.SRCALPHA)
            for _, fi in order:
                b = bl[fi]
                col = (min(255, int(base[0] * b)), min(255, int(base[1] * b)), min(255, int(base[2] * b)), 255)
                pygame.draw.polygon(fl, col, [(SX[i], SY[i]) for i in s.faces[fi]])
            fl.set_alpha(SOLID_FACE_ALPHA)
            spr.blit(fl, (0, 0))
        passes = br.passes(s.rgb)
        lay = StrokeLayers((bw, bh), (x0, y0), n=len(passes))
        px, py, kl = sx.tolist(), sy.tolist(), k.tolist()
        seed = s.uid * 7919 + (int(now * br.anim_hz) if br.animated else 0)
        phase = now * 6.0 if br.animated else 0.0
        for ci, ch in enumerate(s.chains):
            idx = ch.tolist()
            if len(idx) < 2: continue
            if lod and len(idx) > 28:                                       # moving: every 2nd point is plenty
                idx = idx[::2] + ([idx[-1]] if (len(idx) - 1) % 2 else [])
            ctx = br.new_ctx(s.rgb, int(wpx), (px[idx[0]], py[idx[0]]), seed + ci, phase)
            for i in idx[1:]:
                lay.draw(br.feed(ctx, (px[i], py[i]), max(1, int(round(wpx * kl[i])))), ctx.passes)
        lay.render(spr, passes)
        if s.glow > 0:
            spr, gp = add_glow(spr, s.glow); x0 -= gp; y0 -= gp
        s._spr, s._spr_org, s._spr_c, s._spr_key = spr, (x0, y0), (csx, csy), key


# ───────────────────────────── depth from the hand ─────────────────────────────
class DepthEstimator:
    """Approximate interaction depth from hand size (palm grows when the hand nears the camera)."""
    def __init__(self): self.reset()

    def reset(self):
        self.ref: Optional[float] = None
        self.size_s: Optional[float] = None
        self.ratio = 1.0; self.z = 0.0; self.manual = 0.0

    def update(self, size: float, dt: float = 1 / 60) -> float:
        if DEPTH_METHOD == "keyboard":
            self.z = max(-Z_RANGE, min(Z_RANGE, self.manual)); return self.z
        if size <= 0: return self.z
        self.size_s = size if self.size_s is None else self.size_s + DEPTH_SMOOTH * (size - self.size_s)
        if self.ref is None: self.ref = self.size_s
        self.ratio = self.size_s / self.ref
        d = self.ratio - 1.0
        d = 0.0 if abs(d) < DEPTH_DEADBAND else d - math.copysign(DEPTH_DEADBAND, d)
        self.z = max(-Z_RANGE, min(Z_RANGE, d * DEPTH_SENSITIVITY))
        return self.z

    def nudge(self, dz: float):
        self.manual = max(-Z_RANGE, min(Z_RANGE, self.manual + dz))


# ───────────────────────────── "turn it into 3-D" offer ─────────────────────────────
@dataclass
class Suggestion:
    result: ShapeResult
    pts: List
    rgb: Tuple[int, int, int]
    w: int
    brush: str
    src: str                      # "2d" (ink on the canvas) | "3d" (a stroke solid)
    z: float = 0.0
    ref: object = None            # pending-list identity (2d) or solid uid (3d)


TimelinePanel_TAB = pygame.Rect(20, 64, 100, 32)       # leftmost tab (kept in sync with TimelinePanel)


class ConvertChip:
    """Floating 'Circle  ->  SPHERE' offer.  Hold a fingertip on it 0.5 s, press V, or click."""
    def __init__(self):
        pygame.font.init()
        self._f = pygame.font.SysFont("monospace", 15, bold=True)
        self._s = pygame.font.SysFont("monospace", 11)
        self.visible = False
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.text = ""; self._born = 0.0; self._ht = 0.0; self._latched = False

    def show(self, res: ShapeResult, pts):
        P = np.asarray(pts, float)
        self.text = f"{res.shape_name} detected  ->  {res.solid_name.upper()}"
        w = self._f.size(self.text)[0] + 70; h = 32
        cx = (P[:, 0].min() + P[:, 0].max()) / 2
        x = int(max(8, min(WIN_W - w - 8, cx - w / 2)))
        y = int(P[:, 1].min()) - 46
        if y < 114: y = int(P[:, 1].max()) + 16
        y = max(114, min(WIN_H - 120, y))
        self.rect = pygame.Rect(x, y, w, h)
        if self.rect.colliderect(EffectPanel.TAB.union(BrushPanel.TAB).union(TimelinePanel_TAB).inflate(10, 10)):
            self.rect.right = TimelinePanel_TAB.left - 12
        self.visible, self._born, self._ht, self._latched = True, time.perf_counter(), 0.0, False

    def hide(self):
        self.visible = False; self._ht = 0.0

    def blocks(self, pt) -> bool:
        return bool(pt) and self.visible and self.rect.inflate(16, 16).collidepoint(pt)

    def click(self, pos) -> bool:
        return self.visible and self.rect.collidepoint(pos)

    def update(self, tip, dt: float) -> bool:
        if not self.visible: return False
        if time.perf_counter() - self._born > CONVERT_CHIP_S:
            self.hide(); return False
        over = bool(tip) and self.rect.collidepoint(tip)
        if over and not self._latched:
            self._ht += dt
            if self._ht >= UI_HOVER_S:
                self._latched = True; return True
        elif not over:
            self._ht = 0.0; self._latched = False
        return False

    def render(self, surface: pygame.Surface):
        if not self.visible: return
        CY = (170, 120, 255)
        p = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        pygame.draw.rect(p, (10, 12, 28, 215), p.get_rect(), border_radius=12)
        pygame.draw.rect(p, (*CY, 130), (0, 0, int(self.rect.w * min(1.0, self._ht / UI_HOVER_S)), self.rect.h), border_radius=12)
        pygame.draw.rect(p, (*CY, 255), p.get_rect(), 2, border_radius=12)
        surface.blit(p, self.rect.topleft)
        surface.blit(self._f.render(self.text, True, (235, 235, 255)), (self.rect.x + 12, self.rect.y + 7))
        k = self._s.render("V", True, (170, 160, 210))
        surface.blit(k, (self.rect.right - k.get_width() - 10, self.rect.y + 10))


# ───────────────────────────── 3-D grab / move / rotate / scale / throw ─────────────────────────────
class Grab3DController:
    """Thumb+middle pinch on a 3-D solid.
         move   : pinch point (x,y)   |  hand toward/away from camera = object in depth (z)
         rotate : roll wrist (or turn two hands);  arrow keys = yaw / pitch
         scale  : both hands pinching, spread / close      ([ ] keys fallback)
         throw  : release while moving -> velocity + tumble spin."""

    def __init__(self, scene: Scene3D):
        self.scene = scene
        self.active = False
        self.hand: Optional[str] = None
        self.body: Optional[Solid3D] = None
        self.hover: Optional[Solid3D] = None
        self.second: Optional[str] = None
        self.info: List[str] = []
        self.key_scale = self.key_yaw = self.key_pitch = self.key_depth = 0
        self._sm: Optional[np.ndarray] = None
        self._was = {"Right": False, "Left": False}
        self._acq = {"Right": 0.0, "Left": 0.0}
        self._seen_t = 0.0
        self._reset_manip()

    def _reset_manip(self):
        self._hist: Deque = deque(maxlen=10)
        self._ang_last: Optional[float] = None
        self._ang_acc = self._ang_s = self._rot_off = self._rot_prev = 0.0
        self._rot_armed = False
        self._rot_hist: Deque = deque(maxlen=6)
        self._size0: Optional[float] = None; self._size_s = 0.0
        self._depth0 = 0.0; self._off = np.zeros(2)
        self._two = False; self._d0 = 1.0; self._s0 = 1.0; self._d_s = 1.0
        self._spin_prev = None
        self.info = []

    @property
    def owned(self) -> set:
        if not self.active: return set()
        return {self.hand} | ({self.second} if self.second else set())

    def mask_hands(self, hands: List[HandData]) -> List[HandData]:
        """Copies of `hands` where hands busy with a 3-D object no longer look like GRAB to the 2-D controller."""
        own = self.owned
        return [dataclasses.replace(h, gesture="IDLE") if h.label in own else h for h in hands]

    def _smooth(self, p) -> np.ndarray:
        p = np.array(p, float)
        self._sm = p if self._sm is None else GRAB_SMOOTH * p + (1 - GRAB_SMOOTH) * self._sm
        return self._sm.copy()

    def reset(self):
        if self.body is not None:
            self.body.held = False; self.body.selected = False; self.body.extra = ""
        self.active, self.hand, self.body, self.second, self._sm = False, None, None, None, None
        self._reset_manip()

    def release(self) -> str:
        s = self.body; view = self.scene.view
        v = np.zeros(3)
        if s is not None and len(self._hist) >= 2:
            t1, p1 = self._hist[-1]; i = len(self._hist) - 1
            while i > 0 and t1 - self._hist[i - 1][0] <= 0.14: i -= 1
            t0, p0 = self._hist[i]
            if t1 - t0 > 1e-3: v = (p1 - p0) / (t1 - t0)
        sp = float(np.linalg.norm(v))
        if sp > MAX_THROW_SPEED: v *= MAX_THROW_SPEED / sp
        if s is not None:
            roll = 0.0
            if self._rot_armed and len(self._rot_hist) >= 2:
                (t0, a0), (t1, a1) = self._rot_hist[0], self._rot_hist[-1]
                if t1 - t0 > 1e-3: roll = (a1 - a0) / (t1 - t0)
            vv = view.R @ v
            tumble = np.array([-vv[1], vv[0], 0.0]) / max(s.radius * s.scale, 25.0) * 0.5
            w = view.R.T @ tumble + (view.R.T @ np.array([0.0, 0.0, 1.0])) * roll
            wn = float(np.linalg.norm(w))
            if wn > MAX_SPIN: w *= MAX_SPIN / wn
            s.vel, s.ang = v, w
            s.held = s.selected = False; s.extra = ""
            s.active = True; s.launch_t = LAUNCH_RAMP_S * 0.35
        self.active, self.hand, self.body, self.second, self._sm = False, None, None, None, None
        self._reset_manip()
        return "release"

    def _start(self, s: Solid3D, h: HandData):
        self.active, self.hand, self.body = True, h.label, s
        s.held = s.selected = True; s.vel[:] = 0; s.ang[:] = 0
        self._sm = None; self._reset_manip(); self._seen_t = time.perf_counter()
        P0 = self._smooth(h.grab_pt)
        csx, csy, ck, zc = self.scene.center_screen(s)
        self._depth0 = zc; self._off = np.array([csx - P0[0], csy - P0[1]]) / ck
        self._size0 = h.size if h.size > 0 else None; self._size_s = h.size
        self._s0 = s.scale
        self._hist.append((time.perf_counter(), s.pos.copy()))

    def _transfer(self, h: HandData):
        """Hand-off: the other hand is already pinching -> it takes the object, no throw, no jump."""
        s = self.body; sc = self.scene
        keep = s.scale
        self.hand, self.second, self._sm = h.label, None, None
        self._reset_manip()
        P0 = self._smooth(h.grab_pt)
        csx, csy, ck, zc = sc.center_screen(s)
        self._depth0 = zc; self._off = np.array([csx - P0[0], csy - P0[1]]) / ck
        self._size0 = h.size if h.size > 0 else None; self._size_s = h.size
        self._s0 = keep
        s.vel[:] = 0; s.ang[:] = 0
        self._hist.append((time.perf_counter(), s.pos.copy()))

    def _manipulate(self, h: HandData, other: Optional[HandData], dt: float, now: float, spin: Optional[HandData] = None):
        s, view = self.body, self.scene.view
        P = self._smooth(h.grab_pt)
        # ---- depth from hand size ----
        d = 0.0
        if self._size0 and h.size > 0:
            self._size_s += 0.3 * (h.size - self._size_s)
            d = self._size_s / self._size0 - 1.0
            d = 0.0 if abs(d) < DEPTH_DEADBAND else d - math.copysign(DEPTH_DEADBAND, d)
        self._depth0 += self.key_depth * 300.0 * dt
        zv = max(-Z_RANGE, min(Z_RANGE, self._depth0 + d * DEPTH_SENSITIVITY))
        k = float(view.persp(np.array([zv]))[0])
        tgt = P + self._off * k
        target = view.to_world(view.unproject(float(tgt[0]), float(tgt[1]), zv)[None, :])[0]
        s.pos += (target - s.pos) * (1.0 - math.exp(-dt * 45.0))
        self._hist.append((now, s.pos.copy()))

        # ---- two-hand scale + rotation source ----
        two = ENABLE_GRAB_SCALE and other is not None and other.grab_pt is not None
        self.second = other.label if two else None
        if two:
            dd = math.hypot(other.grab_pt[0] - h.grab_pt[0], other.grab_pt[1] - h.grab_pt[1])
            if not self._two:
                self._d0, self._s0, self._d_s = max(dd, 40.0), s.scale, dd; self._ang_last = None
            self._d_s = 0.5 * dd + 0.5 * self._d_s
            ratio = self._d_s / self._d0; db = 1.0 + SCALE3D_DEADBAND
            r = ratio / db if ratio > db else (ratio * db if ratio < 1.0 / db else 1.0)
            s.scale = max(SCALE3D_MIN, min(SCALE3D_MAX, self._s0 * r))
            src = math.atan2(other.grab_pt[1] - h.grab_pt[1], other.grab_pt[0] - h.grab_pt[0])
        else:
            if self._two: self._ang_last = None
            src = h.angle
        self._two = two
        if self.key_scale:
            s.scale = max(SCALE3D_MIN, min(SCALE3D_MAX, s.scale * (1.0 + SCALE_KEY_STEP * self.key_scale)))

        # ---- roll (about the view axis) ----
        if ENABLE_GRAB_ROTATION:
            if self._ang_last is not None: self._ang_acc += _wrap_angle(src - self._ang_last)
            self._ang_last = src
            self._ang_s += ROT_SMOOTH * (self._ang_acc - self._ang_s)
            if not self._rot_armed and abs(self._ang_s) > math.radians(ROT_DEADZONE_DEG):
                self._rot_armed, self._rot_off = True, self._ang_s
            eff = (self._ang_s - self._rot_off) * ROT_GAIN if self._rot_armed else 0.0
            dr = eff - self._rot_prev; self._rot_prev = eff
            if abs(dr) > 1e-7:
                s.R = _orthonormalize(_axis_angle(view.R.T @ np.array([0.0, 0.0, 1.0]), dr) @ s.R)
            self._rot_hist.append((now, eff))
        # ---- free hand's pointing finger = trackball spin ----
        if spin is not None and spin.tip is not None:
            if self._spin_prev is not None:
                dx, dy = spin.tip[0] - self._spin_prev[0], spin.tip[1] - self._spin_prev[1]
                if dx:
                    s.R = _orthonormalize(_axis_angle(view.R.T @ np.array([0.0, 1.0, 0.0]), dx * FREE_FINGER_SPIN) @ s.R)
                if dy:
                    s.R = _orthonormalize(_axis_angle(view.R.T @ np.array([1.0, 0.0, 0.0]), -dy * FREE_FINGER_SPIN) @ s.R)
            self._spin_prev = spin.tip
        else:
            self._spin_prev = None
        # ---- yaw / pitch from the keyboard ----
        if self.key_yaw:
            s.R = _orthonormalize(_axis_angle(view.R.T @ np.array([0.0, 1.0, 0.0]), self.key_yaw * 1.8 * dt) @ s.R)
        if self.key_pitch:
            s.R = _orthonormalize(_axis_angle(view.R.T @ np.array([1.0, 0.0, 0.0]), self.key_pitch * 1.8 * dt) @ s.R)

        bits = []
        if self._rot_armed: bits.append("rot %+.0f" % math.degrees(self._rot_prev))
        if abs(s.scale - 1.0) > 0.02: bits.append("x%.2f" % s.scale)
        bits.append("z %+.0f" % zv)
        if two: bits.append("2-HAND")
        s.extra = " ".join(bits)
        self.info = ["HOLD  : %s hand" % self.hand,
                     "Z     : %+6.0f  (hand size x%.2f)" % (zv, 1.0 + d),
                     "SCALE : x%.2f  %s" % (s.scale, "2-hand" if two else ""),
                     "ROLL  : %+6.1f deg %s" % (math.degrees(self._rot_prev), "(armed)" if self._rot_armed else "(deadzone)")]

    def update(self, hands: List[HandData], dt: float = 1 / 60, now: Optional[float] = None) -> Optional[str]:
        now = time.perf_counter() if now is None else now
        sc = self.scene
        grabbers = {h.label: h for h in hands if h.gesture == "GRAB" and h.grab_pt}
        for lab in ("Right", "Left"):
            if lab in grabbers and not self._was[lab]: self._acq[lab] = now + GRAB_ACQUIRE_S
            self._was[lab] = lab in grabbers
        event = None
        if self.active:
            h = grabbers.get(self.hand)
            other = grabbers.get("Left" if self.hand == "Right" else "Right")
            if self.body not in sc.solids:
                event = self.release()
            elif h is None and other is not None and sc.pick(other.grab_pt, HANDOFF_RADIUS)[0] is self.body:
                self._transfer(other); event = "handoff"           # other hand is on the object: pass it over
            elif h is None and not any(x.label == self.hand for x in hands) and now - self._seen_t < GRAB_HOLD_GRACE:
                pass                                               # tracking blip: keep holding, don't drop it
            elif h is None:
                event = self.release()                             # otherwise a normal throw (the other hand may catch it)
            else:
                self._seen_t = now
                spin = next((x for x in hands if x.label != self.hand and x.gesture == "DRAW" and x.tip), None)
                self._manipulate(h, other, dt, now, spin)
        if not self.active:
            for lab in ("Right", "Left"):
                h = grabbers.get(lab)
                if h and now <= self._acq[lab]:
                    s, _ = sc.pick(h.grab_pt, GRAB_SELECT_RADIUS)
                    if s is not None:
                        flying = s.active and float(np.linalg.norm(s.vel)) >= CATCH_MIN_SPEED
                        self._start(s, h); event = "catch" if flying else "grab"; break
        if not self.active:                                          # catching: a pinching hand meets a thrown solid
            for lab in ("Right", "Left"):
                h = grabbers.get(lab)
                if h is None: continue
                s = sc.pick_flying(h.grab_pt, CATCH_RADIUS)
                if s is not None:
                    self._start(s, h); event = "catch"; break
        for s in sc.solids: s.hover = False
        self.hover = None
        if not self.active:
            for h in hands:
                if h.grab_pt and h.tm is not None and h.tm < GRAB_HOVER_THRESHOLD and h.gesture in ("IDLE", "GRAB"):
                    s, _ = sc.pick(h.grab_pt, GRAB_SELECT_RADIUS)
                    if s is not None:
                        s.hover = True; self.hover = s; break
            if self.hover is not None and self.key_scale:
                self.hover.scale = max(SCALE3D_MIN, min(SCALE3D_MAX, self.hover.scale * (1.0 + SCALE_KEY_STEP * self.key_scale)))
        return event


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — PARTICLE EFFECTS + ENERGY RING
#  11 effects run on the FXSystem above (position, velocity, acceleration, size,
#  lifetime, opacity, rotation, colour).  An effect can be attached to a drawing
#  (not yet launched), a launched physics body, or a 3-D solid; on a drawn
#  CIRCLE it upgrades to an animated energy ring.
# ══════════════════════════════════════════════════════════════════════════════
EFFECT_BUDGET      = 22        # max effect emissions per frame (all targets together)
FX_RING_MAX        = 4         # energy rings drawn at once (each costs a glow layer)
RING_MAX_RADIUS    = 330
EFFECT_ORDER = ["none", "fire", "smoke", "sparks", "snow", "stars", "electricity",
                "plasma", "dust", "magic", "confetti", "bubbles"]


def _hsv(h: float, s: float = 0.7, v: float = 1.0) -> Tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _bolt_pts(rng: random.Random, p0, p1, jitter: float, n: int = 5):
    pts = [p0]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    for i in range(1, n):
        t = i / n; j = rng.uniform(-jitter, jitter)
        pts.append((p0[0] + dx * t + nx * j, p0[1] + dy * t + ny * j))
    pts.append(p1)
    return pts


# ---- the 11 emitters:  f(fx, point, outward_normal_or_None, rng) ----------------------------
def _e_fire(fx, p, n, rng):
    for _ in range(2):
        fx.spawn(x=p[0] + rng.gauss(0, 3), y=p[1] + rng.gauss(0, 3), vx=rng.gauss(0, 16), vy=-rng.uniform(55, 130),
                 ay=-20, size=rng.uniform(4, 7), size_end=1, life=rng.uniform(0.5, 1.0),
                 c0=(255, 225, 130), c1=(220, 50, 10), a0=0.95, turb=26, kind="glow")

def _e_smoke(fx, p, n, rng):
    fx.spawn(x=p[0] + rng.gauss(0, 4), y=p[1], vx=rng.gauss(0, 10), vy=-rng.uniform(15, 40), size=6, size_end=17,
             life=rng.uniform(1.4, 2.4), c0=(155, 155, 160), c1=(60, 60, 66), a0=0.38, turb=14, kind="soft")

def _e_sparks(fx, p, n, rng):
    for _ in range(3):
        a = rng.uniform(0, math.tau) if n is None else math.atan2(n[1], n[0]) + rng.gauss(0, 0.7)
        sp = rng.uniform(120, 300)
        fx.spawn(x=p[0], y=p[1], vx=math.cos(a) * sp, vy=math.sin(a) * sp, ay=420, drag=0.7, size=1.5, size_end=1,
                 life=rng.uniform(0.35, 0.8), c0=(255, 245, 180), c1=(255, 110, 25), a0=1.0, kind="spark")

def _e_snow(fx, p, n, rng):
    fx.spawn(x=p[0] + rng.gauss(0, 6), y=p[1] + rng.gauss(0, 3), vx=rng.gauss(0, 8), vy=rng.uniform(25, 55), size=2.4,
             size_end=2.0, life=rng.uniform(2.0, 3.5), c0=(238, 247, 255), c1=(185, 212, 245), a0=0.95, turb=22, kind="soft")

def _e_stars(fx, p, n, rng):
    fx.spawn(x=p[0] + rng.gauss(0, 5), y=p[1] + rng.gauss(0, 5), vx=rng.gauss(0, 10), vy=-rng.uniform(5, 25), size=rng.uniform(3, 5),
             size_end=1.5, life=rng.uniform(0.8, 1.6), c0=(255, 240, 160), c1=(255, 175, 55), a0=1.0, kind="star")

def _e_electricity(fx, p, n, rng):
    a = rng.uniform(0, math.tau) if n is None else math.atan2(n[1], n[0]) + rng.gauss(0, 0.9)
    L = rng.uniform(16, 42)
    q = (p[0] + math.cos(a) * L, p[1] + math.sin(a) * L)
    fx.spawn(x=p[0], y=p[1], size=2, size_end=2, life=rng.uniform(0.07, 0.16), c0=(210, 232, 255), c1=(80, 120, 255),
             a0=1.0, kind="bolt", pts=_bolt_pts(rng, p, q, 6.0, 4))
    if rng.random() < 0.6:
        fx.spawn(x=q[0], y=q[1], vx=rng.gauss(0, 30), vy=rng.gauss(0, 30), size=3, size_end=1, life=0.25,
                 c0=(200, 225, 255), c1=(60, 100, 255), a0=1.0, kind="glow")

def _e_plasma(fx, p, n, rng):
    for k, (c0, c1) in enumerate((((255, 120, 255), (120, 40, 200)), ((120, 255, 255), (40, 120, 255)))):
        fx.spawn(x=p[0], y=p[1], size=rng.uniform(3, 5), size_end=1, life=rng.uniform(0.7, 1.4), c0=c0, c1=c1, a0=0.95,
                 kind="glow", orbit=(p[0], p[1], rng.uniform(5, 18), rng.uniform(0, math.tau), rng.choice((-1, 1)) * rng.uniform(4, 9)))

def _e_dust(fx, p, n, rng):
    fx.spawn(x=p[0] + rng.gauss(0, 5), y=p[1] + rng.gauss(0, 5), vx=rng.gauss(0, 12), vy=rng.gauss(0, 8), size=1.6, size_end=2.8,
             life=rng.uniform(1.5, 3.0), c0=(192, 172, 142), c1=(120, 105, 85), a0=0.55, turb=10, kind="soft")

def _e_magic(fx, p, n, rng):
    h = rng.random()
    fx.spawn(x=p[0] + rng.gauss(0, 4), y=p[1], vx=rng.gauss(0, 12), vy=-rng.uniform(25, 60), size=rng.uniform(3.5, 5.5), size_end=1.2,
             life=rng.uniform(0.9, 1.7), c0=_hsv(h, 0.55, 1.0), c1=_hsv(h + 0.15, 0.9, 0.7), a0=1.0, turb=30,
             kind="star" if rng.random() < 0.5 else "glow")

def _e_confetti(fx, p, n, rng):
    fx.spawn(x=p[0], y=p[1], vx=rng.gauss(0, 40), vy=-rng.uniform(30, 90), ay=140, drag=0.5, size=rng.uniform(3, 5), size_end=3,
             life=rng.uniform(1.6, 2.6), c0=_hsv(rng.random(), 0.85, 1.0), c1=(255, 255, 255), a0=1.0, kind="confetti",
             rot=rng.uniform(0, math.tau), vrot=rng.uniform(-8, 8))

def _e_bubbles(fx, p, n, rng):
    fx.spawn(x=p[0] + rng.gauss(0, 4), y=p[1], vx=rng.gauss(0, 6), vy=-rng.uniform(25, 60), size=4, size_end=8,
             life=rng.uniform(1.4, 2.6), c0=(175, 225, 255), c1=(120, 190, 255), a0=0.9, turb=10, kind="bubble")


@dataclass
class EffectDef:
    key: str
    name: str
    rate: float          # emissions per second per 200 px of outline
    emit: Callable
    color: Tuple[int, int, int]   # swatch / ring colour
    ring: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]   # outer, mid, core
    arcs: float          # 0..1  electric-arc intensity on the energy ring

EFFECTS: Dict[str, EffectDef] = {e.key: e for e in [
    EffectDef("fire",        "Fire",        26, _e_fire,        (255, 140,  30), ((200, 60, 10), (255, 140, 30), (255, 230, 160)), 0.0),
    EffectDef("smoke",       "Smoke",       10, _e_smoke,       (170, 170, 180), ((90, 90, 100), (150, 150, 160), (215, 215, 225)), 0.0),
    EffectDef("sparks",      "Sparks",      14, _e_sparks,      (255, 210,  90), ((255, 140, 30), (255, 200, 80), (255, 250, 200)), 0.15),
    EffectDef("snow",        "Snow",        14, _e_snow,        (225, 240, 255), ((110, 160, 230), (170, 210, 250), (240, 250, 255)), 0.0),
    EffectDef("stars",       "Stars",       12, _e_stars,       (255, 225, 110), ((200, 150, 40), (255, 210, 90), (255, 250, 200)), 0.25),
    EffectDef("electricity", "Electricity",  7, _e_electricity, (120, 170, 255), ((40, 90, 255), (110, 170, 255), (230, 245, 255)), 1.0),
    EffectDef("plasma",      "Plasma",      16, _e_plasma,      (230,  90, 255), ((160, 40, 220), (255, 90, 255), (150, 255, 255)), 0.6),
    EffectDef("dust",        "Dust",        12, _e_dust,        (190, 170, 140), ((120, 100, 70), (180, 160, 120), (225, 210, 175)), 0.0),
    EffectDef("magic",       "Magic",       18, _e_magic,       (190, 130, 255), ((120, 60, 220), (200, 120, 255), (255, 235, 255)), 0.4),
    EffectDef("confetti",    "Confetti",    10, _e_confetti,    (255, 100, 160), ((220, 60, 120), (255, 160, 60), (255, 255, 200)), 0.0),
    EffectDef("bubbles",     "Bubbles",      9, _e_bubbles,     (130, 200, 255), ((90, 170, 240), (150, 215, 255), (235, 250, 255)), 0.0),
]}


def effect_palette(key: str, t: float):
    """(outer, mid, core) colours; 'magic' and 'confetti' cycle through the rainbow."""
    d = EFFECTS[key]
    if key == "magic":
        h = t * 0.15
        return _hsv(h, 0.8, 0.8), _hsv(h + 0.12, 0.6, 1.0), (255, 245, 255)
    if key == "confetti":
        h = t * 0.2
        return _hsv(h, 0.9, 0.85), _hsv(h + 0.33, 0.8, 1.0), (255, 255, 230)
    return d.ring


# ───────────────────────────── energy ring ─────────────────────────────
def draw_energy_ring(surface: pygame.Surface, cx: float, cy: float, r: float, key: str, t: float, seed: float = 0.0, k: float = 1.0) -> bool:
    """Glow layers + pulsing radius + rotating dash arcs + electric arcs (additive).  False if skipped."""
    d = EFFECTS.get(key)
    if d is None or r < 8 or r > RING_MAX_RADIUS: return False
    pr = r * (1.0 + 0.035 * math.sin(t * 7.5 + seed))                     # pulse (~1.2 Hz)
    pad = int(pr * 0.3) + 30
    S = int(2 * (pr + pad))
    layer = pygame.Surface((S, S)); layer.fill((0, 0, 0))
    c = (S // 2, S // 2); R = int(pr)
    outer, mid, core = effect_palette(key, t)
    if k < 0.999: outer, mid, core = _scl(outer, k), _scl(mid, k), _scl(core, k)      # layer opacity
    pygame.draw.circle(layer, _scl(outer, 0.55), c, R, min(R, 16))
    pygame.draw.circle(layer, _scl(mid, 0.75), c, R, min(R, 8))
    pygame.draw.circle(layer, core, c, R, 2)
    ang = t * 1.7 + seed                                                  # rotation
    for rad, n, w, spd, col in ((pr + 10, 6, 3, 1.0, mid), (pr - 10, 9, 2, -1.4, core)):
        rect = pygame.Rect(0, 0, int(2 * rad), int(2 * rad)); rect.center = c
        for k in range(n):
            a0 = ang * spd + k * math.tau / n
            pygame.draw.arc(layer, col, rect, a0, a0 + math.tau / n * 0.45, w)
    n_arcs = int(round(d.arcs * 4))
    if n_arcs:                                                            # electric arcs (re-rolled ~18x/s)
        rng = random.Random(int(seed * 1000) + int(t * 18))
        for k in range(n_arcs):
            a = rng.uniform(0, math.tau)
            p0 = (c[0] + math.cos(a) * pr, c[1] + math.sin(a) * pr)
            if d.arcs >= 0.9 and k == 0:                                  # one bolt crossing the ring
                a2 = a + math.pi + rng.uniform(-0.7, 0.7)
                p1 = (c[0] + math.cos(a2) * pr, c[1] + math.sin(a2) * pr)
            else:
                L = rng.uniform(14, 34) * (1.0 + pr / 220.0)
                p1 = (p0[0] + math.cos(a) * L, p0[1] + math.sin(a) * L)
            pts = _bolt_pts(rng, p0, p1, 6.0, 6)
            pygame.draw.lines(layer, _scl(mid, 0.8), False, pts, 4)
            pygame.draw.lines(layer, core, False, pts, 1)
    surface.blit(layer, (int(cx) - S // 2, int(cy) - S // 2), special_flags=pygame.BLEND_RGB_ADD)
    return True


def emit_ring_particles(fx: "FXSystem", key: str, cx: float, cy: float, r: float, rng: random.Random, dt: float) -> int:
    """Orbiting sparks that ride the ring, plus the effect's own particles thrown outward."""
    d = EFFECTS.get(key); n = 0
    if d is None: return 0
    _, mid, core = effect_palette(key, time.perf_counter())
    if rng.random() < 14 * dt:
        fx.spawn(x=cx + r, y=cy, size=3, size_end=1, life=rng.uniform(0.9, 1.5), c0=core, c1=mid, a0=0.95, kind="glow",
                 orbit=(cx, cy, r + rng.gauss(0, 2.5), rng.uniform(0, math.tau), rng.choice((-1, 1)) * rng.uniform(1.2, 2.4)))
        n += 1
    if rng.random() < 18 * dt:
        a = rng.uniform(0, math.tau); nx, ny = math.cos(a), math.sin(a)
        d.emit(fx, (cx + nx * r, cy + ny * r), (nx, ny), rng); n += 1
    return n


# ───────────────────────────── effects menu (finger / mouse / key) ─────────────────────────────
class EffectPanel:
    """FX tab left of the BRUSH tab.  Hover a fingertip over the tab 0.5 s to open, over a row to apply."""
    TAB = pygame.Rect(868, 64, 100, 32)
    ROW, PAD = 32, 8
    PANEL = pygame.Rect(868, 102, 230, len(EFFECT_ORDER) * 32 + 8 * 2 + 6)

    def __init__(self):
        pygame.font.init()
        self._f = pygame.font.SysFont("monospace", 14, bold=True)
        self._s = pygame.font.SysFont("monospace", 11)
        self.open = False
        self._hk: Optional[str] = None; self._latched: Optional[str] = None
        self._ht = 0.0; self._idle = 0.0
        self._rows: Dict[str, pygame.Rect] = {}
        y = self.PANEL.y + self.PAD
        for k in EFFECT_ORDER:
            self._rows[k] = pygame.Rect(self.PANEL.x + 6, y, self.PANEL.w - 12, self.ROW - 3); y += self.ROW

    def toggle(self):
        self.open = not self.open
        self._hk, self._ht, self._idle = None, 0.0, 0.0
        self._latched = "_tab" if self._latched == "_tab" else None

    def blocks(self, pt) -> bool:
        return bool(pt) and (self.TAB.collidepoint(pt) or (self.open and self.PANEL.collidepoint(pt)))

    def _hit(self, pt) -> Optional[str]:
        if not pt: return None
        if self.TAB.collidepoint(pt): return "_tab"
        if self.open:
            for k, r in self._rows.items():
                if r.collidepoint(pt): return k
        return None

    def update(self, tip, dt: float) -> Optional[str]:
        hov = self._hit(tip); fired = None
        if hov != self._latched: self._latched = None
        if hov and hov == self._hk and hov != self._latched:
            self._ht += dt
            if self._ht >= UI_HOVER_S:
                self._ht = 0.0; self._latched = hov
                if hov == "_tab": self.toggle()
                else: fired = hov; self.open = False
        elif hov != self._hk:
            self._hk, self._ht = hov, 0.0
        if self.open and not hov:
            self._idle += dt
            if self._idle > 6.0: self.open = False
        else:
            self._idle = 0.0
        return fired

    def click(self, pos) -> Optional[str]:
        hit = self._hit(pos)
        if hit == "_tab": self.toggle()
        elif hit: self.open = False; return hit
        return None

    def render(self, surface: pygame.Surface, current: Optional[str]):
        MG = (190, 130, 255)
        t = pygame.Surface(self.TAB.size, pygame.SRCALPHA)
        pygame.draw.rect(t, (8, 14, 24, 200), t.get_rect(), border_radius=10)
        pygame.draw.rect(t, (*MG, 230), t.get_rect(), 2, border_radius=10)
        if self._hk == "_tab":
            pygame.draw.rect(t, (*MG, 110), (0, 0, int(self.TAB.w * min(1, self._ht / UI_HOVER_S)), self.TAB.h), border_radius=10)
        surface.blit(t, self.TAB.topleft)
        nm = EFFECTS[current].name.upper() if current in EFFECTS else "NONE"
        lb = self._f.render(f"FX: {nm}", True, (240, 230, 255))
        surface.blit(lb, (self.TAB.x + 12, self.TAB.y + (self.TAB.h - lb.get_height()) // 2))
        if not self.open: return
        p = pygame.Surface(self.PANEL.size, pygame.SRCALPHA)
        pygame.draw.rect(p, (8, 14, 24, 205), p.get_rect(), border_radius=12)
        pygame.draw.rect(p, (*MG, 200), p.get_rect(), 2, border_radius=12)
        surface.blit(p, self.PANEL.topleft)
        for k in EFFECT_ORDER:
            r = self._rows[k]
            row = pygame.Surface(r.size, pygame.SRCALPHA)
            if (k == current) or (k == "none" and current not in EFFECTS):
                pygame.draw.rect(row, (*MG, 70), row.get_rect(), border_radius=8)
                pygame.draw.rect(row, (*MG, 255), row.get_rect(), 2, border_radius=8)
            if self._hk == k:
                pygame.draw.rect(row, (255, 255, 255, 90), (0, 0, int(r.w * min(1, self._ht / UI_HOVER_S)), r.h), border_radius=8)
            surface.blit(row, r.topleft)
            col = EFFECTS[k].color if k in EFFECTS else (150, 150, 160)
            pygame.draw.circle(surface, col, (r.x + 16, r.centery), 7)
            pygame.draw.circle(surface, (255, 255, 255), (r.x + 16, r.centery), 7, 1)
            surface.blit(self._f.render(EFFECTS[k].name if k in EFFECTS else "No effect", True, (235, 245, 255)), (r.x + 32, r.y + 5))


class TrackerThread:
    """Runs hand tracking off the render thread.  The main loop always takes the newest finished result and
    never waits; each camera frame is processed exactly once."""

    def __init__(self, cam, clf):
        self.cam, self.clf = cam, clf
        self._lock = threading.Lock()
        self._res = None
        self._stop = threading.Event()
        self._last_seq = -1
        self._started = False
        self.proc_ms = 0.0
        self.fps = 0.0
        self.errors = 0

    def start(self):
        if not self._started:
            self._started = True
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last_pub = time.perf_counter()
        while not self._stop.is_set():
            got = self.cam.read_with_seq()
            if got is None or got[0] == self._last_seq:
                time.sleep(0.002)
                continue
            seq, frame = got
            self._last_seq = seq
            t0 = time.perf_counter()
            try:
                hands = self.clf.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for hd in hands:
                    self.clf.draw_skeleton(frame, hd.landmarks, hd.label)
            except Exception:
                hands = []                                   # a bad frame must never kill tracking
                self.errors += 1
            now = time.perf_counter()
            self.proc_ms += 0.1 * ((now - t0) * 1000.0 - self.proc_ms)
            self.fps += 0.1 * (1.0 / max(now - last_pub, 1e-3) - self.fps)
            last_pub = now
            with self._lock:
                self._res = (seq, frame, hands)

    def latest(self):
        with self._lock:
            return self._res

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — EDIT DRAWINGS · LAYERS · REDO · HISTORY TIMELINE
#    Layer / LayerManager      show-hide, lock, opacity, reorder, clear, rename
#    FlyoutPanel (+3 panels)   EDIT / LAYERS / TIME menus (finger-hold, mouse, keys)
#    HistoryLog                labelled events, each with a restorable snapshot
#    pack/unpack_canvas        compact (bounding-box + zlib) canvas snapshots
# ══════════════════════════════════════════════════════════════════════════════
HISTORY_MAX   = 60
EDIT_MIN_W, EDIT_MAX_W = 1, 40
LAYER_NAMES = {"background": "Background", "drawing": "Drawing", "objects": "Physics Objects",
               "effects": "Effects", "ui": "UI"}
LAYER_FIXED_BOTTOM, LAYER_FIXED_TOP = "background", "ui"


# ───────────────────────────── canvas snapshots ─────────────────────────────
def pack_canvas(canvas: pygame.Surface):
    """Only the inked bounding box, zlib-compressed: a typical snapshot is a few KB instead of 3.7 MB."""
    r = canvas.get_bounding_rect()
    if r.w <= 0 or r.h <= 0: return None
    return (r.x, r.y, r.w, r.h, zlib.compress(pygame.image.tostring(canvas.subsurface(r), "RGBA"), 1))


def unpack_canvas(canvas: pygame.Surface, packed):
    canvas.fill((0, 0, 0, 0))
    if packed is None: return
    x, y, w, h, data = packed
    img = pygame.image.frombytes(zlib.decompress(data), (w, h), "RGBA")
    canvas.blit(img, (x, y), special_flags=pygame.BLEND_RGBA_ADD)     # onto transparent == exact copy


def add_glow(spr: pygame.Surface, strength: float, pad: int = 22):
    """Soft halo behind a sprite (bake-time only): a tight + a wide blur, stacked.  Returns (new_sprite, pad)."""
    w, h = spr.get_size()
    big = pygame.Surface((w + 2 * pad, h + 2 * pad), pygame.SRCALPHA)
    big.blit(spr, (pad, pad))
    out = pygame.Surface(big.get_size(), pygame.SRCALPHA)
    cap = 255 * min(1.0, 0.3 + 0.7 * strength)
    for div, gain in ((3, 1.4), (7, 2.6)):
        sw, sh = max(1, big.get_width() // div), max(1, big.get_height() // div)
        blur = pygame.transform.smoothscale(pygame.transform.smoothscale(big, (sw, sh)), big.get_size())
        a = pygame.surfarray.pixels_alpha(blur)
        a[:] = np.clip(a.astype(np.float32) * (gain * strength * 2.0), 0, cap).astype(np.uint8)
        del a
        out.blit(blur, (0, 0))
    out.blit(big, (0, 0))
    return out, pad


# ───────────────────────────── layers ─────────────────────────────
@dataclass
class Layer:
    id: str
    name: str
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0


class LayerManager:
    """Background (bottom) and UI (top) are fixed; Drawing / Physics Objects / Effects can be re-ordered."""
    def __init__(self):
        self.layers: Dict[str, Layer] = {i: Layer(i, n) for i, n in LAYER_NAMES.items()}
        self.order: List[str] = ["drawing", "objects", "effects"]       # bottom -> top
        self.selected: str = "drawing"

    def __getitem__(self, i: str) -> Layer: return self.layers[i]
    def visible(self, i: str) -> bool: return self.layers[i].visible
    def locked(self, i: str) -> bool: return self.layers[i].locked
    def opacity(self, i: str) -> float: return self.layers[i].opacity

    def display_rows(self) -> List[str]:
        """Top -> bottom, as the panel lists them."""
        return [LAYER_FIXED_TOP] + list(reversed(self.order)) + [LAYER_FIXED_BOTTOM]

    def toggle_visible(self, i: str): self.layers[i].visible = not self.layers[i].visible
    def toggle_lock(self, i: str): self.layers[i].locked = not self.layers[i].locked

    def step_opacity(self, i: str, d: float):
        self.layers[i].opacity = round(max(0.1, min(1.0, self.layers[i].opacity + d)), 2)

    def move(self, i: str, direction: int) -> bool:
        """direction +1 = up (drawn later / on top), -1 = down.  Only the three content layers move."""
        if i not in self.order: return False
        k = self.order.index(i); j = k + direction
        if not 0 <= j < len(self.order): return False
        self.order[k], self.order[j] = self.order[j], self.order[k]
        return True

    def rename(self, i: str, name: str) -> bool:
        name = name.strip()[:16]
        if not name: return False
        self.layers[i].name = name
        return True

    def snapshot(self):
        return ({i: (l.name, l.visible, l.locked, l.opacity) for i, l in self.layers.items()}, list(self.order), self.selected)

    def restore(self, snap):
        st, order, sel = snap
        for i, (n, v, lk, op) in st.items():
            l = self.layers[i]; l.name, l.visible, l.locked, l.opacity = n, v, lk, op
        self.order, self.selected = list(order), sel


# ───────────────────────────── history ─────────────────────────────
@dataclass
class HistoryEntry:
    id: int
    label: str
    t: float                      # seconds since the session started
    snap: Optional[dict] = None


class HistoryLog:
    """Labelled events, newest last.  Capped, so memory can never grow without bound."""
    def __init__(self, limit: int = HISTORY_MAX):
        self.limit = limit
        self.entries: List[HistoryEntry] = []
        self._next = 1
        self.t0 = time.perf_counter()
        self.scroll = 0                                   # entries hidden below the visible window

    def add(self, label: str, snap: Optional[dict]) -> HistoryEntry:
        e = HistoryEntry(self._next, label, time.perf_counter() - self.t0, snap)
        self._next += 1
        self.entries.append(e)
        if len(self.entries) > self.limit:
            del self.entries[0]
        self.scroll = 0
        return e

    def get(self, eid: int) -> Optional[HistoryEntry]:
        return next((e for e in self.entries if e.id == eid), None)

    def clear(self):
        self.entries.clear(); self.scroll = 0

    @staticmethod
    def stamp(t: float) -> str:
        return "%02d:%02d" % (int(t) // 60, int(t) % 60)


# ───────────────────────────── generic flyout panel ─────────────────────────────
@dataclass
class Cell:
    text: str = ""
    key: str = ""                       # "" = display only (not clickable)
    arg: Any = None
    w: int = 0
    repeat: bool = False                # keep firing while the finger stays on it (steppers)
    rgb: Optional[Tuple[int, int, int]] = None
    on: bool = False
    enabled: bool = True


@dataclass
class RowSpec:
    cells: List[Cell] = field(default_factory=list)
    header: str = ""
    h: int = 28


class FlyoutPanel:
    """A tab + a drop-down panel made of rows of buttons.  Subclasses only describe the rows."""
    def __init__(self, title: str, tab: pygame.Rect, width: int, accent: Tuple[int, int, int]):
        pygame.font.init()
        self.title, self.TAB, self.width, self.accent = title, tab, width, accent
        self._f = pygame.font.SysFont("monospace", 13, bold=True)
        self._s = pygame.font.SysFont("monospace", 11)
        self.open = False
        self.panel_rect = pygame.Rect(tab.x, tab.bottom + 4, width, 40)
        self.hits: List[Tuple[pygame.Rect, str, Any, bool]] = []
        self._draw: List[Tuple[pygame.Rect, RowSpec, List[Tuple[pygame.Rect, Cell]]]] = []
        self._hk = None; self._ht = 0.0; self._latched = False; self._rep = 0.0; self._idle = 0.0

    def specs(self, ctx) -> List[RowSpec]:
        raise NotImplementedError

    def toggle(self):
        self.open = not self.open
        self._hk, self._ht, self._latched, self._idle = None, 0.0, True, 0.0

    def layout(self, ctx):
        rows = self.specs(ctx)
        h = sum(r.h for r in rows) + 14
        x = min(self.TAB.x, WIN_W - self.width - 4)
        self.panel_rect = pygame.Rect(x, self.TAB.bottom + 4, self.width, h)
        y = self.panel_rect.y + 7
        self.hits, self._draw = [], []
        for r in rows:
            rr = pygame.Rect(x + 6, y, self.width - 12, r.h - 2)
            placed, cx = [], rr.right
            for c in reversed(r.cells):
                cw = c.w or (22 if c.rgb else max(28, self._f.size(c.text)[0] + 14))
                cx -= cw + 4
                placed.append((pygame.Rect(cx, rr.y, cw, rr.h), c))
            for rect, c in placed:
                if c.enabled and c.key: self.hits.append((rect, c.key, c.arg, c.repeat))
            self._draw.append((rr, r, placed))
            y += r.h

    def blocks(self, pt) -> bool:
        return bool(pt) and (self.TAB.collidepoint(pt) or (self.open and self.panel_rect.collidepoint(pt)))

    def _hit(self, pt):
        if not pt: return None
        if self.TAB.collidepoint(pt): return ("_tab", None, False)
        if self.open:
            for rect, key, arg, rep in self.hits:
                if rect.collidepoint(pt): return (key, arg, rep)
        return None

    def update(self, tip, dt: float):
        """Returns (key, arg) when a button fires (0.5 s finger-hold; steppers repeat), else None."""
        hit = self._hit(tip); fired = None
        hov = None if hit is None else (hit[0], repr(hit[1]))
        if hov != self._hk:
            self._hk, self._ht, self._latched, self._rep = hov, 0.0, False, 0.0
        elif hov is not None:
            self._ht += dt
            if not self._latched and self._ht >= UI_HOVER_S:
                self._latched, self._rep = True, 0.0
                if hit[0] == "_tab": self.toggle()
                else: fired = (hit[0], hit[1])
            elif self._latched and hit[2] and hit[0] != "_tab":
                self._rep += dt
                if self._rep >= 0.25:
                    self._rep = 0.0; fired = (hit[0], hit[1])
        if self.open and hov is None:
            self._idle += dt
            if self._idle > 10.0: self.open = False
        else:
            self._idle = 0.0
        return fired

    def click(self, pos):
        hit = self._hit(pos)
        if hit is None: return None
        if hit[0] == "_tab": self.toggle(); return None
        return (hit[0], hit[1])

    def render(self, surface: pygame.Surface, ctx):
        A = self.accent
        t = pygame.Surface(self.TAB.size, pygame.SRCALPHA)
        pygame.draw.rect(t, (8, 14, 24, 200), t.get_rect(), border_radius=10)
        pygame.draw.rect(t, (*A, ui_pulse(230, 20) if self.open else 230), t.get_rect(), 2, border_radius=10)
        if self._hk is not None and self._hk[0] == "_tab":
            pygame.draw.rect(t, (*A, 110), (0, 0, int(self.TAB.w * min(1, self._ht / UI_HOVER_S)), self.TAB.h), border_radius=10)
        surface.blit(t, self.TAB.topleft)
        lb = self._f.render(self.title, True, (240, 240, 255))
        surface.blit(lb, (self.TAB.x + (self.TAB.w - lb.get_width()) // 2, self.TAB.y + (self.TAB.h - lb.get_height()) // 2))
        if not self.open: return
        self.layout(ctx)
        p = pygame.Surface(self.panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(p, (8, 14, 24, 215), p.get_rect(), border_radius=12)
        pygame.draw.rect(p, (*A, ui_pulse(200, 40)), p.get_rect(), 2, border_radius=12)
        surface.blit(p, self.panel_rect.topleft)
        for rr, r, placed in self._draw:
            if r.header:
                surface.blit(self._f.render(r.header, True, (*A,)), (rr.x + 4, rr.y + 5)); continue
            for rect, c in placed:
                b = pygame.Surface(rect.size, pygame.SRCALPHA)
                base = (40, 48, 70, 200) if (c.key and c.enabled) else (22, 26, 40, 160)
                if c.rgb: base = (*c.rgb, 255)
                pygame.draw.rect(b, base, b.get_rect(), border_radius=6)
                if c.on: pygame.draw.rect(b, (255, 255, 255, 255), b.get_rect(), 2, border_radius=6)
                elif c.key: pygame.draw.rect(b, (*A, 150), b.get_rect(), 1, border_radius=6)
                if self._hk == (c.key, repr(c.arg)) and c.key:
                    pygame.draw.rect(b, (255, 255, 255, 100), (0, 0, int(rect.w * min(1, self._ht / UI_HOVER_S)), rect.h), border_radius=6)
                surface.blit(b, rect.topleft)
                if c.text:
                    col = (235, 240, 255) if (c.enabled or not c.key) else (110, 115, 135)
                    im = self._f.render(c.text, True, col)
                    surface.blit(im, (rect.x + (rect.w - im.get_width()) // 2, rect.y + (rect.h - im.get_height()) // 2))


def _stepper(label: str, key: str, value: str, w_val: int = 64, step: Any = None):
    """  Label        [-] value [+]  """
    return RowSpec(cells=[Cell(label, w=78), Cell("-", key, -1, repeat=True), Cell(value, w=w_val),
                          Cell("+", key, +1, repeat=True)])


def _chooser(label: str, key: str, value: str, w_val: int = 92):
    return RowSpec(cells=[Cell(label, w=78), Cell("<", key, -1), Cell(value, w=w_val), Cell(">", key, +1)])


class EditPanel(FlyoutPanel):
    """Properties of the selected drawing / 3-D object."""
    def __init__(self):
        super().__init__("EDIT", pygame.Rect(762, 64, 100, 32), 420, (255, 170, 70))

    def specs(self, app) -> List[RowSpec]:
        t = app.edit_target()
        if t is None:
            return [RowSpec(header="EDIT: nothing selected"),
                    RowSpec(cells=[Cell("Pinch a drawing (thumb+middle),", w=300)], h=24),
                    RowSpec(cells=[Cell("or launch / convert one first.", w=300)], h=24)]
        kind, obj = t
        eff = getattr(obj, "effect", None) if kind != "pending" else app._pending_effect_key(obj)
        en = EFFECT_ORDER[0] if eff not in EFFECTS else eff
        if kind == "pending":
            return [RowSpec(header="EDIT: stroke on canvas"),
                    _chooser("Effect", "effect", en),
                    RowSpec(cells=[Cell("Launch it (SPACE) to edit more.", w=300)], h=24)]
        if kind == "body":
            uid, frozen = obj.uid, obj.frozen
            color, width = obj.color, obj.width
            mass, bounce, fric, grav = obj.mass_mult, obj.elast, obj.fric, obj.gscale
        else:
            uid, frozen = obj.uid, obj.frozen
            color, width = obj.rgb, obj.width
            mass, bounce, fric, grav = obj.mass, obj.bounce, obj.fric, obj.gscale
        sw = [Cell(w=22, key="color", arg=k, rgb=PALETTE[k]["rgb"], on=(tuple(PALETTE[k]["rgb"]) == tuple(color))) for k in PALETTE]
        return [
            RowSpec(header="EDIT  #%d  (%s)" % (uid, "drawing" if kind == "body" else "3D " + obj.name)),
            RowSpec(cells=[Cell("Color", w=48)] + sw),
            _chooser("Brush", "brush", BRUSHES[obj.brush].name if obj.brush in BRUSHES else obj.brush),
            _stepper("Size", "size", "%dpx" % width),
            _stepper("Opacity", "opacity", "%d%%" % round(obj.opacity * 100)),
            _stepper("Glow", "glow", "%d%%" % round(obj.glow * 100)),
            _chooser("Effect", "effect", EFFECTS[eff].name if eff in EFFECTS else "None"),
            RowSpec(cells=[Cell("Physics", w=78), Cell("FROZEN" if frozen else "DYNAMIC", "freeze", w=96, on=frozen),
                           Cell("HIDDEN" if not obj.visible else "VISIBLE", "visible", w=92, on=not obj.visible)]),
            _stepper("Mass", "mass", "x%.2f" % mass),
            _stepper("Bounce", "bounce", "%.1f" % bounce),
            _stepper("Friction", "fric", "%.1f" % fric),
            _stepper("Gravity", "gscale", "x%.2f" % grav),
            RowSpec(cells=[Cell("DUPLICATE", "dup", w=140), Cell("DELETE", "delete", w=140)], h=32),
        ]


class LayerPanel(FlyoutPanel):
    def __init__(self):
        super().__init__("LAYERS", pygame.Rect(656, 64, 100, 32), 480, (90, 210, 160))

    def specs(self, app) -> List[RowSpec]:
        L = app.lyr; rows = [RowSpec(header="LAYERS  (top of list = drawn last)")]
        for i in L.display_rows():
            lay = L[i]; renaming = app.rename_id == i
            name = (app.rename_buf + "_") if renaming else lay.name
            cells = [Cell(name[:15], "select", i, w=128, on=(L.selected == i)),
                     Cell("SHOW" if lay.visible else "HIDE", "vis", i, w=44, on=not lay.visible),
                     Cell("LOCK" if lay.locked else "free", "lock", i, w=44, on=lay.locked, enabled=i in ("drawing", "objects", "effects")),
                     Cell("-", "op", (i, -0.1), repeat=True, enabled=i != "ui"), Cell("%d%%" % round(lay.opacity * 100) if i != "ui" else "--", w=42),
                     Cell("+", "op", (i, 0.1), repeat=True, enabled=i != "ui")]
            movable = i in L.order
            cells += [Cell("UP", "up", i, w=30, enabled=movable), Cell("DN", "down", i, w=30, enabled=movable),
                      Cell("CLR", "clr", i, w=34, enabled=(i in ("drawing", "objects", "effects")))]
            rows.append(RowSpec(cells=cells, h=30))
        rows.append(RowSpec(header="Ctrl+R renames the selected layer"))
        return rows


class TimelinePanel(FlyoutPanel):
    VISIBLE = 10
    def __init__(self):
        super().__init__("TIME", pygame.Rect(550, 64, 100, 32), 420, (120, 190, 255))

    def specs(self, app) -> List[RowSpec]:
        H = app.hist; ents = H.entries
        rows = [RowSpec(header="HISTORY  (%d)" % len(ents)),
                RowSpec(cells=[Cell("UNDO", "undo", w=96), Cell("REDO", "redo", w=96)], h=30)]
        end = len(ents) - H.scroll; start = max(0, end - self.VISIBLE)
        for e in reversed(ents[start:end]):
            rows.append(RowSpec(cells=[Cell("%s  %s" % (H.stamp(e.t), e.label), "goto", e.id, w=296, on=(e is ents[-1]),
                                            enabled=e.snap is not None)], h=26))
        if not ents: rows.append(RowSpec(cells=[Cell("(nothing yet - draw something)", w=296)], h=26))
        if len(ents) > self.VISIBLE:
            rows.append(RowSpec(cells=[Cell("OLDER", "older", w=96, enabled=start > 0), Cell("NEWER", "newer", w=96, enabled=H.scroll > 0)], h=28))
        return rows


# ───────────────────────────── (de)serialising objects for snapshots / duplicate ─────────────────────────────
_SOLID_PROPS = ("opacity", "glow", "visible", "mass", "bounce", "fric", "gscale", "frozen", "effect")

def solid_to_dict(s: "Solid3D") -> dict:
    d = dict(kind=s.kind, V=s.V.copy(), chains=[c.copy() for c in s.chains], faces=list(s.faces), rgb=s.rgb, brush=s.brush,
             width=s.width, pos=s.pos.copy(), name=s.name, R=s.R.copy(), scale=s.scale, vel=s.vel.copy(), ang=s.ang.copy(),
             active=s.active, launch_t=s.launch_t)
    d.update({k: getattr(s, k) for k in _SOLID_PROPS})
    return d

def dict_to_solid(d: dict) -> "Solid3D":
    s = Solid3D(d["kind"], d["V"].copy(), [c.copy() for c in d["chains"]], list(d["faces"]), d["rgb"], d["brush"], d["width"],
                d["pos"].copy(), d["name"])
    s.R, s.scale, s.vel, s.ang = d["R"].copy(), d["scale"], d["vel"].copy(), d["ang"].copy()
    s.active, s.launch_t = d["active"], d["launch_t"]
    for k in _SOLID_PROPS: setattr(s, k, d[k])
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 7 — AI: drawing recognition · shape cleanup · sketch enhancement
#  Deterministic geometry only (no ML, no network).  Heavy work runs once per
#  settled drawing, on a worker thread, never per frame.
#  Extension point:  RECOGNIZER_PROVIDERS  (list of callables strokes -> Optional[Recognition])
# ══════════════════════════════════════════════════════════════════════════════
AI_SETTLE_S      = 0.9          # a drawing is recognised this long after the last stroke
AI_CLUSTER_S     = 12.0         # strokes older than this are not part of the same drawing
AI_MIN_CONF      = 0.60
CLEANABLE        = ("circle", "ellipse", "square", "rectangle", "quad", "triangle", "star", "line")


@dataclass
class Recognition:
    label: str                      # circle square rectangle triangle star line heart house tree car person arrow cloud sun unknown
    conf: float                     # fraction of the rule's geometric constraints satisfied (NOT a probability)
    bbox: Tuple[float, float, float, float]       # x0, y0, x1, y1
    strokes: List[int]              # indices (into the strokes passed in) that make up the drawing
    detail: str = ""

RECOGNIZER_PROVIDERS: List[Callable] = []        # optional ML hooks: fn(strokes) -> Optional[Recognition]


def _P(s): return np.asarray(s, float)
def _bbox(P):
    P = _P(P); return (float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max()))
def _plen(P):
    P = _P(P); return float(np.sum(np.hypot(*np.diff(P, axis=0).T))) if len(P) > 1 else 0.0
def _clip01(v): return max(0.0, min(1.0, float(v)))
def _ramp(v, lo, hi):
    """1.0 when v is inside [lo, hi], falling to 0 over a margin of half the range outside it."""
    m = 0.5 * (hi - lo) if hi > lo else 1.0
    if v < lo: return _clip01(1.0 - (lo - v) / m)
    if v > hi: return _clip01(1.0 - (v - hi) / m)
    return 1.0


# ─────────────────────────────── icon library (unit box, y down) ───────────────────────────────
def _circ(cx, cy, r, n=28, a0=0.0, a1=math.tau):
    t = np.linspace(a0, a1, n)
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in t]

def _icon_heart():
    t = np.linspace(0, math.tau, 64)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    x = (x - x.min()) / (x.max() - x.min()); y = (y - y.min()) / (y.max() - y.min())
    return [list(zip(x, y))]

def _icon_cloud():
    """Outline of the union of a few discs on a flat base (rasterised, then traced)."""
    N = 400
    img = np.zeros((N, N), np.uint8)
    for cx, cy, r in [(0.24, 0.62, 0.17), (0.42, 0.42, 0.22), (0.64, 0.46, 0.19), (0.78, 0.64, 0.15)]:
        cv2.circle(img, (int(cx * N), int(cy * N)), int(r * N), 255, -1)
    cv2.rectangle(img, (int(0.24 * N), int(0.55 * N)), (int(0.78 * N), int(0.79 * N)), 255, -1)
    img[int(0.79 * N):, :] = 0
    cs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0, :].astype(float)
    c = c[::max(1, len(c) // 90)]
    c = np.vstack([c, c[:1]])
    c[:, 0] = (c[:, 0] - c[:, 0].min()) / np.ptp(c[:, 0]); c[:, 1] = (c[:, 1] - c[:, 1].min()) / np.ptp(c[:, 1])
    return [[tuple(p) for p in c]]

def _icon_star(n=5):
    pts = []
    for i in range(2 * n + 1):
        r = 0.5 if i % 2 == 0 else 0.21
        a = -math.pi / 2 + i * math.pi / n
        pts.append((0.5 + r * math.cos(a), 0.52 + r * math.sin(a)))
    return [pts]

def _icon_galaxy():
    arms = []
    for k in range(2):
        arms.append([(0.5 + (0.04 + 0.46 * (t / 60)) * math.cos(t / 60 * 3.6 * math.pi + k * math.pi),
                      0.5 + (0.04 + 0.46 * (t / 60)) * math.sin(t / 60 * 3.6 * math.pi + k * math.pi)) for t in range(61)])
    return arms + [_circ(0.5, 0.5, 0.05, 12)]

ICONS: Dict[str, List[List[Tuple[float, float]]]] = {
    "circle":    [_circ(0.5, 0.5, 0.5, 40)],
    "square":    [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
    "rectangle": [[(0, 0.15), (1, 0.15), (1, 0.85), (0, 0.85), (0, 0.15)]],
    "triangle":  [[(0.5, 0), (1, 1), (0, 1), (0.5, 0)]],
    "line":      [[(0, 0.5), (1, 0.5)]],
    "star":      _icon_star(5),
    "heart":     _icon_heart(),
    "cloud":     _icon_cloud(),
    "galaxy":    _icon_galaxy(),
    "house":     [[(0.08, 0.5), (0.5, 0.08), (0.92, 0.5), (0.08, 0.5)],
                  [(0.16, 0.5), (0.16, 0.96), (0.84, 0.96), (0.84, 0.5)],
                  [(0.42, 0.96), (0.42, 0.7), (0.58, 0.7), (0.58, 0.96)]],
    "tree":      [_circ(0.5, 0.3, 0.3, 34), [(0.43, 0.58), (0.43, 1.0), (0.57, 1.0), (0.57, 0.58)]],
    "car":       [[(0.0, 0.62), (0.0, 0.42), (0.18, 0.4), (0.3, 0.14), (0.7, 0.14), (0.84, 0.4), (1.0, 0.44), (1.0, 0.62), (0.0, 0.62)],
                  _circ(0.25, 0.68, 0.12, 20), _circ(0.75, 0.68, 0.12, 20)],
    "person":    [_circ(0.5, 0.12, 0.12, 20), [(0.5, 0.24), (0.5, 0.62)], [(0.12, 0.42), (0.5, 0.34), (0.88, 0.42)],
                  [(0.5, 0.62), (0.22, 1.0)], [(0.5, 0.62), (0.78, 1.0)]],
    "arrow":     [[(0.0, 0.5), (1.0, 0.5)], [(0.72, 0.18), (1.0, 0.5), (0.72, 0.82)]],
    "sun":       [],
    "spaceship": [[(0.5, 0.0), (0.64, 0.2), (0.64, 0.7), (0.36, 0.7), (0.36, 0.2), (0.5, 0.0)],
                  [(0.36, 0.45), (0.1, 0.78), (0.1, 0.92), (0.36, 0.7)], [(0.64, 0.45), (0.9, 0.78), (0.9, 0.92), (0.64, 0.7)],
                  _circ(0.5, 0.32, 0.07, 16), [(0.42, 0.7), (0.5, 1.0), (0.58, 0.7)]],
}
# the sun needs rays: build them explicitly (kept out of the dict literal above for readability)
ICONS["sun"] = [_circ(0.5, 0.5, 0.2, 28)] + [
    [(0.5 + 0.3 * math.cos(a), 0.5 + 0.3 * math.sin(a)), (0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a))]
    for a in [i * math.tau / 8 for i in range(8)]]
ICON_ALIASES = {"rocket": "spaceship", "ship": "spaceship", "spaceship": "spaceship", "space ship": "spaceship", "love": "heart",
                "home": "house", "automobile": "car", "vehicle": "car", "man": "person", "human": "person", "stick figure": "person",
                "people": "person", "star": "star", "stars": "star", "sunshine": "sun", "galaxy": "galaxy", "spiral": "galaxy"}

def icon_strokes(name: str, box: Tuple[float, float, float, float], flip_x: bool = False) -> List[List[Tuple[int, int]]]:
    """Polylines of a named icon mapped into box (x0, y0, x1, y1), absolute screen px."""
    name = ICON_ALIASES.get(name, name)
    if name not in ICONS: return []
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    out = []
    for pl in ICONS[name]:
        out.append([(int(round(x0 + (1 - u if flip_x else u) * w)), int(round(y0 + v * h))) for u, v in pl])
    return out


# ─────────────────────────────── drawing recognition ───────────────────────────────
CLOUD_N = 48
ICON_CLOUDS: Dict[str, np.ndarray] = {}


def _resample_multi(strokes, n: int = CLOUD_N) -> np.ndarray:
    """All strokes -> one point cloud of n points spread in proportion to stroke length."""
    S = [np.asarray(s, float) for s in strokes if len(s) >= 2]
    if not S: return np.zeros((0, 2))
    L = np.array([max(_plen(s), 1e-6) for s in S])
    cnt = np.maximum(2, np.round(n * L / L.sum()).astype(int))
    out = []
    for s, c in zip(S, cnt):
        seg = np.hypot(*np.diff(s, axis=0).T); cum = np.concatenate([[0], np.cumsum(seg)])
        if cum[-1] < 1e-9: out.append(np.repeat(s[:1], c, axis=0)); continue
        u = np.linspace(0, cum[-1], int(c))
        out.append(np.stack([np.interp(u, cum, s[:, 0]), np.interp(u, cum, s[:, 1])], 1))
    return np.vstack(out)


def _unit_box(P: np.ndarray) -> np.ndarray:
    lo, hi = P.min(0), P.max(0); sz = np.maximum(hi - lo, 1e-6)
    return (P - lo) / sz


def _chamfer(A: np.ndarray, B: np.ndarray) -> float:
    D = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
    return 0.5 * float(D.min(1).mean() + D.min(0).mean())


def _icon_cloud(name: str) -> np.ndarray:
    c = ICON_CLOUDS.get(name)
    if c is None:
        c = ICON_CLOUDS[name] = _unit_box(_resample_multi(ICONS[name]))
    return c


CHAMFER_SCALE = 0.17          # chamfer distance (unit-box) at which confidence reaches 0
PRIMITIVE_KINDS = ("circle", "ellipse", "square", "rectangle", "triangle", "star", "line")


def match_icons(strokes) -> List[Tuple[str, float]]:
    """Every icon with a confidence in [0,1] (1 - chamfer/CHAMFER_SCALE), best first."""
    P = _resample_multi(strokes)
    if len(P) < 6: return []
    bb = _bbox(P)
    if max(bb[2] - bb[0], bb[3] - bb[1]) < 18: return []
    Q = _unit_box(P)
    Qd = _unit_box(_resample_multi(strokes, 160))
    res = [(n, _adjust(n, _clip01(1.0 - _chamfer(Q, _icon_cloud(n)) / CHAMFER_SCALE), Qd, strokes)) for n in ICONS]
    res.sort(key=lambda t: -t[1])
    return res


def _notch_depth(Q: np.ndarray) -> float:
    """Top-edge profile: how far the middle of the outline sits below the two side lobes (heart: big, circle: ~0).
    An empty gap at the top-middle (stroke ended there) counts as a notch."""
    bins = 12
    top = np.ones(bins)
    ix = np.clip((Q[:, 0] * bins).astype(int), 0, bins - 1)
    for i, y in zip(ix, Q[:, 1]): top[i] = min(top[i], y)
    lobes = max(top[:5].min(), top[7:].min())
    return float(top[4:8].max() - lobes)


def _bottom_width(Q: np.ndarray) -> float:
    b = Q[Q[:, 1] > 0.86]
    return float(np.ptp(b[:, 0])) if len(b) > 1 else 0.0


def _net_turning(strokes) -> float:
    """Largest |signed total turning| of any single stroke (radians): ~2*pi for a loop, >10 for a spiral."""
    best = 0.0
    for s in strokes:
        if len(s) < 8: continue
        P = _resample_multi([s], 40)
        d = np.diff(P, axis=0); a = np.arctan2(d[:, 1], d[:, 0]); da = np.diff(a)
        da = (da + math.pi) % math.tau - math.pi
        best = max(best, abs(float(da.sum())))
    return best


def _adjust(name: str, conf: float, Q: np.ndarray, strokes) -> float:
    """Shape-specific evidence that a plain point-cloud distance cannot see (smoothing rounds corners and notches)."""
    if name in ("heart", "circle"):
        nd = _notch_depth(Q); bw = _bottom_width(Q)
        if name == "heart":
            if nd > 0.06: conf += 0.12 + (0.05 if bw < 0.56 else 0.0)
            elif nd < 0.025: conf -= 0.2
        elif nd > 0.09: conf -= 0.15
    elif name == "galaxy":
        if _net_turning(strokes) < 5.0: conf -= 0.25
    return _clip01(conf)


def _complexity_ok(label: str, strokes) -> bool:
    """Cheap sanity gates that stop a scribble from matching a complex icon."""
    P = _resample_multi(strokes); bb = _bbox(P); w, h = bb[2] - bb[0], bb[3] - bb[1]
    tot = sum(_plen(s) for s in strokes if len(s) > 1)
    diag = math.hypot(w, h) + 1e-6
    if label in ("heart", "cloud", "circle") and not (1.4 * diag < tot < 4.2 * diag): return False
    if label in ("house", "tree", "car", "person", "spaceship", "sun", "galaxy", "arrow") and tot < 1.6 * diag: return False
    if tot > 9.0 * diag: return False                   # a scribble
    return True


def recognize_drawing(strokes) -> Recognition:
    """strokes: list of polylines (screen px).  Primitive shapes come from ShapeRecognizer (strict geometry);
    everything else is matched against the icon library.  Always returns a Recognition ('unknown' at worst)."""
    S = [list(s) for s in strokes if len(s) >= 2]
    if not S: return Recognition("unknown", 0.0, (0, 0, 0, 0), [], "no strokes")
    allp = np.vstack([np.asarray(s, float) for s in S]); bb = _bbox(allp); idx = list(range(len(S)))
    for fn in RECOGNIZER_PROVIDERS:                     # optional ML hook wins if it is confident
        try:
            r = fn(S)
            if r is not None and r.conf >= AI_MIN_CONF: return r
        except Exception:
            pass
    if len(S) == 1 and len(S[0]) >= 4:                  # a straight (even wobbly) stroke is a line
        sv = np.linalg.svd(allp - allp.mean(0), compute_uv=False)
        if sv[0] > 40 and sv[1] / sv[0] < 0.09:
            return Recognition("line", 0.9, bb, idx, "Line")
    res = None
    if len(S) == 1:
        try: res = ShapeRecognizer.recognize(S[0])
        except Exception: res = None
        if res is not None and res.kind in PRIMITIVE_KINDS:
            return Recognition(res.kind, 0.9, bb, idx, f"{res.shape_name}")
    m = match_icons(S)
    if m:
        (l1, c1), c2 = m[0], (m[1][1] if len(m) > 1 else 0.0)
        # circle / square etc. drawn with several strokes still come out as icons of the same name
        if c1 >= AI_MIN_CONF and (c1 - c2 >= 0.03 or l1 == m[1][0]) and _complexity_ok(l1, S):
            return Recognition(l1, c1, bb, idx, l1)
    if len(S) == 1 and res is not None:
        return Recognition("unknown", 0.0, bb, idx, f"free-form {res.shape_name.lower()}")
    return Recognition("unknown", 0.0, bb, idx, "not recognised")


def recognition_label(r: Recognition) -> str:
    return "?" if r.label == "unknown" else r.label.upper()


# ─────────────────────────────── cleanup / enhance ───────────────────────────────
def _box_of(strokes, pad: float = 0.0):
    bb = _bbox(np.vstack([np.asarray(s, float) for s in strokes]))
    return (bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)


def chaikin(P, rounds: int = 2, closed: bool = False) -> np.ndarray:
    P = np.asarray(P, float)
    for _ in range(rounds):
        if len(P) < 3: break
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            Q.append(0.75 * a + 0.25 * b); Q.append(0.25 * a + 0.75 * b)
        Q.append(P[-1]); P = np.array(Q)
    return P


def smooth_stroke(pts, rounds: int = 2, win: int = 5) -> List[Tuple[int, int]]:
    """Hand-tremor removal: moving average (ends pinned) then Chaikin rounding."""
    P = np.asarray(pts, float)
    if len(P) < 4: return [tuple(map(int, p)) for p in P]
    k = min(win, len(P) // 2 * 2 - 1) or 1
    pad = k // 2
    Q = P.copy()
    for d in (0, 1):
        col = np.pad(P[:, d], (pad, pad), mode="edge")
        Q[:, d] = np.convolve(col, np.ones(k) / k, mode="valid")
    Q[0], Q[-1] = P[0], P[-1]
    Q = chaikin(Q, rounds)
    return [(int(round(x)), int(round(y))) for x, y in Q]


def clean_strokes(strokes, rec: Recognition) -> List[List[Tuple[int, int]]]:
    """Idealised polylines for a recognised drawing (empty list when there is nothing to clean)."""
    if rec.label == "unknown": return []
    if rec.label in PRIMITIVE_KINDS and len(strokes) == 1:
        try:
            res = ShapeRecognizer.recognize(strokes[0])
        except Exception:
            res = None
        if res is not None and res.kind == rec.label:
            return [[(int(round(x)), int(round(y))) for x, y in res.outline]]
    box = _box_of(strokes)
    if rec.label in ICONS:
        return icon_strokes(rec.label, box)
    return []


def enhance_strokes(strokes, rec: Recognition) -> List[List[Tuple[int, int]]]:
    """Beautified version: ideal shape/icon when recognised, otherwise tremor-free smoothing."""
    c = clean_strokes(strokes, rec)
    if c: return c
    return [smooth_stroke(s) for s in strokes if len(s) >= 3]


# ─────────────────────────────── air calculator / air math ───────────────────────────────
# Handwritten symbol recogniser (digits, + - x / = . ( ) and the variable x) built from stroke templates and
# the same chamfer matcher as the icon library.  Pure geometry, no model, no network.
def _arc(cx, cy, rx, ry, a0, a1, n=14):
    t = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in t]

# glyph boxes: x 0..0.6, y 0..1 (y down).  Angles in degrees, 0 = right, 90 = down (screen coordinates).
_G = {
    "0": [[*_arc(0.3, 0.5, 0.29, 0.5, 0, 360, 30)]],
    "1": [[(0.1, 0.22), (0.32, 0.0), (0.32, 1.0)]],
    "2": [[*_arc(0.3, 0.27, 0.27, 0.27, 190, 410, 14), (0.0, 1.0), (0.6, 1.0)]],
    "3": [[*_arc(0.28, 0.25, 0.26, 0.25, 200, 450, 12), *_arc(0.28, 0.75, 0.3, 0.25, 270, 520, 14)]],
    "4": [[(0.45, 1.0), (0.45, 0.0), (0.0, 0.68), (0.62, 0.68)]],
    "5": [[(0.55, 0.0), (0.08, 0.0), (0.03, 0.46), *_arc(0.3, 0.7, 0.3, 0.3, 235, 500, 16)]],
    "6": [[(0.5, 0.0), *_arc(0.3, 0.68, 0.3, 0.32, 200, -160, 26)]],
    "7": [[(0.0, 0.0), (0.6, 0.0), (0.2, 1.0)]],
    "8": [[*_arc(0.3, 0.26, 0.24, 0.26, 90, 450, 18), *_arc(0.3, 0.74, 0.29, 0.26, -90, 270, 20)]],
    "9": [[*_arc(0.3, 0.32, 0.29, 0.32, 0, 360, 22), (0.58, 0.32), (0.46, 1.0)]],
}
_G_ALT = {
    "1": [[(0.3, 0.0), (0.3, 1.0)]],
    "4": [[(0.4, 0.0), (0.0, 0.68), (0.62, 0.68)], [(0.42, 0.3), (0.42, 1.0)]],
    "7": [[(0.0, 0.0), (0.6, 0.0), (0.2, 1.0)], [(0.12, 0.5), (0.48, 0.5)]],
    "2": [[(0.02, 0.2), (0.15, 0.03), (0.4, 0.0), (0.58, 0.2), (0.45, 0.45), (0.0, 1.0), (0.62, 1.0)]],
    "9": [[*_arc(0.3, 0.3, 0.29, 0.3, 0, 360, 22)], [(0.58, 0.3), (0.55, 1.0)]],
    "6": [[(0.45, 0.0), (0.12, 0.45), *_arc(0.3, 0.7, 0.29, 0.3, 180, 540, 24)]],
    "3": [[(0.0, 0.1), (0.3, 0.0), (0.55, 0.15), (0.25, 0.48), (0.58, 0.72), (0.3, 1.0), (0.0, 0.9)]],
    "5": [[(0.55, 0.0), (0.08, 0.0), (0.03, 0.46), (0.35, 0.4), (0.6, 0.65), (0.4, 0.97), (0.0, 0.88)]],
}
GLYPHS: Dict[str, List[List[Tuple[float, float]]]] = {k: v for k, v in _G.items()}
GLYPH_VARIANTS: Dict[str, List[List[List[Tuple[float, float]]]]] = {k: [v] + ([_G_ALT[k]] if k in _G_ALT else []) for k, v in _G.items()}
GLYPH_VARIANTS["("] = [[[*_arc(0.55, 0.5, 0.45, 0.5, 100, 260, 14)]]]
GLYPH_VARIANTS[")"] = [[[*_arc(0.05, 0.5, 0.45, 0.5, -80, 80, 14)]]]
_GLYPH_CLOUDS: Dict[str, List[np.ndarray]] = {}


def _glyph_cloud(P: np.ndarray) -> np.ndarray:
    """Scale uniformly (aspect kept) and centre in the unit box, so '1' stays thin and '0' stays round."""
    lo, hi = P.min(0), P.max(0); sz = max(float((hi - lo).max()), 1e-6)
    Q = (P - lo) / sz
    return Q - (Q.max(0) + Q.min(0)) / 2 + 0.5


def _digit_clouds() -> Dict[str, List[np.ndarray]]:
    if not _GLYPH_CLOUDS:
        for k, variants in GLYPH_VARIANTS.items():
            cl = []
            for v in variants:
                cl.append(_glyph_cloud(_resample_multi(v, 40)))
                # the finger filter rounds corners: also match a low-pass filtered copy of every template
                sm = []
                for pl in v:
                    P = np.asarray(pl, float) * 100.0
                    P = _resample_multi([P], 60)
                    sm.append([tuple(q) for q in smooth_stroke(P, rounds=1, win=11)])
                cl.append(_glyph_cloud(_resample_multi(sm, 40)))
            _GLYPH_CLOUDS[k] = cl
    return _GLYPH_CLOUDS


def _is_flat(P, ratio: float = 3.0) -> bool:
    bb = _bbox(P); w, h = bb[2] - bb[0], bb[3] - bb[1]
    return w > ratio * max(h, 1.0)

def _is_tall(P, ratio: float = 3.0) -> bool:
    bb = _bbox(P); w, h = bb[2] - bb[0], bb[3] - bb[1]
    return h > ratio * max(w, 1.0)

def _straight(P, tol: float = 0.2) -> bool:
    P = _P(P)
    if len(P) < 3: return True
    sv = np.linalg.svd(P - P.mean(0), compute_uv=False)
    return bool(sv[1] / max(sv[0], 1e-6) < tol)

def _dir_angle(P) -> float:
    P = _P(P); d = P[-1] - P[0]
    if np.hypot(*d) < 1e-6: d = P[np.argmax(np.hypot(*(P - P[0]).T))] - P[0]
    return math.degrees(math.atan2(d[1], d[0])) % 180.0


def recognize_symbol(strokes, ref_h: float = 0.0, mode: str = "calc") -> Tuple[str, float]:
    """One symbol (1-3 strokes) -> (char, confidence).  ref_h = typical symbol height on the page (for '.').
    mode 'calc': an X is the multiply sign; mode 'math': an X is the variable x."""
    S = [np.asarray(s, float) for s in strokes if len(s) >= 2 or True]
    S = [s if len(s) >= 2 else np.vstack([s, s + 0.5]) for s in S]
    allp = np.vstack(S); bb = _bbox(allp); w, h = bb[2] - bb[0], bb[3] - bb[1]
    diag = math.hypot(w, h)
    if ref_h and diag < 0.28 * ref_h: return (".", 0.9)
    if diag < 12: return (".", 0.7)
    n = len(S)
    if n == 1:
        P = S[0]
        if _is_flat(P, 2.6): return ("-", 0.95)
        # a straight slash: treated as divide in calc mode
        sv = np.linalg.svd(P - P.mean(0), compute_uv=False)
        if sv[1] / max(sv[0], 1e-6) < 0.1 and not _is_tall(P, 4.0) and not _is_flat(P, 2.6):
            ang = _dir_angle(P)
            if 14 < abs(ang - 90) < 76: return ("/", 0.8)          # a clearly tilted straight stroke; a near-vertical one is '1' 
    elif n == 2:
        a, b = S
        fa, fb = _is_flat(a, 2.2), _is_flat(b, 2.2)
        if fa and fb:
            gap = abs(a.mean(0)[1] - b.mean(0)[1])
            if gap > 0.12 * max(w, 1) or gap > 6: return ("=", 0.9)
        # crossing strokes: + (one flat/one tall) or X
        la, lb = _plen(a), _plen(b)
        if min(la, lb) > 0.5 * diag * 0.6 and _straight(a) and _straight(b):
            ca, cb = a.mean(0), b.mean(0)
            if np.hypot(*(ca - cb)) < 0.35 * diag:
                aa, ab = _dir_angle(a), _dir_angle(b)
                hv = lambda t: min(t, 180 - t) < 28
                vv = lambda t: abs(t - 90) < 28
                if (hv(aa) and vv(ab)) or (vv(aa) and hv(ab)): return ("+", 0.9)
                if not hv(aa) and not vv(aa) and not hv(ab) and not vv(ab) and abs(aa - ab) > 40:
                    return (("x" if mode == "math" else "*"), 0.9)
    elif n == 3:
        fl = [i for i, s in enumerate(S) if _is_flat(s, 2.0)]
        if len(fl) == 1:
            others = [S[i] for i in range(3) if i != fl[0]]
            line_y = S[fl[0]].mean(0)[1]
            if all(_bbox(o)[2] - _bbox(o)[0] < 0.45 * w + 8 for o in others) and any(o.mean(0)[1] < line_y for o in others) \
                    and any(o.mean(0)[1] > line_y for o in others):
                return ("/", 0.85)
    # digits / brackets via the chamfer matcher
    Q = _glyph_cloud(_resample_multi(S, 40))
    best = ("?", 0.0)
    for ch, clouds in _digit_clouds().items():
        for T in clouds:
            c = _clip01(1.0 - _chamfer(Q, T) / 0.2)
            if c > best[1]: best = (ch, c)
    return best


@dataclass
class SymbolGroup:
    strokes: List[List[Tuple[float, float]]] = field(default_factory=list)
    t_last: float = 0.0
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)


def group_symbols(strokes, times=None, gap_s: float = 1.6) -> List[SymbolGroup]:
    """Stroke list (in drawing order) -> symbol groups.  A stroke joins the previous group when it is drawn soon after
    it AND overlaps it horizontally (so the two bars of '=' or the two lines of '+' stay together)."""
    groups: List[SymbolGroup] = []
    for i, s in enumerate(strokes):
        if len(s) < 1: continue
        bb = _bbox(np.asarray(s, float) if len(s) > 1 else np.asarray([s[0], s[0]], float))
        t = times[i] if times is not None else float(i)
        if groups:
            g = groups[-1]; gw = g.bbox[2] - g.bbox[0]; gh = g.bbox[3] - g.bbox[1]
            ov = min(bb[2], g.bbox[2]) - max(bb[0], g.bbox[0])
            near_x = ov > -0.15 * max(gw, bb[2] - bb[0], 8)
            vgap = max(1.2 * gh, 0.9 * gw, 30.0)
            vert_ok = bb[1] < g.bbox[3] + vgap and bb[3] > g.bbox[1] - vgap
            if (t - g.t_last) <= gap_s and near_x and vert_ok and len(g.strokes) < 3:
                g.strokes.append(list(s)); g.t_last = t
                g.bbox = (min(g.bbox[0], bb[0]), min(g.bbox[1], bb[1]), max(g.bbox[2], bb[2]), max(g.bbox[3], bb[3]))
                continue
        groups.append(SymbolGroup([list(s)], t, bb))
    return groups


def read_expression(strokes, times=None, mode: str = "calc") -> Tuple[str, List[Tuple[str, float]]]:
    """Strokes drawn left-to-right -> expression text (e.g. '25+30') + per-symbol (char, conf)."""
    groups = group_symbols(strokes, times)
    groups.sort(key=lambda g: (g.bbox[0] + g.bbox[2]) / 2)
    hs = sorted(g.bbox[3] - g.bbox[1] for g in groups if g.bbox[3] - g.bbox[1] > 14)
    ref_h = hs[len(hs) // 2] if hs else 0.0
    syms = [recognize_symbol(g.strokes, ref_h, mode) for g in groups]
    if mode == "math":                                   # raised small symbol after a symbol = exponent (x²)
        out = []
        for i, (g, (ch, c)) in enumerate(zip(groups, syms)):
            if i > 0 and ch in "0123456789" and ref_h:
                p = groups[i - 1]; gh = g.bbox[3] - g.bbox[1]; pcy = (p.bbox[1] + p.bbox[3]) / 2
                if gh < 0.75 * (p.bbox[3] - p.bbox[1]) and (g.bbox[1] + g.bbox[3]) / 2 < pcy - 0.1 * (p.bbox[3] - p.bbox[1]):
                    ch = "^" + ch
            out.append((ch, c))
        syms = out
    return "".join(ch for ch, _ in syms), syms


# ---- safe arithmetic (no eval): shunting-yard over + - * / with unary minus and brackets ----
def _tokenise(expr: str):
    toks, i = [], 0
    while i < len(expr):
        c = expr[i]
        if c.isdigit() or c == ".":
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == "."): j += 1
            if expr[i:j].count(".") > 1: raise ValueError("bad number")
            toks.append(("n", float(expr[i:j]))); i = j; continue
        if c in "+-*/()": toks.append((c, c)); i += 1; continue
        raise ValueError(f"unexpected '{c}'")
    return toks


def evaluate(expr: str) -> float:
    """Evaluate a +,-,*,/ expression with brackets.  Raises ValueError/ZeroDivisionError on bad input."""
    expr = expr.replace("×", "*").replace("÷", "/").replace("x", "*").replace(" ", "").rstrip("=")
    toks = _tokenise(expr)
    if not toks: raise ValueError("empty")
    prec = {"+": 1, "-": 1, "*": 2, "/": 2, "u": 3}
    out, ops, prev = [], [], None
    for kind, v in toks:
        if kind == "n": out.append(v)
        elif kind == "(": ops.append("(")
        elif kind == ")":
            while ops and ops[-1] != "(": out.append(ops.pop())
            if not ops: raise ValueError("unbalanced")
            ops.pop()
        else:
            op = "u" if (kind == "-" and (prev is None or prev[0] in "+-*/(")) else kind
            if op == "+" and prev is None: continue
            while ops and ops[-1] != "(" and (prec[ops[-1]] > prec[op] or (prec[ops[-1]] == prec[op] and op != "u")):
                out.append(ops.pop())
            ops.append(op)
        prev = (kind, v)
    while ops:
        o = ops.pop()
        if o == "(": raise ValueError("unbalanced")
        out.append(o)
    st = []
    for t in out:
        if isinstance(t, float): st.append(t)
        elif t == "u":
            if not st: raise ValueError("bad")
            st.append(-st.pop())
        else:
            if len(st) < 2: raise ValueError("bad")
            b, a = st.pop(), st.pop()
            st.append(a + b if t == "+" else a - b if t == "-" else a * b if t == "*" else a / b)
    if len(st) != 1: raise ValueError("bad")
    return st[0]


def fmt_num(v: float) -> str:
    if abs(v - round(v)) < 1e-9: return str(int(round(v)))
    return f"{v:.6g}"


# ---- tiny polynomial algebra for "air math":  x^2+2x+1  ->  (x+1)^2 ----
def parse_poly(expr: str) -> Dict[int, float]:
    """Sum of terms  [n]x[^k]  or  n  -> {power: coeff}.  Raises ValueError for anything else."""
    s = expr.replace(" ", "").replace("²", "^2").replace("³", "^3").replace("−", "-")
    if not s: raise ValueError("empty")
    if s[0] not in "+-": s = "+" + s
    terms = re.findall(r"[+-][^+-]+", s)
    if "".join(terms) != s: raise ValueError("bad polynomial")
    P: Dict[int, float] = {}
    for t in terms:
        sign = -1.0 if t[0] == "-" else 1.0; body = t[1:]
        m = re.fullmatch(r"(\d+(?:\.\d+)?)?(x(?:\^(\d))?)?", body)
        if not m or not body: raise ValueError("bad term")
        coef = float(m.group(1)) if m.group(1) else 1.0
        pw = (int(m.group(3)) if m.group(3) else 1) if m.group(2) else 0
        P[pw] = P.get(pw, 0.0) + sign * coef
    return {k: v for k, v in P.items() if abs(v) > 1e-12} or {0: 0.0}


def poly_str(P: Dict[int, float]) -> str:
    out = ""
    for pw in sorted(P, reverse=True):
        c = P[pw]
        if abs(c) < 1e-12: continue
        sign = "-" if c < 0 else "+"; a = abs(c)
        body = ("" if (a == 1 and pw > 0) else fmt_num(a)) + ("" if pw == 0 else "x" + ("" if pw == 1 else "²" if pw == 2 else "^%d" % pw))
        out += (sign + body) if out else (("-" if c < 0 else "") + body)
    return out or "0"


def simplify_poly(expr: str) -> str:
    """Collect terms and, for a quadratic with rational roots, show the factored form."""
    P = parse_poly(expr)
    simp = poly_str(P)
    a, b, c = P.get(2, 0.0), P.get(1, 0.0), P.get(0, 0.0)
    if max(P) == 2 and all(abs(v - round(v)) < 1e-9 for v in (a, b, c)):
        a, b, c = int(round(a)), int(round(b)), int(round(c))
        D = b * b - 4 * a * c
        if D >= 0:
            r = int(round(math.sqrt(D)))
            if r * r == D:
                from fractions import Fraction
                r1, r2 = Fraction(-b + r, 2 * a), Fraction(-b - r, 2 * a)
                def lin(root):
                    # (qx - p) for root p/q
                    q, p = root.denominator, root.numerator
                    qx = "x" if q == 1 else f"{q}x"
                    return f"({qx}{'-' if p > 0 else '+'}{abs(p)})" if p != 0 else f"({qx})"
                lead = "" if a == 1 else ("-" if a == -1 else str(a))
                if r1 == r2: return f"{lead}{lin(r1)}²"
                return f"{lead}{lin(r1)}{lin(r2)}"
    return simp


def solve_equation(expr: str) -> str:
    """'2x+3=11'  ->  'x = 4'   (linear or quadratic with real roots)."""
    L, R = expr.split("=", 1)
    P = parse_poly(L); Q = parse_poly(R or "0")
    for k, v in Q.items(): P[k] = P.get(k, 0.0) - v
    a, b, c = P.get(2, 0.0), P.get(1, 0.0), P.get(0, 0.0)
    if max([k for k, v in P.items() if abs(v) > 1e-12] or [0]) > 2: raise ValueError("degree > 2")
    if abs(a) > 1e-12:
        D = b * b - 4 * a * c
        if D < 0: return "no real solution"
        x1, x2 = (-b + math.sqrt(D)) / (2 * a), (-b - math.sqrt(D)) / (2 * a)
        return f"x = {fmt_num(x1)}" if abs(x1 - x2) < 1e-9 else f"x = {fmt_num(x1)} or {fmt_num(x2)}"
    if abs(b) > 1e-12: return f"x = {fmt_num(-c / b)}"
    return "any x" if abs(c) < 1e-12 else "no solution"


def solve_expression(text: str, mode: str = "calc") -> Tuple[str, str]:
    """Recognised text -> (display text, result text).  Raises ValueError for unusable input."""
    t = text.replace("*", "×").replace("/", "÷").replace("^2", "²").replace("^3", "³")
    if mode == "math":
        if "=" in text and text.rstrip("=") != text.replace("=", "", 1) and text.split("=", 1)[1]:
            return t, solve_equation(text)
        return t.rstrip("="), simplify_poly(text.rstrip("="))
    v = evaluate(text)
    return t.rstrip("="), fmt_num(v)


# ─────────────────────────────── drawing assistant (natural language -> structured commands) ───────────────────────────────
# Command format:  {"action": "change_color", "target": "selected", "value": "red"}
#                  {"action": "add_effect",   "target": "selected", "effect": "stars"}
# The core app never depends on an LLM: DrawingAssistant uses built-in rules unless an optional provider
# (a callable  text, context -> list of command dicts)  is supplied.
KNOWN_ACTIONS = {"create", "change_color", "add_effect", "scale", "delete", "convert_3d", "transform", "clean", "enhance",
                 "recognize", "calculate", "undo", "redo", "clear", "launch", "blast", "gravity", "set_mode", "set_brush",
                 "pen_size", "move", "auto_clean", "help", "app_action", "unknown"}
TARGET_WORDS = {"left": "left", "right": "right", "top": "top", "upper": "top", "bottom": "bottom", "lower": "bottom",
                "all": "all", "everything": "all", "last": "last", "latest": "last", "first": "first", "selected": "selected",
                "this": "selected", "it": "selected", "that": "selected", "these": "all", "them": "all"}
EFFECT_WORDS = {"fire": "fire", "flame": "fire", "flames": "fire", "smoke": "smoke", "spark": "sparks", "sparks": "sparks",
                "snow": "snow", "star": "stars", "stars": "stars", "electric": "electricity", "electricity": "electricity",
                "lightning": "electricity", "plasma": "plasma", "dust": "dust", "magic": "magic", "confetti": "confetti",
                "bubble": "bubbles", "bubbles": "bubbles"}
NUM_WORDS = {"plus": "+", "minus": "-", "times": "*", "multiplied by": "*", "divided by": "/", "over": "/", "x": "*"}
# later phases register extra (regex, command) rules here
ASSISTANT_EXTRA_RULES: List[Tuple[str, dict]] = []


class DrawingAssistant:
    """Turns a sentence into a list of command dicts.  provider: optional callable(text, ctx) -> list[dict]."""
    def __init__(self, provider: Optional[Callable] = None):
        self.provider = provider

    # -- vocab ------------------------------------------------------------------------------------
    @staticmethod
    def _colors() -> Dict[str, str]:
        d = {v["name"].lower(): k for k, v in PALETTE.items()}
        d.update({"gold": d.get("yellow", "4"), "violet": d.get("purple", "6"), "cyan": d.get("cyan", "5")})
        return d

    def interpret(self, text: str, ctx: Optional[dict] = None) -> List[dict]:
        ctx = ctx or {}
        t = (text or "").strip()
        if not t: return []
        if self.provider is not None:
            try:
                out = self.provider(t, ctx)
                good = [c for c in (out or []) if isinstance(c, dict) and c.get("action") in KNOWN_ACTIONS]
                if good: return good
            except Exception:
                pass                                           # a broken provider must never break drawing
        return self._rules(t.lower(), ctx)

    # -- rule parser ------------------------------------------------------------------------------
    def _rules(self, s: str, ctx: dict) -> List[dict]:
        s = re.sub(r"[?!,;:]", " ", s); s = re.sub(r"\.(?=\s|$)", " ", s); s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"^(please|can you|could you|would you|hey|ok|okay)\s+", "", s)
        for pat, cmd in ASSISTANT_EXTRA_RULES:
            if re.search(pat, s): return [dict(cmd)]
        words = s.split()
        colors = self._colors()
        color = next((colors[w] for w in words if w in colors), None)
        cname = next((w for w in words if w in colors), None)
        target = next((TARGET_WORDS[w] for w in words if w in TARGET_WORDS), "selected")
        # --- arithmetic ---
        if re.search(r"\b(what is this|what is it|what did i draw|what am i drawing|recogni[sz]e|identify|what('s| is) that)\b", s):
            return [{"action": "recognize"}]
        m = re.match(r"^(?:calculate|compute|evaluate|what is|what's|whats|solve|math)\s+(.+)$", s)
        if m and re.search(r"\d|x", m.group(1)):
            e = m.group(1)
            for k, v in NUM_WORDS.items():
                if k != "x": e = e.replace(k, v)
            e = e.replace("×", "*").replace("÷", "/").replace("^", "^")
            return [{"action": "calculate", "value": e.replace(" ", "")}]
        # --- simple app verbs ---
        if re.fullmatch(r"(undo|go back|step back)( that)?", s): return [{"action": "undo"}]
        if re.fullmatch(r"(redo|do it again)", s): return [{"action": "redo"}]
        if re.fullmatch(r"(clear|clear all|clear everything|reset|start over|new canvas)", s): return [{"action": "clear"}]
        if re.fullmatch(r"(launch|go|release|let it go|drop (it|them))", s): return [{"action": "launch"}]
        if re.search(r"\b(blast|explode|explosion)\b", s): return [{"action": "blast"}]
        if re.search(r"\bgravity\b", s): return [{"action": "gravity"}]
        if re.fullmatch(r"(help|what can you do|commands)", s): return [{"action": "help"}]
        if re.search(r"\b(what is this|what is it|what did i draw|what am i drawing|recogni[sz]e|identify|what('s| is) that)\b", s):
            return [{"action": "recognize"}]
        if re.search(r"\bauto ?clean\b", s):
            return [{"action": "auto_clean", "value": not re.search(r"\b(off|disable|stop)\b", s)}]
        m = re.search(r"\b(2d|3d)\b.*\bmode\b|\bmode\b.*\b(2d|3d)\b|^(?:switch to |go to )?(2d|3d)$", s)
        if m: return [{"action": "set_mode", "value": next(g for g in m.groups() if g)}]
        # --- delete ---
        if re.search(r"\b(remove|delete|erase|get rid of|destroy)\b", s):
            if re.search(r"\b(effect|effects|particles|particle)\b", s): return [{"action": "add_effect", "target": target, "effect": "none"}]
            return [{"action": "delete", "target": target}]
        # --- convert / transform / 3D ---
        if re.search(r"\b(sphere|cube|pyramid|3d|three d|three-d)\b", s) and re.search(r"\b(turn|convert|make|change|into|to)\b", s) \
                and not re.search(r"\b(create|draw|add|build|paint)\b", s):
            return [{"action": "convert_3d", "target": target}]
        shape = self._shape(words)
        refers = any(w in ("this", "it", "that", "selected", "these", "them") for w in words) and not re.search(r"\b(create|draw|add|build|paint)\b", s)
        if refers:
            mm0 = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:x|times)\b", s)
            if re.search(r"\b(bigger|larger|enlarge|grow|increase|expand|magnify|twice|double)\b", s):
                f = 2.0 if re.search(r"\b(twice|double)\b", s) else (float(mm0.group(1)) if mm0 else 1.3)
                return [{"action": "scale", "target": target, "value": f}]
            if re.search(r"\b(smaller|shrink|reduce|decrease|minimi[sz]e|tiny|half)\b", s):
                return [{"action": "scale", "target": target, "value": 0.5 if "half" in s else 0.77}]
            if color: return [{"action": "change_color", "target": target, "value": cname}]
        m = re.search(r"\b(turn|convert|change|transform|morph|make)\b.*\b(into|to)\b\s+(?:an?\s+|the\s+)?(.+)$", s)
        if m and shape and not re.search(r"\b(create|draw|add|build|paint)\b", s):
            return [{"action": "transform", "target": target, "shape": shape}]
        # --- cleanup / enhance ---
        if re.search(r"\b(clean|clean up|cleanup|straighten|perfect(ly)?|tidy|neaten)\b", s): return [{"action": "clean", "target": target}]
        if re.search(r"\b(enhance|beautify|improve|polish|prettify|refine|smooth)\b", s): return [{"action": "enhance", "target": target}]
        # --- create ---
        effect = next((EFFECT_WORDS[w] for w in words if w in EFFECT_WORDS), None)
        if re.search(r"\b(create|draw|make|add|build|paint|give me|sketch|generate)\b", s) and shape \
                and not (effect and shape == "star" and re.search(r"\b(stars|particles|sparkles?)\b", s)):
            cmd = {"action": "create", "shape": shape, "target": "free"}
            if color: cmd["color"] = color
            pos = next((w for w in words if w in ("left", "right", "top", "bottom", "center", "centre", "middle")), None)
            if pos: cmd["position"] = {"centre": "center", "middle": "center"}.get(pos, pos)
            if re.search(r"\b(big|large|huge|giant)\b", s): cmd["size"] = 1.5
            if re.search(r"\b(small|tiny|little|mini)\b", s): cmd["size"] = 0.55
            return [cmd]
        # --- effects ---
        if effect and re.search(r"\b(add|apply|use|give|put|set|with|make|turn|enable|show|particles?|effect)\b", s):
            return [{"action": "add_effect", "target": target, "effect": effect}]
        # --- scale ---
        mm = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:x|times)\b", s)
        if re.search(r"\b(bigger|larger|enlarge|grow|increase|expand|magnify)\b", s) or re.search(r"\b(twice|double)\b", s):
            f = 2.0 if re.search(r"\b(twice|double)\b", s) else (float(mm.group(1)) if mm else 1.3)
            return [{"action": "scale", "target": target, "value": f}]
        if re.search(r"\b(smaller|shrink|reduce|decrease|minimi[sz]e|tiny|half)\b", s):
            return [{"action": "scale", "target": target, "value": 0.5 if "half" in s else 0.77}]
        # --- colour ---
        if color and re.search(r"\b(make|change|turn|paint|color|colour|set|use|switch|recolou?r)\b|^\w+$", s):
            return [{"action": "change_color", "target": target, "value": cname}]
        # --- brush / pen ---
        m = re.search(r"\b(?:pen|brush|line|stroke)\s*(?:size|width|thickness)?\s*(?:to|=)?\s*(\d+)\b", s)
        if m: return [{"action": "pen_size", "value": int(m.group(1))}]
        for k in BRUSH_ORDER:
            nm = BRUSHES[k].name.lower()
            if re.search(r"\b%s\b" % re.escape(nm), s) and re.search(r"\b(brush|pen|pencil|use|switch|draw with|set)\b|^%s$" % re.escape(nm), s):
                return [{"action": "set_brush", "value": k}]
        if re.search(r"\b(thicker|thick|bolder)\b", s): return [{"action": "pen_size", "value": "+"}]
        if re.search(r"\b(thinner|thin)\b", s): return [{"action": "pen_size", "value": "-"}]
        # --- move ---
        m = re.search(r"\bmove\b.*\b(left|right|up|down)\b", s)
        if m: return [{"action": "move", "target": target, "value": m.group(1)}]
        if shape and not effect: return [{"action": "create", "shape": shape, "target": "free"}] if len(words) <= 3 else [{"action": "unknown", "text": s}]
        return [{"action": "unknown", "text": s}]

    @staticmethod
    def _shape(words: List[str]) -> Optional[str]:
        txt = " ".join(words)
        for alias in sorted(ICON_ALIASES, key=len, reverse=True):
            if re.search(r"\b%s\b" % re.escape(alias), txt): return ICON_ALIASES[alias]
        for name in ICONS:
            if re.search(r"\b%s\b" % re.escape(name), txt): return name
        return None


ASSISTANT_HELP = ("Try: 'draw a blue spaceship' | 'make it red' | 'make it bigger' | 'add fire' | 'remove the left object' | "
                  "'turn this into a galaxy' | 'convert to a 3D sphere' | 'clean up' | 'what is this' | 'calculate 25+30'")


# ─────────────────────────────── AI panel + App mixin (recognition badge, clean/enhance, assistant, calculator) ───────────────────────────────
AI_CLUSTER_GAP_S = 4.0       # strokes drawn within this many seconds of each other (and close together) are one drawing
CALC_IDLE_S      = 1.4       # the calculator evaluates when the hand has been idle this long
AI_REPLY_S       = 7.0
AI_TAB_X         = WIN_W - 214 - 206 - 108 * 4 + 6


def cluster_suffix(strokes, times, gap_s: float = AI_CLUSTER_GAP_S) -> int:
    """Index of the first stroke of the drawing the LAST stroke belongs to (strokes: list of polylines, times: seconds)."""
    n = len(strokes)
    if n == 0: return 0
    start, bb = n - 1, _bbox(strokes[-1])
    for i in range(n - 2, -1, -1):
        if times[i + 1] - times[i] > gap_s: break
        b = _bbox(strokes[i])
        gx = max(b[0] - bb[2], bb[0] - b[2], 0.0); gy = max(b[1] - bb[3], bb[1] - b[3], 0.0)
        ref = 0.25 * max(bb[2] - bb[0], bb[3] - bb[1], b[2] - b[0], b[3] - b[1], 60.0)
        if math.hypot(gx, gy) > ref: break
        bb = (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3])); start = i
    return start


def group_drawings(strokes, times, gap_s: float = AI_CLUSTER_GAP_S) -> List[List[int]]:
    """All strokes -> drawings (lists of indices): strokes drawn close in time AND space belong together."""
    groups: List[List[int]] = []; boxes = []
    for i, s in enumerate(strokes):
        b = _bbox(s)
        if groups:
            gb = boxes[-1]; last = groups[-1][-1]
            gx = max(b[0] - gb[2], gb[0] - b[2], 0.0); gy = max(b[1] - gb[3], gb[1] - b[3], 0.0)
            ref = 0.25 * max(gb[2] - gb[0], gb[3] - gb[1], b[2] - b[0], b[3] - b[1], 60.0)
            if times[i] - times[last] <= gap_s and math.hypot(gx, gy) <= ref:
                groups[-1].append(i); boxes[-1] = (min(gb[0], b[0]), min(gb[1], b[1]), max(gb[2], b[2]), max(gb[3], b[3])); continue
        groups.append([i]); boxes.append(b)
    return groups


class AIPanel(FlyoutPanel):
    def __init__(self):
        super().__init__("AI", pygame.Rect(444, 64, 100, 32), 420, (190, 120, 255))

    def specs(self, app) -> List[RowSpec]:
        r = app.ai_rec
        seen = "-" if r is None else ("not recognised" if r.label == "unknown" else "%s  %d%%" % (recognition_label(r), round(r.conf * 100)))
        rows = [RowSpec(header="AI  recognise / clean / ask"),
                RowSpec(cells=[Cell("Seen", w=52), Cell(seen, w=240)], h=26),
                RowSpec(cells=[Cell("RECOGNIZE", "recognize", w=146), Cell("ASK  [ / ]", "ask", w=146)], h=30),
                RowSpec(cells=[Cell("CLEAN UP", "clean", w=146), Cell("ENHANCE", "enhance", w=146)], h=30),
                RowSpec(cells=[Cell("AUTO CLEAN: " + ("ON" if app.ai_auto_clean else "OFF"), "autoclean", w=296, on=app.ai_auto_clean)], h=30),
                RowSpec(cells=[Cell("CALC  [K]", "calc", w=146, on=(app.calc_mode == "calc")),
                               Cell("MATH", "math", w=146, on=(app.calc_mode == "math"))], h=30)]
        if app.calc_mode:
            rows.append(RowSpec(cells=[Cell("Write with your finger: 25+30", w=296)], h=24))
        if app.ask_reply and time.perf_counter() - app.ask_reply_t < 30:
            rows.append(RowSpec(cells=[Cell(app.ask_reply[:40], w=296)], h=24))
        return rows


class AIMixin:
    """Phase 7 behaviour of App.  All heavy work runs once per settled drawing, never per frame."""

    def _ai_init(self):
        self.assistant = DrawingAssistant()
        self.ai_auto_clean = False
        self.ai_rec: Optional[Recognition] = None
        self.ai_rec_ids: List[int] = []
        self.ai_panel = AIPanel()
        self.ask_active, self.ask_buf, self.ask_reply, self.ask_reply_t = False, "", "", 0.0
        self.calc_mode: Optional[str] = None
        self.calc_strokes: List[Tuple[list, float]] = []
        self.calc_res: Optional[dict] = None
        self._calc_final = False
        self._stroke_t: Dict[int, float] = {}
        self._last_stroke_t = 0.0
        self._ai_pending = False
        self._ai_undo: List[tuple] = []
        pygame.font.init()
        self._ai_f = pygame.font.SysFont("monospace", 15, bold=True)
        self._ai_big = pygame.font.SysFont("monospace", 34, bold=True)

    # ---- stroke bookkeeping --------------------------------------------------------------------
    def _ai_note_stroke(self, entry):
        now = time.perf_counter()
        self._stroke_t[id(entry[0])] = now
        self._last_stroke_t = now
        if self.calc_mode:
            if self.calc_strokes:
                E = _bbox(np.vstack([np.asarray(s, float) for s, _ in self.calc_strokes]))
                nb = _bbox(np.asarray(entry[0], float)); cy = (nb[1] + nb[3]) / 2; refh = max(E[3] - E[1], 60.0)
                if cy > E[3] + 0.35 * refh or cy < E[1] - 0.35 * refh:        # written on a new line
                    self.calc_strokes.clear(); self._calc_final = False
            if self._calc_final:
                self.calc_strokes.clear(); self._calc_final = False        # a finished '=' line: new line starts here
            self.calc_strokes.append((list(entry[0]), now))
            self.calc_res = None
        else:
            self._ai_pending = True
        if len(self._stroke_t) > 400:                                       # bounded
            live = {id(e[0]) for e in self.pending}
            self._stroke_t = {k: v for k, v in self._stroke_t.items() if k in live}

    def _ai_cluster(self) -> List[int]:
        if not self.pending: return []
        strokes = [e[0] for e in self.pending]
        now = time.perf_counter()
        times = [self._stroke_t.get(id(s), now) for s in strokes]
        return list(range(cluster_suffix(strokes, times), len(strokes)))

    def _ai_recognize_now(self) -> Optional[Recognition]:
        idx = self._ai_cluster()
        if not idx:
            self.ai_rec, self.ai_rec_ids = None, []
            return None
        S = [self.pending[i][0] for i in idx]
        self.ai_rec = recognize_drawing(S)
        self.ai_rec_ids = [id(s) for s in S]
        self.ai_rec.strokes = idx
        return self.ai_rec

    def _ai_rec_valid(self) -> bool:
        if self.ai_rec is None: return False
        live = {id(e[0]) for e in self.pending}
        return all(i in live for i in self.ai_rec_ids)

    # ---- per-frame ------------------------------------------------------------------------------
    def _ai_update(self, dt: float):
        if self.cur_stroke or self.stroke_ctx is not None: return
        now = time.perf_counter()
        if self.calc_mode:
            if self.calc_strokes and self.calc_res is None and now - self._last_stroke_t >= CALC_IDLE_S:
                self._calc_eval()
            return
        if self._ai_pending and now - self._last_stroke_t >= AI_SETTLE_S:
            self._ai_pending = False
            rec = self._ai_recognize_now()
            if rec is not None and self.ai_auto_clean and rec.label in PRIMITIVE_KINDS and rec.conf >= 0.7 and not self._drawing_locked():
                self.ai_clean(auto=True)
        if self.ai_rec is not None and not self._ai_rec_valid():
            self.ai_rec, self.ai_rec_ids = None, []

    # ---- render rendering -----------------------------------------------------------------------
    def _ai_render(self, screen):
        now = time.perf_counter()
        r = self.ai_rec
        if r is not None and r.label != "unknown" and not self.calc_mode and self._ai_rec_valid():
            x0, y0, x1, y1 = r.bbox
            txt = "%s  %d%%" % (recognition_label(r), round(r.conf * 100))
            im = self._ai_f.render(txt, True, (235, 220, 255))
            w, h = im.get_width() + 18, im.get_height() + 8
            bx = int(min(max((x0 + x1) / 2 - w / 2, 4), WIN_W - w - 4)); by = int(max(y0 - h - 10, 100))
            b = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(b, (24, 12, 44, 205), b.get_rect(), border_radius=9)
            pygame.draw.rect(b, (190, 120, 255, 235), b.get_rect(), 2, border_radius=9)
            screen.blit(b, (bx, by)); screen.blit(im, (bx + 9, by + 4))
            pygame.draw.rect(screen, (190, 120, 255), pygame.Rect(int(x0) - 6, int(y0) - 6, int(x1 - x0) + 12, int(y1 - y0) + 12), 1, border_radius=6)
        if self.calc_mode and self.calc_strokes:
            self._calc_render(screen)
        if self.calc_mode and not self.calc_strokes:
            hint = "AIR %s  -  write with your finger (e.g. 25+30 or 12/4).  [K] to leave" % self.calc_mode.upper()
            im = self._ai_f.render(hint, True, (190, 230, 255))
            screen.blit(im, (WIN_W // 2 - im.get_width() // 2, WIN_H - 70))
        reply_on = self.ask_reply and now - self.ask_reply_t < AI_REPLY_S
        if self.ask_active or reply_on:
            lines = []
            if reply_on: lines.append((self.ask_reply, (200, 235, 255)))
            if self.ask_active: lines.append(("ASK > " + self.ask_buf + ("_" if int(now * 2) % 2 == 0 else " "), (255, 255, 255)))
            h = 26 * len(lines) + 12; w = 760
            b = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(b, (10, 12, 26, 225), b.get_rect(), border_radius=10)
            pygame.draw.rect(b, (190, 120, 255, 235), b.get_rect(), 2, border_radius=10)
            screen.blit(b, (WIN_W // 2 - w // 2, WIN_H - 40 - h))
            for i, (t, c) in enumerate(lines):
                screen.blit(self._ai_f.render(t[:88], True, c), (WIN_W // 2 - w // 2 + 14, WIN_H - 40 - h + 8 + 26 * i))

    def _calc_render(self, screen):
        S = [s for s, _ in self.calc_strokes]
        bb = _bbox(np.vstack([np.asarray(s, float) for s in S]))
        res = self.calc_res
        if res is None:
            pygame.draw.rect(screen, (120, 200, 255), pygame.Rect(int(bb[0]) - 8, int(bb[1]) - 8, int(bb[2] - bb[0]) + 16, int(bb[3] - bb[1]) + 16), 1, border_radius=8)
            return
        col = (255, 120, 120) if res.get("error") else (120, 255, 190)
        line1 = res["text"]; line2 = ("= " + res["result"]) if not res.get("error") else "= ?  (" + res["error"] + ")"
        i1 = self._ai_f.render(line1, True, (200, 220, 255)); i2 = self._ai_big.render(line2, True, col)
        w = max(i1.get_width(), i2.get_width()) + 28; h = i1.get_height() + i2.get_height() + 18
        x = int(min(max(bb[0], 6), WIN_W - w - 6)); y = int(min(bb[3] + 16, WIN_H - h - 6))
        b = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(b, (8, 14, 26, 225), b.get_rect(), border_radius=12)
        pygame.draw.rect(b, (*col, 235), b.get_rect(), 2, border_radius=12)
        screen.blit(b, (x, y)); screen.blit(i1, (x + 14, y + 6)); screen.blit(i2, (x + 14, y + 10 + i1.get_height()))

    # ---- calculator ------------------------------------------------------------------------------
    def _calc_set_mode(self, mode: Optional[str]):
        self.calc_mode = mode if mode != self.calc_mode else None
        self.calc_strokes.clear(); self.calc_res = None; self._calc_final = False
        if self.calc_mode:
            self.notify.show(f"Air {self.calc_mode}: write with your finger, pause to evaluate", (150, 220, 255))
        else:
            self.notify.show("Air calculator off")

    def _calc_eval(self):
        S = [s for s, _ in self.calc_strokes]; T = [t for _, t in self.calc_strokes]
        text, syms = read_expression(S, T, self.calc_mode or "calc")
        weak = [c for c, cf in syms if cf < 0.4]
        final = text.endswith("=")
        res = {"text": text, "result": "", "error": ""}
        try:
            if not text: raise ValueError("nothing read")
            disp, out = solve_expression(text, self.calc_mode or "calc")
            res.update(text=disp, result=out)
            if weak: res["text"] += "   (check: unsure about %d symbol%s)" % (len(weak), "s" if len(weak) > 1 else "")
        except ZeroDivisionError:
            res.update(text=text.replace("*", "×").replace("/", "÷"), error="divide by zero")
        except Exception:
            res.update(text=text.replace("*", "×").replace("/", "÷"), error="can't read that")
        self.calc_res = res
        self._calc_final = final and not res["error"]
        if not res["error"]: self.notify.show(f"{res['text'].split('   ')[0]} = {res['result']}", (120, 255, 190))

    # ---- keyboard ---------------------------------------------------------------------------------
    def _ai_key(self, ev, k: str) -> bool:
        """Consume the key when the ASK box is open.  Returns True if consumed."""
        if not self.ask_active: return False
        if k in ("return", "enter"):
            text, self.ask_buf, self.ask_active = self.ask_buf, "", False
            if text.strip(): self.ai_run(text)
        elif k == "escape":
            self.ask_active, self.ask_buf = False, ""
        elif k == "backspace":
            self.ask_buf = self.ask_buf[:-1]
        else:
            ch = getattr(ev, "unicode", "")
            if ch and ch.isprintable() and len(self.ask_buf) < 80: self.ask_buf += ch
        return True

    def _ai_say(self, text: str):
        self.ask_reply, self.ask_reply_t = text, time.perf_counter()

    # ---- panel actions ----------------------------------------------------------------------------
    def ai_apply(self, key: str, arg=None):
        if key == "recognize": self._ai_say(self.ai_recognize_text())
        elif key == "clean": self.ai_clean()
        elif key == "enhance": self.ai_enhance()
        elif key == "autoclean":
            self.ai_auto_clean = not self.ai_auto_clean
            self.notify.show(f"Auto clean {'ON' if self.ai_auto_clean else 'OFF'}", (190, 120, 255))
        elif key == "ask":
            self.ask_active = True; self.ask_buf = ""; self.ai_panel.open = False
        elif key == "calc": self._calc_set_mode("calc")
        elif key == "math": self._calc_set_mode("math")

    def ai_recognize_text(self) -> str:
        if not self.pending:
            return "Nothing drawn on the canvas yet"
        r = self._ai_recognize_now()
        if r is None or r.label == "unknown": return "I can't tell what that is (%s)" % (r.detail if r else "nothing drawn")
        return "That looks like a %s  (%d%% match)" % (r.label, round(r.conf * 100))

    # ---- rendering a polyline with the active brush onto the canvas ---------------------------------------
    def _render_poly(self, pts, rgb, w, brush_key):
        br = BRUSHES.get(brush_key) or BRUSHES[BRUSH_ORDER[0]]
        ctx = br.new_ctx(rgb, w, pts[0], seed=random.getrandbits(32))
        self.layers.clear()
        prev = pts[0]
        for p in pts[1:]:
            for q in self._densify(prev, p): self.layers.draw(br.feed(ctx, q, ctx.w), ctx.passes)
            prev = p
        self.layers.flatten_into(self.canvas, ctx.passes)
        self.layers.clear()

    def _add_pending_poly(self, pts, rgb, w, bk):
        pts = [(int(x), int(y)) for x, y in pts]
        if len(pts) < 2: return None
        self.undo.push(self.canvas)
        self._render_poly(pts, rgb, w, bk)
        e = (pts, rgb, w, bk)
        self.pending.append(e)
        self.action_log.append("ink")
        self._stroke_t[id(pts)] = time.perf_counter()
        return e

    def _rebuild_canvas(self):
        self.canvas.fill((0, 0, 0, 0))
        for pts, rgb, w, bk in self.pending: self._render_poly(pts, rgb, w, bk)

    # ---- clean / enhance (original preserved by undo; enhance keeps the original underneath) ---------------
    def _ai_pick(self):
        if not self.pending:
            self.notify.show("Draw something first"); return None
        if not self._ai_rec_valid(): self._ai_recognize_now()
        r = self.ai_rec
        return r

    def ai_clean(self, auto: bool = False) -> bool:
        if self._drawing_locked(): self._warn_locked(self.lyr["drawing"].name); return False
        r = self._ai_pick()
        if r is None: return False
        idx = r.strokes; S = [self.pending[i][0] for i in idx]
        polys = clean_strokes(S, r)
        if not polys:
            if not auto: self.notify.show("Nothing to clean - shape not recognised", (255, 190, 120))
            return False
        k = len(idx); ent = self.pending[idx[0]]
        rgb, w, bk = ent[1], ent[2], ent[3]
        if k > len(self.undo._s) or idx[-1] != len(self.pending) - 1:
            return self._ai_enhance_add(polys, rgb, w, bk, "Cleaned (kept original)", r)
        fx = self._pending_effect_key(S[-1]) if hasattr(self, "_pending_effect_key") else None
        for _ in range(k):                                 # lift the originals off (exact snapshot restore)
            self.undo.pop(self.canvas); self.pending.pop()
            if self.action_log and self.action_log[-1] == "ink": self.action_log.pop()
        newe = [self._add_pending_poly(p, rgb, w, bk) for p in polys]
        newe = [e for e in newe if e]
        if newe:
            self.ai_rec = Recognition(r.label, r.conf, r.bbox, list(range(len(self.pending) - len(newe), len(self.pending))), r.detail)
            self.ai_rec_ids = [id(e[0]) for e in newe]
            if len(newe) == 1 and r.label in PRIMITIVE_KINDS:
                self._offer_conversion(newe[0][0], rgb, w, bk, "2d", 0.0, newe[0][0])
            else:
                self.fx_last = ("pending", newe[-1][0])
            if fx and len(newe) == 1: self.apply_effect(fx)
        self.notify.show(("Auto-cleaned: " if auto else "Cleaned: ") + r.label, (190, 120, 255))
        self._log("Clean: " + r.label)
        return True

    def ai_enhance(self) -> bool:
        if self._drawing_locked(): self._warn_locked(self.lyr["drawing"].name); return False
        r = self._ai_pick()
        if r is None: return False
        idx = r.strokes; S = [self.pending[i][0] for i in idx]
        polys = enhance_strokes(S, r)
        if not polys: self.notify.show("Nothing to enhance"); return False
        ent = self.pending[idx[0]]
        bk = "neon" if "neon" in BRUSHES else ent[3]
        return self._ai_enhance_add(polys, ent[1], max(ent[2], 3), bk, "Enhanced: " + ("smoothed" if r.label == "unknown" else r.label), r)

    def _ai_enhance_add(self, polys, rgb, w, bk, label, r) -> bool:
        added = [self._add_pending_poly(p, rgb, w, bk) for p in polys]
        added = [e for e in added if e]
        if not added: return False
        self.fx_last = ("pending", added[-1][0])
        self.notify.show(label, (190, 120, 255))
        self._log(label)
        return True

    # ---- generic helpers for assistant edits ---------------------------------------------------------------
    def _ai_items(self):
        items = []
        if self.pending:
            now = time.perf_counter()
            strokes = [e[0] for e in self.pending]
            times = [self._stroke_t.get(id(s), now) for s in strokes]
            for grp in group_drawings(strokes, times):
                ents = [self.pending[i] for i in grp]
                P = np.vstack([np.asarray(e[0], float) for e in ents])
                bb = _bbox(P); items.append(("pending", ents, ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)))
        for pb in self.phys.bodies:
            if pb.visible:
                cx, cy, _ = self.phys.object_bounds(pb); items.append(("body", pb, (cx, cy)))
        for s in self.scene3d.solids:
            if s.visible and not s.building:
                cx, cy, _, _ = self.scene3d.center_screen(s); items.append(("solid", s, (cx, cy)))
        return items

    def _ai_targets(self, target: str):
        items = self._ai_items()
        if not items: return []
        if target == "all": return items
        if target in ("left", "right", "top", "bottom"):
            key = {"left": lambda it: it[2][0], "right": lambda it: -it[2][0], "top": lambda it: it[2][1], "bottom": lambda it: -it[2][1]}[target]
            return [min(items, key=key)]
        if target == "first": return [items[0]]
        if target == "last": return [items[-1]]
        t = self.edit_target()
        if t is not None:
            for it in items:
                if it[1] is t[1] or (t[0] == "pending" and it[0] == "pending" and any(e[0] is t[1] for e in it[1])): return [it]
        return [items[-1]]

    def _body_points(self, pb):
        P = []
        for sh in pb.shapes:
            if isinstance(sh, pymunk.Segment):
                a, b = pb.body.local_to_world(sh.a), pb.body.local_to_world(sh.b)
                P += [(a.x, a.y), (b.x, b.y)]
        return [P[0]] + [P[i] for i in range(1, len(P), 2)] if P else []

    def _ai_push_undo(self):
        self._ai_undo.append((self._snapshot(), list(self.undo._s), list(self._redo)))
        if len(self._ai_undo) > 20: del self._ai_undo[0]

    def _ai_done(self, label: str):
        self.action_log.append("ai")
        self._log(label)

    def _ai_undo_step(self) -> bool:
        if not (self.action_log and self.action_log[-1] == "ai" and self._ai_undo): return False
        snap, ustack, redo = self._ai_undo.pop()
        cur = (self._snapshot(), list(self.undo._s), list(redo))
        self._restore(snap)
        self.undo._s.clear(); self.undo._s.extend(ustack)
        self._redo[:] = redo
        self._redo.append(("ai", cur))
        self.notify.show("Undone (AI edit)")
        self._log("Undo", clear_redo=False)
        return True

    def _ai_redo_step(self, rec) -> None:
        _, (snap, ustack, redo) = rec
        keep = list(self._redo)
        self._ai_undo.append((self._snapshot(), list(self.undo._s), list(keep)))
        self._restore(snap)
        self.undo._s.clear(); self.undo._s.extend(ustack)
        self._redo[:] = keep
        self.action_log.append("ai")
        self.notify.show("Redone (AI edit)")
        self._log("Redo", clear_redo=False)

    def _ai_modify_pending(self, changes: Dict[int, Optional[tuple]]):
        """changes: id(pts) -> new entry (or None to delete)."""
        new, old_fx, old_shape = [], dict(self.pending_fx), dict(self.pending_shape)
        self.pending_fx.clear(); self.pending_shape.clear()
        for e in self.pending:
            ch = changes.get(id(e[0]), e)
            if ch is None: continue
            new.append(ch)
            if ch is not e:
                fx = old_fx.get(id(e[0]))
                if fx: self.pending_fx[id(ch[0])] = dict(fx, pts=ch[0], ring=False, geom=None)
            else:
                if id(e[0]) in old_fx: self.pending_fx[id(e[0])] = old_fx[id(e[0])]
                if id(e[0]) in old_shape: self.pending_shape[id(e[0])] = old_shape[id(e[0])]
        self.pending[:] = new
        self._rebuild_canvas()
        self.undo._s.clear()
        self.fx_last = ("pending", new[-1][0]) if new else None

    def _ai_free_box(self, size: float, position: Optional[str]) -> Tuple[float, float, float, float]:
        cands = {"left": (WIN_W * 0.27, WIN_H * 0.55), "right": (WIN_W * 0.73, WIN_H * 0.55), "top": (WIN_W * 0.5, WIN_H * 0.34),
                 "bottom": (WIN_W * 0.5, WIN_H * 0.72), "center": (WIN_W * 0.5, WIN_H * 0.55)}
        order = [position] if position in cands else ["center", "left", "right", "top", "bottom"]
        used = [it[2] for it in self._ai_items()]
        pick = cands[order[0]]
        for nm in order:
            c = cands[nm]
            if all(math.hypot(c[0] - u[0], c[1] - u[1]) > size * 0.8 for u in used): pick = c; break
        h = size * 0.5
        return (pick[0] - h, pick[1] - h, pick[0] + h, pick[1] + h)

    # ---- assistant ----------------------------------------------------------------------------------------------
    def ai_run(self, text: str):
        cmds = self.assistant.interpret(text, {"selected": self.edit_target() is not None})
        replies = []
        for c in cmds:
            try:
                replies.append(self.ai_exec(c))
            except Exception as e:                                      # the assistant must never take the app down
                replies.append("That didn't work (%s)" % e.__class__.__name__)
        self._ai_say(" | ".join(r for r in replies if r) or "…")

    def ai_exec(self, c: dict) -> str:
        a = c.get("action"); tgt = c.get("target", "selected")
        if a in ("create", "change_color", "scale", "delete", "transform", "move", "clean", "enhance") and self._drawing_locked() and a in ("create", "clean", "enhance"):
            self._warn_locked(self.lyr["drawing"].name); return "Drawing layer is locked"
        if a == "create":
            shape = c["shape"]; size = 230.0 * float(c.get("size", 1.0))
            box = self._ai_free_box(size, c.get("position"))
            polys = icon_strokes(shape, box)
            if not polys: return "I don't know how to draw a %s" % shape
            key = c.get("color") or self.color_key
            rgb = PALETTE[key]["rgb"]
            added = [self._add_pending_poly(p, rgb, self.pen_w, self.brush_key) for p in polys]
            added = [e for e in added if e]
            if len(added) == 1 and shape in PRIMITIVE_KINDS:
                self._offer_conversion(added[0][0], rgb, self.pen_w, self.brush_key, "2d", 0.0, added[0][0])
            elif added:
                self.fx_last = ("pending", added[-1][0])
            self._ai_pending = False
            self._ai_recognize_now()
            self._log(f"Assistant: {shape}")
            return "Drew a %s%s" % (PALETTE[key]["name"].lower() + " " if c.get("color") else "", shape)
        if a == "clean": return "Cleaned" if self.ai_clean() else ""
        if a == "enhance": return "Enhanced" if self.ai_enhance() else ""
        if a == "recognize": return self.ai_recognize_text()
        if a == "calculate":
            v = c.get("value", "")
            try:
                if "x" in v or "=" in v: return "%s  ->  %s" % (v, solve_expression(v, "math")[1])
                return "%s = %s" % (v.replace("*", "×").replace("/", "÷"), fmt_num(evaluate(v)))
            except ZeroDivisionError: return "Can't divide by zero"
            except Exception: return "I couldn't work that out"
        if a == "undo": self._do_action("undo"); return "Undone"
        if a == "redo": self._do_action("redo"); return "Redone"
        if a == "clear": self._do_action("clear"); return "Cleared"
        if a == "launch": self._do_action("launch"); return "Launched"
        if a == "blast": self._do_action("explode"); return "Boom"
        if a == "gravity": self._do_action("gravity"); return "Gravity changed"
        if a == "help": return ASSISTANT_HELP
        if a == "auto_clean":
            self.ai_auto_clean = bool(c.get("value")); return "Auto clean %s" % ("ON" if self.ai_auto_clean else "OFF")
        if a == "set_mode":
            want = c.get("value") == "3d"
            if want != self.mode3d: self._toggle_mode()
            return "%s mode" % c.get("value").upper()
        if a == "set_brush":
            self._select_brush(c["value"]); return "Brush: %s" % BRUSHES[c["value"]].name
        if a == "pen_size":
            v = c["value"]
            self.pen_w = max(MIN_PEN_W, min(MAX_PEN_W, self.pen_w + (2 if v == "+" else -2) if v in ("+", "-") else int(v)))
            return "Pen size %d" % self.pen_w
        if a == "app_action":
            self._do_action(c.get("value")); return "Done"
        if a == "convert_3d":
            if self.suggest is not None and self.convert_suggestion(): return "Converted to 3D"
            return "Draw a shape first, then I can turn it into 3D"
        if a == "add_effect":
            ts = self._ai_targets(tgt)
            if not ts: return "Draw something first"
            key = c.get("effect", "none")
            ok = 0
            for kind, obj, _ in ts:
                for tgt_obj in ([("pending", e[0]) for e in obj] if kind == "pending" else [(kind, obj)]):
                    self.fx_last = tgt_obj
                    if self.grab.body is None and self.grab3d.body is None and self.apply_effect(key): ok += 1
            return ("Applied %s" % key) if (ok and key != "none") else ("Effect removed" if ok else "Couldn't apply that")
        # ---- edits that need a target ----
        ts = self._ai_targets(tgt)
        if not ts: return "Nothing to edit yet - draw something first"
        if a == "change_color":
            name = c.get("value"); key = DrawingAssistant._colors().get(name)
            if key is None: return "I don't know the colour %s" % name
            rgb = PALETTE[key]["rgb"]
            if self.lyr.locked("objects") and any(t[0] != "pending" for t in ts): self._warn_locked("Physics Objects"); return "Layer locked"
            self._ai_push_undo()
            ch = {}
            for kind, obj, _ in ts:
                if kind == "body": obj.color = rgb; obj.sprite = None
                elif kind == "solid": obj.rgb = rgb; obj._spr = None
                else:
                    for e in obj: ch[id(e[0])] = (e[0], rgb, e[2], e[3])
            if ch: self._ai_modify_pending(ch)
            self.color_key = key
            self._ai_done(f"Assistant: {name}")
            return "Changed to %s" % name
        if a in ("scale", "move", "delete", "transform"):
            if self.lyr.locked("objects") and any(t[0] != "pending" for t in ts): self._warn_locked("Physics Objects"); return "Layer locked"
            self._ai_push_undo()
            ch = {}
            msg = ""
            for kind, obj, cen in list(ts):
                if a == "delete":
                    if kind == "body":
                        if self.grab.body is obj: self.grab.reset()
                        self.phys.remove_body(obj)
                    elif kind == "solid":
                        if self.grab3d.body is obj: self.grab3d.reset()
                        self.scene3d.remove(obj)
                    else:
                        for e in obj: ch[id(e[0])] = None
                    msg = "Removed"
                elif a == "scale":
                    f = float(c.get("value", 1.3))
                    if kind == "solid": obj.scale = max(0.2, min(5.0, obj.scale * f)); obj._spr = None
                    elif kind == "body":
                        pts = self._body_points(obj); cx, cy = cen
                        self._replace_body(obj, [(cx + (x - cx) * f, cy + (y - cy) * f) for x, y in pts])
                    else:
                        cx, cy = cen
                        for e in obj:
                            ch[id(e[0])] = ([(int(cx + (x - cx) * f), int(cy + (y - cy) * f)) for x, y in e[0]], e[1], e[2], e[3])
                    msg = "Bigger" if f > 1 else "Smaller"
                elif a == "move":
                    dx, dy = {"left": (-120, 0), "right": (120, 0), "up": (0, -120), "down": (0, 120)}[c["value"]]
                    if kind == "solid": obj.pos[0] += dx; obj.pos[1] += dy
                    elif kind == "body": self._replace_body(obj, [(x + dx, y + dy) for x, y in self._body_points(obj)])
                    else:
                        for e in obj: ch[id(e[0])] = ([(int(x + dx), int(y + dy)) for x, y in e[0]], e[1], e[2], e[3])
                    msg = "Moved " + c["value"]
                else:                                                      # transform into an icon of the same size/colour
                    shape = c["shape"]
                    if kind == "pending": rgb, w, bk = obj[0][1], obj[0][2], obj[0][3]; box = _box_of([e[0] for e in obj], 0)
                    elif kind == "body":
                        P = self._body_points(obj); rgb, w, bk = obj.color, obj.width, obj.brush; box = _box_of([P], 0)
                        self.phys.remove_body(obj)
                    else:
                        self.notify.show("Convert 3D objects with 'turn it into a 3D sphere' instead"); continue
                    polys = icon_strokes(shape, (box[0], box[1], max(box[2], box[0] + 60), max(box[3], box[1] + 60)))
                    if not polys: return "I don't know how to draw a %s" % shape
                    if kind == "pending":
                        for e in obj: ch[id(e[0])] = None
                    self._pending_new = getattr(self, "_pending_new", []) + [(polys, rgb, w, bk)]
                    msg = "Turned into a %s" % shape
            if ch: self._ai_modify_pending(ch)
            for polys, rgb, w, bk in getattr(self, "_pending_new", []):
                for p in polys: self._add_pending_poly(p, rgb, w, bk)
            self._pending_new = []
            self.fx_last = None if a == "delete" else self.fx_last
            self._ai_done("Assistant: " + a)
            return msg
        return "I didn't get that.  " + ASSISTANT_HELP

    def _replace_body(self, pb, pts):
        props = {f: getattr(pb, f) for f in ("effect", "ring", "shape", "opacity", "glow", "visible", "mass_mult", "elast", "fric", "gscale")}
        vel, av, col, wid, bk, fr = tuple(pb.body.velocity), pb.body.angular_velocity, pb.color, pb.width, pb.brush, pb.frozen
        self.phys.remove_body(pb)
        self.phys.add_stroke([(float(x), float(y)) for x, y in pts], col, wid, bk)
        nb = self.phys.bodies[-1]
        for k, v in props.items(): setattr(nb, k, v)
        self.phys.apply_props(nb)
        nb.body.velocity = vel; nb.body.angular_velocity = av
        if fr: self.phys.set_frozen(nb, True)
        return nb




# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 8 — FUTURISTIC UI:  themes · HUD (off / classic / system) · mode system · animated panels
#  Every background is procedural (no image files).  Static layers are built once and cached; only a handful of
#  cheap draw calls run per frame, so a theme costs ~1-3 ms.
# ══════════════════════════════════════════════════════════════════════════════
THEME_ORDER = ["camera", "cyberpunk", "space", "lab", "holo"]
THEMES: Dict[str, dict] = {
    "camera":    dict(label="CAMERA",      accent=(120, 200, 255), cam=1.0),
    "cyberpunk": dict(label="CYBERPUNK",   accent=(255, 70, 210),  cam=0.16),
    "space":     dict(label="SPACE",       accent=(150, 170, 255), cam=0.0),
    "lab":       dict(label="LAB",         accent=(90, 255, 170),  cam=0.0),
    "holo":      dict(label="HOLOGRAPHIC", accent=(80, 210, 255),  cam=0.5),
}
MODE_ORDER = ["DRAW", "SELECT", "GRAB", "EDIT", "PHYSICS", "3D", "AI", "GAME", "SETTINGS"]
MODE_HINT = {
    "DRAW": "point with one finger to draw",
    "SELECT": "point at an object (0.4 s) to select it - drawing paused",
    "GRAB": "pinch thumb+middle to grab / throw - drawing paused",
    "EDIT": "edit panel open - point at an object to select it",
    "PHYSICS": "every stroke becomes a physics object the moment you lift your finger",
    "3D": "3D drawing: move your hand toward the camera for depth",
    "AI": "AI panel open - draw, then pause to recognise",
    "GAME": "game mode",
    "SETTINGS": "themes / HUD / modes",
}
HUD_MODES = ["classic", "system", "off"]
UI_STATE = {"anim": True}


def ui_pulse(base: int, amp: int = 26, speed: float = 3.0) -> int:
    """Breathing alpha for panel borders (animated UI); constant when animation is off."""
    if not UI_STATE["anim"]: return base
    return max(0, min(255, int(base + amp * math.sin(time.perf_counter() * speed))))


def _lerp_col(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _vgradient(w, h, top, bottom) -> pygame.Surface:
    s = pygame.Surface((w, h))
    for y in range(h):
        pygame.draw.line(s, _lerp_col(top, bottom, y / max(1, h - 1)), (0, y), (w, y))
    return s


class ThemeBackground:
    """Procedural AR-style backgrounds.  render() returns False for the plain CAMERA theme (caller draws the feed)."""
    def __init__(self):
        self._cache: Dict[str, pygame.Surface] = {}
        self._font = None
        self._rng = random.Random(7)
        self._stars = [(self._rng.random() * WIN_W, self._rng.random() * WIN_H, self._rng.random(), self._rng.uniform(0.2, 1.0),
                        self._rng.uniform(0, 6.28)) for _ in range(150)]
        self.fps_hist: Deque[float] = deque(maxlen=90)
        self.spd_hist: Deque[float] = deque(maxlen=90)
        self._hist_t = 0.0
        self.last_ms = 0.0

    # ---- cached static layers -------------------------------------------------------------
    def _surf(self, key: str, build):
        s = self._cache.get(key)
        if s is None: s = self._cache[key] = build()
        return s

    def _build_cyber(self) -> pygame.Surface:
        s = _vgradient(WIN_W, WIN_H, (8, 4, 22), (40, 8, 56))
        hz = int(WIN_H * 0.56)
        sun = pygame.Surface((WIN_W, hz), pygame.SRCALPHA)
        cx, cy, R = WIN_W // 2, hz - 10, 170
        for r in range(R, 0, -3):
            f = r / R
            pygame.draw.circle(sun, (*_lerp_col((255, 220, 90), (255, 40, 150), 1 - f * f), 120), (cx, cy), r)
        for k in range(8):                                    # retro slats across the lower half of the sun
            y = cy + 4 + k * 13
            pygame.draw.rect(sun, (0, 0, 0, 0), (cx - R, y, 2 * R, 3 + k))
        s.blit(sun, (0, 0))
        pygame.draw.line(s, (255, 90, 220), (0, hz), (WIN_W, hz), 2)
        return s

    def _build_space(self) -> pygame.Surface:
        rng = np.random.default_rng(11)
        w, h = 160, 90
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.zeros((h, w, 3), float) + np.array([4, 5, 14.0])
        for col, n in (((120, 40, 190), 4), ((30, 90, 200), 4), ((200, 50, 130), 3)):
            for _ in range(n):
                cx, cy = rng.uniform(0, w), rng.uniform(0, h)
                sx, sy = rng.uniform(12, 34), rng.uniform(8, 22)
                g = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)) * rng.uniform(0.35, 0.8)
                img += g[..., None] * np.array(col, float) * 0.55
        img = np.clip(img, 0, 255).astype(np.uint8)
        small = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        return pygame.transform.smoothscale(small, (WIN_W, WIN_H)).convert()

    def _planet(self, r, base, ring=False) -> pygame.Surface:
        pad = int(r * 0.9) if ring else 2
        s = pygame.Surface((2 * r + 2 * pad, 2 * r + 2 * pad), pygame.SRCALPHA)
        c = (r + pad, r + pad)
        if ring:
            pygame.draw.ellipse(s, (*_lerp_col(base, (255, 255, 255), 0.3), 110), (c[0] - int(r * 1.8), c[1] - int(r * 0.45), int(r * 3.6), int(r * 0.9)), 3)
        for i in range(r, 0, -2):
            f = i / r
            pygame.draw.circle(s, (*_lerp_col(tuple(int(b * 1.25) for b in base), tuple(int(b * 0.18) for b in base), f), 255),
                               (c[0] - int((1 - f) * r * 0.35), c[1] - int((1 - f) * r * 0.35)), i)
        if ring:
            pygame.draw.arc(s, (*_lerp_col(base, (255, 255, 255), 0.3), 160), (c[0] - int(r * 1.8), c[1] - int(r * 0.45), int(r * 3.6), int(r * 0.9)), math.pi, math.tau, 3)
        return s

    def _build_lab(self) -> pygame.Surface:
        s = pygame.Surface((WIN_W, WIN_H)); s.fill((8, 16, 20))
        for x in range(0, WIN_W, 20): pygame.draw.line(s, (14, 34, 40) if x % 100 else (22, 62, 70), (x, 0), (x, WIN_H))
        for y in range(0, WIN_H, 20): pygame.draw.line(s, (14, 34, 40) if y % 100 else (22, 62, 70), (0, y), (WIN_W, y))
        f = pygame.font.SysFont("monospace", 11)
        for x in range(0, WIN_W, 100):
            pygame.draw.line(s, (90, 255, 170), (x, 0), (x, 12)); s.blit(f.render(str(x), True, (90, 200, 150)), (x + 3, 14))
        for y in range(0, WIN_H, 100):
            pygame.draw.line(s, (90, 255, 170), (0, y), (12, y)); s.blit(f.render(str(y), True, (90, 200, 150)), (16, y + 2))
        for x in range(0, WIN_W, 20):
            if x % 100: pygame.draw.line(s, (60, 170, 120), (x, 0), (x, 5))
        for y in range(0, WIN_H, 20):
            if y % 100: pygame.draw.line(s, (60, 170, 120), (0, y), (5, y))
        return s

    def _build_scanlines(self) -> pygame.Surface:
        s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for y in range(0, WIN_H, 3): pygame.draw.line(s, (0, 0, 0, 38), (0, y), (WIN_W, y))
        return s

    def _build_hex(self) -> pygame.Surface:
        s = pygame.Surface((WIN_W, WIN_H)); s.fill((0, 40, 78))             # constant blue tint ...
        r = 34; dx = r * 1.5; dy = r * math.sqrt(3) / 2
        col = 0
        x = 0.0
        while x < WIN_W + r:                                                # ... plus a faint hex grid (added, not alpha-blended)
            y = (dy if col % 2 else 0.0)
            while y < WIN_H + r:
                pts = [(x + r * math.cos(math.radians(60 * i)), y + r * math.sin(math.radians(60 * i))) for i in range(6)]
                pygame.draw.polygon(s, (30, 90, 130), pts, 1)
                y += 2 * dy
            x += dx; col += 1
        return s

    def _scan(self, screen, strength: int = 24):
        """CRT scanlines: one cached overlay blit (cheaper than 240 per-line fills)."""
        screen.blit(self._surf("scan", self._build_scanlines), (0, 0))

    def _build_band(self) -> pygame.Surface:
        s = pygame.Surface((WIN_W, 70), pygame.SRCALPHA)
        for y in range(70):
            a = int(46 * math.sin(math.pi * y / 69) ** 2)
            pygame.draw.line(s, (120, 220, 255, a), (0, y), (WIN_W, y))
        return s

    # ---- per-frame ---------------------------------------------------------------------------------
    def sample(self, fps: float, speed: float):
        now = time.perf_counter()
        if now - self._hist_t >= 0.12:
            self._hist_t = now
            self.fps_hist.append(fps); self.spd_hist.append(speed)

    def render(self, screen: pygame.Surface, name: str, t: float, cam: Optional[pygame.Surface], tips: dict, accent) -> bool:
        if name == "camera" or name not in THEMES: return False
        t0 = time.perf_counter()
        cfg = THEMES[name]
        if name == "cyberpunk":
            screen.blit(self._surf("cyber", self._build_cyber), (0, 0))
            self._cyber_grid(screen, t)
            self._ghost(screen, cam, cfg["cam"])
            self._scan(screen, 22)
        elif name == "space":
            screen.blit(self._surf("space", self._build_space), (0, 0))
            self._space_stars(screen, t)
            self._space_planets(screen, t)
        elif name == "lab":
            screen.blit(self._surf("lab", self._build_lab), (0, 0))
            self._lab_graphs(screen, tips)
        else:                                                  # holographic
            screen.fill((2, 8, 20))
            self._ghost(screen, cam, cfg["cam"])
            screen.blit(self._surf("hex", self._build_hex), (0, 0), special_flags=pygame.BLEND_RGB_ADD)
            self._holo_rings(screen, t)
            y = int((t * 90) % (WIN_H + 70)) - 70
            screen.blit(self._surf("band", self._build_band), (0, y))
            self._scan(screen, 26)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return True

    def _ghost(self, screen, cam, k, tint=None):
        if cam is None or k <= 0: return
        cam.set_alpha(int(255 * k))
        screen.blit(cam, (0, 0))
        cam.set_alpha(255)
        if tint is not None:
            screen.fill(tint, special_flags=pygame.BLEND_RGB_ADD)

    def _cyber_grid(self, screen, t):
        hz = int(WIN_H * 0.56); cx = WIN_W // 2
        n = 9
        for i in range(-n, n + 1):                              # perspective verticals
            x_bot = cx + i * (WIN_W / 7.0)
            pygame.draw.line(screen, (90, 20, 120), (cx + i * 40, hz), (x_bot, WIN_H), 3)
            pygame.draw.line(screen, (255, 70, 210), (cx + i * 40, hz), (x_bot, WIN_H), 1)
        ph = (t * 0.45) % 1.0
        for k in range(12):                                     # horizontals racing toward the viewer
            f = ((k + ph) / 12.0) ** 2.2
            y = int(hz + f * (WIN_H - hz))
            pygame.draw.line(screen, (20, 140, 170), (0, y), (WIN_W, y), 3)
            pygame.draw.line(screen, (40, 230, 255), (0, y), (WIN_W, y), 1)

    def _space_stars(self, screen, t):
        for x, y, z, b, ph in self._stars:
            xx = (x - t * (3 + 14 * z)) % WIN_W
            tw = 0.55 + 0.45 * math.sin(t * (1.0 + 2.0 * z) + ph)
            v = int(255 * b * tw)
            if z > 0.8: pygame.draw.circle(screen, (v // 2, v // 2, v), (int(xx), int(y)), 2)
            else: screen.set_at((int(xx), int(y)), (v, v, v))

    def _space_planets(self, screen, t):
        specs = (("p1", 74, (70, 120, 210), False, 0.18, 0.62, 3.0), ("p2", 38, (200, 110, 70), False, 0.78, 0.2, 5.0),
                 ("p3", 26, (160, 190, 170), True, 0.9, 0.78, 7.0))
        for key, r, col, ring, fx, fy, speed in specs:
            s = self._surf(key, lambda r=r, col=col, ring=ring: self._planet(r, col, ring))
            x = (fx * WIN_W - t * speed) % (WIN_W + s.get_width()) - s.get_width() // 2
            y = fy * WIN_H + math.sin(t * 0.2 + r) * 6
            screen.blit(s, (int(x - s.get_width() // 2), int(y - s.get_height() // 2)))

    def _lab_graphs(self, screen, tips):
        if self._font is None: self._font = pygame.font.SysFont("monospace", 12)
        f = self._font
        x0, y0, w, h = WIN_W - 250, WIN_H - 190, 230, 120
        p = pygame.Surface((w + 20, h + 54), pygame.SRCALPHA)
        pygame.draw.rect(p, (6, 18, 22, 190), p.get_rect(), border_radius=8)
        pygame.draw.rect(p, (90, 255, 170, 200), p.get_rect(), 1, border_radius=8)
        screen.blit(p, (x0 - 10, y0 - 28))
        for gi, (lab, hist, col, ymax) in enumerate((("FPS", self.fps_hist, (90, 255, 170), 70.0), ("HAND px/s", self.spd_hist, (255, 200, 80), 2500.0))):
            gy = y0 + gi * 66
            screen.blit(f.render(lab + (" %.0f" % hist[-1] if hist else ""), True, col), (x0, gy - 22 if gi == 0 else gy - 16))
            pygame.draw.rect(screen, (30, 70, 70), (x0, gy, w, 52), 1)
            if len(hist) > 1:
                pts = [(x0 + i * w / (hist.maxlen - 1), gy + 51 - min(1.0, v / ymax) * 50) for i, v in enumerate(hist)]
                pygame.draw.lines(screen, col, False, pts, 2)
        for lab in ("Right", "Left"):                           # live measurement crosshair at each fingertip
            tip = tips.get(lab)
            if not tip: continue
            c = (90, 255, 170) if lab == "Right" else (120, 200, 255)
            for a, b in (((tip[0], 20), (tip[0], tip[1] - 14)), ((20, tip[1]), (tip[0] - 14, tip[1]))):
                pygame.draw.line(screen, (c[0] // 3, c[1] // 3, c[2] // 3), a, b, 1)
            screen.blit(f.render("%s x:%d y:%d" % (lab[0], tip[0], tip[1]), True, c), (min(tip[0] + 16, WIN_W - 130), min(tip[1] + 14, WIN_H - 20)))

    def _holo_rings(self, screen, t):
        for i, (cx, cy, r) in enumerate(((120, 130, 64), (WIN_W - 130, 150, 52), (WIN_W // 2, WIN_H - 120, 40))):
            cy2 = cy + math.sin(t * 0.9 + i) * 8
            a0 = t * (0.7 + 0.2 * i)
            for k in range(3):
                pygame.draw.arc(screen, (80, 210, 255), (cx - r - k * 10, cy2 - r - k * 10, 2 * (r + k * 10), 2 * (r + k * 10)), a0 + k, a0 + k + 1.6, 2)
            pygame.draw.circle(screen, (40, 120, 170), (cx, int(cy2)), 3)


def draw_hud_system(surface: pygame.Surface, app, hands) -> None:
    """Compact 'AIR DRAWING SYSTEM' read-out (spec #43)."""
    fonts = app._ui_fonts
    acc = THEMES[app.theme]["accent"]
    rh = next((h for h in hands if h.label == "Right"), None); lh = next((h for h in hands if h.label == "Left"), None)
    g = (rh or lh).gesture if (rh or lh) else "NONE"
    br = BRUSHES.get(app.brush_key)
    rows = [("FPS", "%d" % round(app.fps)), ("HANDS", str(len(hands))), ("GESTURE", g), ("TOOL", (br.name if br else app.brush_key).upper()),
            ("BRUSH", "%dpx" % app.pen_w), ("COLOR", PALETTE[app.color_key]["name"].upper()),
            ("GRAVITY", str(GRAVITY_LABELS[app.phys.gravity_idx]).upper()),
            ("OBJECTS", str(len(app.phys.bodies) + len(app.scene3d.solids))),
            ("MODE", "%s  %s" % (app.mode_name, "3D" if app.mode3d else "2D"))]
    w, h = 252, 38 + len(rows) * 20 + 8
    p = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(p, (6, 12, 24, 170), p.get_rect(), border_radius=12)
    pygame.draw.rect(p, (*acc, ui_pulse(210, 20)), p.get_rect(), 2, border_radius=12)
    surface.blit(p, (10, 10))
    surface.blit(fonts[0].render("AIR DRAWING SYSTEM", True, acc), (22, 18))
    pygame.draw.line(surface, acc, (22, 38), (w - 2, 38), 1)
    for i, (k, v) in enumerate(rows):
        y = 46 + i * 20
        surface.blit(fonts[1].render(k + ":", True, (150, 170, 200)), (22, y))
        surface.blit(fonts[1].render(v, True, (235, 245, 255)), (112, y))


class UIPanel(FlyoutPanel):
    def __init__(self):
        super().__init__("UI", pygame.Rect(338, 64, 100, 32), 420, (110, 220, 255))

    def specs(self, app) -> List[RowSpec]:
        rows = [RowSpec(header="UI  theme / HUD / mode")]
        names = THEME_ORDER
        for chunk in (names[:3], names[3:]):
            rows.append(RowSpec(cells=[Cell(("CAMERA" if n == "camera" else "CYBER" if n == "cyberpunk" else "SPACE" if n == "space"
                                             else "LAB" if n == "lab" else "HOLO"), "theme", n, w=94, on=(app.theme == n)) for n in chunk], h=30))
        rows.append(RowSpec(cells=[Cell("HUD: %s  [`]" % app.hud_mode.upper(), "hud", None, w=190, on=(app.hud_mode != "off")),
                                   Cell("ANIM: " + ("ON" if UI_STATE["anim"] else "OFF"), "anim", None, w=100, on=UI_STATE["anim"]),
                                   Cell("GUIDE  Ctrl+/", "help", None, w=100)], h=30))
        rows.append(RowSpec(header="MODE  (Ctrl+1..9)"))
        for i in range(0, len(MODE_ORDER), 3):
            rows.append(RowSpec(cells=[Cell(m, "mode", m, w=94, on=(app.mode_name == m)) for m in MODE_ORDER[i:i + 3]], h=30))
        rows.append(RowSpec(cells=[Cell(MODE_HINT.get(app.mode_name, "")[:46], w=296)], h=24))
        return rows


class UIMixin:
    """Phase 8 behaviour of App: theme backgrounds, HUD modes, central mode system."""

    def _ui_init(self):
        self.theme = "camera"
        self.hud_mode = "classic"
        self.mode_name = "DRAW"
        self.ui_panel = UIPanel()
        self._ext.append((self.ui_panel, self.ui_apply))
        self.theme_bg = ThemeBackground()
        pygame.font.init()
        self._ui_fonts = (pygame.font.SysFont("monospace", 17, bold=True), pygame.font.SysFont("monospace", 15))
        self._tips: dict = {}
        self._tip_prev: dict = {}
        self._hand_speed = 0.0
        self._sel_dwell = 0.0
        self._sel_cand = None

    # ---- themes / hud ----------------------------------------------------------------------------
    def set_theme(self, name: str):
        if name not in THEMES: return
        self.theme = name
        self.notify.show("Theme: " + THEMES[name]["label"], THEMES[name]["accent"])
        self._log("Theme: " + THEMES[name]["label"])

    def cycle_theme(self):
        self.set_theme(THEME_ORDER[(THEME_ORDER.index(self.theme) + 1) % len(THEME_ORDER)])

    def cycle_hud(self):
        self.hud_mode = HUD_MODES[(HUD_MODES.index(self.hud_mode) + 1) % len(HUD_MODES)]
        self.notify.show("HUD: " + self.hud_mode.upper())

    def _themed_background(self, frame) -> bool:
        """Draw the active theme's background.  False -> the caller draws the camera feed as before."""
        if self.theme == "camera": return False
        cam = None
        if THEMES[self.theme]["cam"] > 0:
            cam = self._bg_surface(frame)
        return self.theme_bg.render(self.screen, self.theme, time.perf_counter(), cam, self._tips, THEMES[self.theme]["accent"])

    def _ui_track(self, dt: float, rtip, ltip):
        self._tips = {"Right": rtip, "Left": ltip}
        sp = 0.0
        for k, tip in self._tips.items():
            pv = self._tip_prev.get(k)
            if tip and pv and dt > 0: sp = max(sp, math.hypot(tip[0] - pv[0], tip[1] - pv[1]) / dt)
            self._tip_prev[k] = tip
        self._hand_speed += 0.25 * (sp - self._hand_speed)
        if self.theme == "lab": self.theme_bg.sample(self.fps, self._hand_speed)

    # ---- mode system -------------------------------------------------------------------------------
    def _mode_blocks_ink(self) -> bool:
        return self.mode_name in ("SELECT", "GRAB", "EDIT", "SETTINGS")

    def set_mode(self, name: str, announce: bool = True):
        if name not in MODE_ORDER or name == self.mode_name: return
        if name == "3D" and not ENABLE_3D:
            self.notify.show("3D is disabled (ENABLE_3D)"); return
        if self.cur_stroke: self._commit()
        prev = self.mode_name
        self.mode_name = name
        if prev == "3D" and name != "3D" and self.mode3d: self._toggle_mode(False)
        for p in (self.brush_panel, self.fx_panel, self.edit_panel, self.layer_panel, self.time_panel) + tuple(p for p, _ in self._ext):
            p.open = False
        if name == "3D":
            if not self.mode3d: self._toggle_mode(True)
            self.mode_name = "3D"
        elif name == "EDIT": self.edit_panel.open = True
        elif name == "AI": self.ai_panel.open = True
        elif name == "SETTINGS": self.ui_panel.open = True
        elif name == "GAME" and hasattr(self, "_game_enter"): self._game_enter()
        if announce:
            self.notify.show("MODE: %s - %s" % (name, MODE_HINT[name]), (150, 220, 255))
        self._log("Mode: " + name)

    def cycle_mode(self):
        self.set_mode(MODE_ORDER[(MODE_ORDER.index(self.mode_name) + 1) % len(MODE_ORDER)])

    def ui_apply(self, key: str, arg=None):
        if key == "theme": self.set_theme(arg)
        elif key == "hud": self.cycle_hud()
        elif key == "help":
            self.toggle_help()
        elif key == "anim":
            UI_STATE["anim"] = not UI_STATE["anim"]
            self.notify.show("UI animation " + ("ON" if UI_STATE["anim"] else "OFF"))
        elif key == "mode": self.set_mode(arg)

    def _select_update(self, dt: float, rtip, rg: str):
        """SELECT / EDIT mode: hold the pointing finger over an object for 0.4 s to select it."""
        if self.mode_name not in ("SELECT", "EDIT") or rtip is None or rg not in ("DRAW", "IDLE"):
            self._sel_dwell, self._sel_cand = 0.0, None; return
        best, bd = None, 80.0
        for kind, obj, cen in self._ai_items():
            d = math.hypot(cen[0] - rtip[0], cen[1] - rtip[1])
            if kind == "pending": d = min(d, self._dist_to_entries(obj, rtip))
            if d < bd: best, bd = (kind, obj), d
        if best is None:
            self._sel_dwell, self._sel_cand = 0.0, None; return
        key = id(best[1][0][0]) if best[0] == "pending" else id(best[1])
        if key != self._sel_cand: self._sel_cand, self._sel_dwell = key, 0.0
        self._sel_dwell += dt
        if self._sel_dwell >= 0.4:
            kind, obj = best
            self.fx_last = ("pending", obj[-1][0]) if kind == "pending" else (kind, obj)
            if self._sel_dwell < 0.4 + dt * 1.5: self.notify.show("Selected", (255, 170, 70))

    @staticmethod
    def _dist_to_entries(entries, p) -> float:
        best = 1e9
        for e in entries:
            P = np.asarray(e[0], float)
            if len(P): best = min(best, float(np.min(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]))))
        return best

    _GLOW_CACHE: dict = {}

    def _theme_hand_glow(self, hands):
        """'Glowing hands': an additive accent halo on every fingertip while a non-camera theme is active."""
        if self.theme == "camera": return
        acc = THEMES[self.theme]["accent"]
        g = self._GLOW_CACHE.get(acc)
        if g is None:
            R = 46
            g = pygame.Surface((2 * R, 2 * R))
            for r in range(R, 0, -2):
                f = (1 - r / R) ** 1.6
                pygame.draw.circle(g, tuple(int(c * f * 0.55) for c in acc), (R, R), r)
            self._GLOW_CACHE[acc] = g
        R = g.get_width() // 2
        for hd in hands:
            tip = self._cursor_pt(hd)
            if tip: self.screen.blit(g, (tip[0] - R, tip[1] - R), special_flags=pygame.BLEND_RGB_ADD)




# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 9 (part A) — pure helpers: symmetry · generative art · project (de)serialisation · SVG · GIF · challenge metrics
#  No App dependency, so every function here is unit-tested on its own.
# ══════════════════════════════════════════════════════════════════════════════
import json, base64, zipfile, queue, struct

SYM_MODES = ["off", "horizontal", "vertical", "radial", "kaleidoscope"]
SYM_COUNTS = [4, 6, 8, 12, 16]
SYM_CENTER = (WIN_W / 2.0, WIN_H / 2.0)
PROJECT_DIR = "projects"
EXPORT_DIR = "exports"
PROJECT_EXT = ".airdraw"
AUTOSAVE_S = 30.0
MAX_PROJECT_BYTES = 200 * 1024 * 1024


def _rot(p, a, c):
    ca, sa = math.cos(a), math.sin(a)
    x, y = p[0] - c[0], p[1] - c[1]
    return (c[0] + x * ca - y * sa, c[1] + x * sa + y * ca)


def symmetry_copies(pts, mode: str, n: int = 8, center=SYM_CENTER) -> List[List[Tuple[int, int]]]:
    """The extra strokes a symmetry mode adds for one drawn stroke (the original is not included).
    horizontal: mirrored across a horizontal axis (top <-> bottom)   vertical: across a vertical axis (left <-> right)
    radial: n-fold rotation   kaleidoscope: n-fold rotation of the stroke AND of its mirror image."""
    cx, cy = center
    out: List[list] = []
    if mode == "horizontal":
        out.append([(x, 2 * cy - y) for x, y in pts])
    elif mode == "vertical":
        out.append([(2 * cx - x, y) for x, y in pts])
    elif mode == "radial":
        for k in range(1, n): out.append([_rot(p, math.tau * k / n, center) for p in pts])
    elif mode == "kaleidoscope":
        mir = [(2 * cx - x, y) for x, y in pts]
        for k in range(n):
            if k: out.append([_rot(p, math.tau * k / n, center) for p in pts])
            out.append([_rot(p, math.tau * k / n, center) for p in mir])
    return [[(int(round(x)), int(round(y))) for x, y in c] for c in out]


# ─────────────────────────────── generative art ───────────────────────────────
GEN_MAX_POLYS = 240
GEN_MAX_POINTS = 6000


def _resample_poly(P, n):
    P = np.asarray(P, float)
    if len(P) < 2: return P
    seg = np.hypot(*np.diff(P, axis=0).T); cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1e-9: return P[:1]
    u = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(u, cum, P[:, 0]), np.interp(u, cum, P[:, 1])], 1)


def _gen_tree(a, b, strength):
    """Fractal tree grown from the end of a line stroke (branch angle 27 deg, length ratio 0.72)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a; L = float(np.hypot(*d))
    if L < 8: return []
    depth = 2 + strength                              # strength 1..5  ->  3..7 levels
    out = []
    def grow(p, ang, ln, lvl):
        q = p + ln * np.array([math.cos(ang), math.sin(ang)])
        out.append(([tuple(p), tuple(q)], lvl / depth))
        if lvl >= depth or ln < 4: return
        for da in (-0.47, 0.47):
            grow(q, ang + da, ln * 0.72, lvl + 1)
    grow(b, math.atan2(d[1], d[0]), L * 0.7, 1)
    return out


def _gen_mandala(c, r, strength, folds, rot):
    layers = strength + 1
    out = []
    for i in range(1, layers + 1):
        ri = r * i / layers
        twist = math.radians(rot) * i
        pet = max(4, folds)
        for k in range(pet):
            a = twist + math.tau * k / pet
            w = math.tau / pet * 0.5
            pts = []
            for t in np.linspace(-1, 1, 14):               # petal = lens between two arcs
                rr = ri * (0.55 + 0.45 * math.cos(t * math.pi / 2))
                aa = a + t * w
                pts.append((c[0] + rr * math.cos(aa), c[1] + rr * math.sin(aa)))
            pts += [(c[0] + ri * 0.3 * math.cos(a), c[1] + ri * 0.3 * math.sin(a))]
            out.append((pts, i / layers))
        out.append((list(zip(c[0] + ri * 0.22 * np.cos(np.linspace(0, math.tau, 24)), c[1] + ri * 0.22 * np.sin(np.linspace(0, math.tau, 24)))), i / layers))
    return out


def _gen_sierpinski(V, strength, rot):
    V = [np.asarray(v, float) for v in V[:3]]
    cen = sum(V) / 3.0
    if rot:
        a = math.radians(rot)
        V = [np.array(_rot(v, a, cen)) for v in V]
    out = []
    depth = max(1, min(5, strength))
    def rec(A, B, C, lvl):
        if lvl == 0:
            out.append(([tuple(A), tuple(B), tuple(C), tuple(A)], 0.0)); return
        AB, BC, CA = (A + B) / 2, (B + C) / 2, (C + A) / 2
        rec(A, AB, CA, lvl - 1); rec(AB, B, BC, lvl - 1); rec(CA, BC, C, lvl - 1)
    rec(V[0], V[1], V[2], depth)
    n = len(out)
    return [(p, i / max(1, n)) for i, (p, _) in enumerate(out)]


def _gen_rosette(P, strength, folds, rot):
    P = np.asarray(P, float)
    c = P.mean(0)
    P = _resample_poly(P, 40)
    layers = strength + 1
    out = []
    for L in range(layers):
        s = 1.0 - 0.8 * L / max(1, layers)
        for k in range(folds):
            a = math.tau * k / folds + math.radians(rot) * L
            Q = [_rot(c + (p - c) * s, a, c) for p in P]
            out.append((Q, L / layers))
    return out


def generate_pattern(pts, label: str, strength: int = 3, folds: int = 8, rotation: float = 0.0,
                     outline=None) -> List[Tuple[List[Tuple[int, int]], float]]:
    """One simple stroke -> a list of (polyline, hue_shift 0..1).
       line / open curve -> fractal tree   circle -> mandala   triangle -> Sierpinski   anything else -> rosette."""
    P = np.asarray(pts, float)
    if len(P) < 4: return []
    strength = max(1, min(5, int(strength))); folds = max(2, min(16, int(folds)))
    bb = _bbox(P); cen = P.mean(0)
    if label in ("line", "open", "unknown") and np.hypot(*(P[0] - P[-1])) > 0.5 * math.hypot(bb[2] - bb[0], bb[3] - bb[1]):
        raw = _gen_tree(P[0], P[-1], strength)
    elif label == "circle" or label == "ellipse":
        r = float(np.mean(np.hypot(*(P - cen).T)))
        raw = _gen_mandala(cen, r, strength, folds, rotation)
    elif label == "triangle" and outline is not None and len(outline) >= 3:
        raw = _gen_sierpinski(outline, strength, rotation)
    else:
        raw = _gen_rosette(P, strength, folds, rotation)
    out, total = [], 0
    for poly, hue in raw:
        if len(out) >= GEN_MAX_POLYS or total >= GEN_MAX_POINTS: break
        q = [(int(round(x)), int(round(y))) for x, y in poly]
        if len(q) >= 2:
            out.append((q, hue)); total += len(q)
    return out


def hue_shift(rgb, h: float, amount: float = 0.35):
    hh, ss, vv = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    r, g, b = colorsys.hsv_to_rgb((hh + amount * h) % 1.0, max(ss, 0.55), max(vv, 0.85))
    return (int(r * 255), int(g * 255), int(b * 255))


# ─────────────────────────────── safe project (de)serialisation ───────────────────────────────
_DATACLASSES = {"ShapeResult": ShapeResult}


def enc(o):
    """Python object -> JSON-able, with explicit tags for tuples, arrays and bytes.  No pickle, no eval."""
    if isinstance(o, np.ndarray): return {"__nd__": o.tolist(), "dt": str(o.dtype)}
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, (bytes, bytearray)): return {"__b__": base64.b64encode(bytes(o)).decode("ascii")}
    if isinstance(o, tuple): return {"__t__": [enc(x) for x in o]}
    if isinstance(o, list): return [enc(x) for x in o]
    if isinstance(o, dict): return {str(k): enc(v) for k, v in o.items()}
    if dataclasses.is_dataclass(o) and not isinstance(o, type) and type(o).__name__ in _DATACLASSES:
        return {"__dc__": type(o).__name__, "f": {f.name: enc(getattr(o, f.name)) for f in dataclasses.fields(o)}}
    if o is None or isinstance(o, (bool, int, float, str)): return o
    raise TypeError("cannot serialise %s" % type(o).__name__)


def dec(o):
    if isinstance(o, list): return [dec(x) for x in o]
    if isinstance(o, dict):
        if "__nd__" in o: return np.array(o["__nd__"], dtype=o.get("dt", "float64"))
        if "__b__" in o: return base64.b64decode(o["__b__"])
        if "__t__" in o: return tuple(dec(x) for x in o["__t__"])
        if "__dc__" in o:
            cls = _DATACLASSES.get(o["__dc__"])
            if cls is None: raise ValueError("unknown record type")
            return cls(**{k: dec(v) for k, v in o["f"].items()})
        return {k: dec(v) for k, v in o.items()}
    return o


def write_airdraw(path: str, snap: dict, preview: Optional[bytes] = None, meta: Optional[dict] = None):
    """A .airdraw file is a zip: project.json (+ preview.png).  Written to a temp name first, then renamed."""
    doc = {"format": "airdraw", "version": 1, "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "meta": meta or {}, "snapshot": enc(snap)}
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps(doc, separators=(",", ":")))
        if preview: z.writestr("preview.png", preview)
    os.replace(tmp, path)


def read_airdraw(path: str, with_meta: bool = False):
    """Validate and load.  Raises ValueError with a readable message for anything that is not a good project."""
    try:
        with zipfile.ZipFile(path) as z:
            info = z.getinfo("project.json")
            if info.file_size > MAX_PROJECT_BYTES: raise ValueError("project file is too large")
            raw = z.read("project.json")
    except (zipfile.BadZipFile, KeyError, OSError) as e:
        raise ValueError("not a valid .airdraw project (%s)" % e.__class__.__name__)
    try:
        doc = json.loads(raw)
    except Exception:
        raise ValueError("project data is corrupt")
    if not isinstance(doc, dict) or doc.get("format") != "airdraw": raise ValueError("not an Air Drawing project")
    if int(doc.get("version", 0)) > 1: raise ValueError("project was made by a newer version")
    snap = dec(doc.get("snapshot"))
    for k in ("canvas", "pending", "bodies", "solids", "settings", "layers"):
        if not isinstance(snap, dict) or k not in snap: raise ValueError("project is missing '%s'" % k)
    return (snap, doc.get("meta") or {}) if with_meta else snap


# ─────────────────────────────── SVG / JSON export ───────────────────────────────
GLOW_BRUSHES = {"neon", "fire", "galaxy", "electric", "plasma"}


def _hex(rgb): return "#%02x%02x%02x" % tuple(int(max(0, min(255, c))) for c in rgb[:3])


def svg_document(items, w: int = WIN_W, h: int = WIN_H, bg: Optional[Tuple[int, int, int]] = (14, 16, 24)) -> str:
    """items: dicts {pts, rgb, width, brush, opacity}.  Pure vector output (polylines with round caps)."""
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (w, h, w, h),
           '<defs><filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="4" result="b"/>'
           '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>']
    if bg is not None: out.append('<rect width="100%%" height="100%%" fill="%s"/>' % _hex(bg))
    for it in items:
        P = it["pts"]
        if len(P) < 2: continue
        d = " ".join("%.1f,%.1f" % (x, y) for x, y in P)
        flt = ' filter="url(#glow)"' if it.get("brush") in GLOW_BRUSHES else ""
        op = it.get("opacity", 1.0)
        out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="%s" stroke-linecap="round" stroke-linejoin="round"%s%s/>'
                   % (d, _hex(it["rgb"]), max(1, int(it.get("width", 4))), ("" if op >= 0.999 else ' opacity="%.2f"' % op), flt))
    out.append("</svg>")
    return "\n".join(out)


# ─────────────────────────────── GIF writer (Pillow when present, otherwise this encoder) ───────────────────────────────
def _web_palette() -> np.ndarray:
    pal = np.zeros((256, 3), np.uint8)
    i = 0
    for r in range(6):
        for g in range(6):
            for b in range(6):
                pal[i] = (r * 51, g * 51, b * 51); i += 1
    for k in range(40): pal[216 + k] = (int(k * 255 / 39),) * 3          # grey ramp for the remaining slots
    return pal


def _quantise(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.int32)
    r, g, b = (f[..., 0] * 5 + 127) // 255, (f[..., 1] * 5 + 127) // 255, (f[..., 2] * 5 + 127) // 255
    return (r * 36 + g * 6 + b).astype(np.uint8)


def _lzw(indices: bytes, min_code: int = 8) -> bytes:
    clear, eoi = 1 << min_code, (1 << min_code) + 1
    out = bytearray(); buf = 0; nbits = 0
    code_size = min_code + 1; nxt = eoi + 1
    table = {bytes([i]): i for i in range(clear)}
    def emit(code):
        nonlocal buf, nbits
        buf |= code << nbits; nbits += code_size
        while nbits >= 8:
            out.append(buf & 0xFF); buf >>= 8; nbits -= 8
    emit(clear)
    cur = b""
    for byte in indices:
        nb = cur + bytes([byte])
        if nb in table: cur = nb; continue
        emit(table[cur])
        if nxt < 4096:
            table[nb] = nxt; nxt += 1
            if nxt - 1 == (1 << code_size) and code_size < 12: code_size += 1
        else:
            emit(clear); table = {bytes([i]): i for i in range(clear)}; nxt = eoi + 1; code_size = min_code + 1
        cur = bytes([byte])
    if cur: emit(table[cur])
    emit(eoi)
    if nbits: out.append(buf & 0xFF)
    return bytes(out)


def encode_gif(frames: List[np.ndarray], path: str, fps: float = 8.0) -> str:
    """frames: list of HxWx3 uint8 RGB arrays (same size).  Returns the path.  Looping, ~fps frames per second."""
    if not frames: raise ValueError("no frames")
    try:
        from PIL import Image
        ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=False)
        return path
    except ImportError:
        pass
    H, W = frames[0].shape[:2]
    delay = max(2, int(round(100.0 / fps)))
    pal = _web_palette()
    buf = bytearray(b"GIF89a")
    buf += struct.pack("<HHBBB", W, H, 0xF7, 0, 0) + pal.tobytes()
    buf += b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00"
    for fr in frames:
        buf += b"\x21\xF9\x04\x04" + struct.pack("<H", delay) + b"\x00\x00"
        buf += b"\x2C" + struct.pack("<HHHHB", 0, 0, W, H, 0)
        data = _lzw(_quantise(fr).tobytes())
        buf += bytes([8])
        for i in range(0, len(data), 255):
            chunk = data[i:i + 255]; buf += bytes([len(chunk)]) + chunk
        buf += b"\x00"
    buf += b"\x3B"
    with open(path, "wb") as fh: fh.write(buf)
    return path


# ─────────────────────────────── drawing-challenge metrics (every number is explainable) ───────────────────────────────
CHALLENGES = [("circle", "CIRCLE", 2.5), ("square", "SQUARE", 4.0), ("triangle", "TRIANGLE", 3.5), ("star", "STAR", 5.0), ("line", "STRAIGHT LINE", 1.5)]
EQUIV = {"circle": ("circle", "ellipse"), "square": ("square", "rectangle", "quad"), "triangle": ("triangle",), "star": ("star",), "line": ("line",)}


def _dist_to_polyline(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """distance of each point of P to the polyline Q."""
    A, B = Q[:-1][None, :, :], Q[1:][None, :, :]
    AB = B - A; L2 = np.maximum((AB ** 2).sum(-1), 1e-9)
    t = np.clip((((P[:, None, :] - A) * AB).sum(-1)) / L2, 0, 1)
    proj = A + t[..., None] * AB
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1)).min(1)


def score_challenge(target: str, pts, duration: float) -> dict:
    """Shape accuracy, speed, smoothness, completion - each with the plain-language reason behind the number."""
    goal = next(g for k, _, g in CHALLENGES if k == target)
    P = np.asarray(pts, float)
    res = {"target": target, "ok": False, "recognised": "nothing", "lines": []}
    if len(P) < 5: res["lines"] = ["Stroke too short"]; return res
    bb = _bbox(P); diag = math.hypot(bb[2] - bb[0], bb[3] - bb[1]); diag = max(diag, 1.0)
    # --- what shape is it?
    if target == "line":
        sv = np.linalg.svd(P - P.mean(0), compute_uv=False)
        kind = "line" if sv[0] > 30 and sv[1] / sv[0] < 0.12 else "curve"
        ideal = np.array([P[0], P[-1]]) if kind == "line" else None
    else:
        try: r = ShapeRecognizer.recognize(P.tolist())
        except Exception: r = None
        kind = r.kind if r is not None else "unknown"
        ideal = np.asarray(r.outline, float) if (r is not None and kind in EQUIV[target]) else None
        if ideal is not None and len(ideal) >= 3 and np.hypot(*(ideal[0] - ideal[-1])) > 1e-6: ideal = np.vstack([ideal, ideal[:1]])      # outlines are open: close the last side
    res["recognised"] = kind
    ok = (kind in EQUIV[target])
    res["ok"] = ok
    if ok and ideal is not None and len(ideal) >= 2:
        dev = float(_dist_to_polyline(P, ideal).mean())
        acc = _clip01(1.0 - (dev / (0.5 * diag)) / 0.15)
        res["lines"].append("Shape Accuracy: %d%%  (average %.1f px away from the ideal %s; 0%% at 15%% of its size)" % (round(acc * 100), dev, target))
    else:
        acc = 0.0
        res["lines"].append("Shape Accuracy: 0%%  (it was read as %s, not %s)" % (kind, target))
    res["accuracy"] = acc
    spd = _clip01(goal / max(duration, goal))
    res["speed"] = spd
    res["lines"].append("Speed: %d%%  (drawn in %.1f s; full marks at %.1f s or less)" % (round(spd * 100), duration, goal))
    sm = smooth_stroke(P.tolist(), rounds=1, win=7)
    S = np.asarray(sm, float); wob = float(np.sqrt((_dist_to_polyline(P, S) ** 2).mean())) if len(S) >= 2 else 0.0
    smooth = _clip01(1.0 - (wob / diag) / 0.04)
    res["smoothness"] = smooth
    res["lines"].append("Smoothness: %d%%  (hand wobble %.1f px; 0%% at 4%% of the shape size)" % (round(smooth * 100), wob))
    if target != "line":
        per = max(_plen(P), 1.0); gap = float(np.hypot(*(P[0] - P[-1])))
        comp = _clip01(1.0 - (gap / per) / 0.15)
        res["completion"] = comp
        res["lines"].append("Completion: %d%%  (start and end %.0f px apart; 0%% at 15%% of the outline)" % (round(comp * 100), gap))
    else:
        res["completion"] = 1.0
    res["duration"] = duration
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 9 (part B) — recorder · game modes
# ══════════════════════════════════════════════════════════════════════════════
REC_MAX_S = 120.0
GIF_FPS, GIF_MAX_FRAMES, GIF_SIZE = 8.0, 120, (480, 270)
MP4_FPS, MP4_SIZE = 24.0, (960, 540)


class Recorder:
    """Screen recorder.  MP4/AVI: frames go to a writer thread (the game loop only copies the screen bytes).
    GIF: small frames are kept in memory and encoded when recording stops."""
    def __init__(self, done_cb: Callable[[str, bool], None]):
        self.kind: Optional[str] = None
        self.path = ""
        self.t0 = 0.0
        self._last = 0.0
        self._q: "queue.Queue" = queue.Queue(maxsize=10)
        self._th: Optional[threading.Thread] = None
        self._frames: List[np.ndarray] = []
        self.dropped = 0
        self.count = 0
        self._cb = done_cb
        self.last_thread: Optional[threading.Thread] = None

    @property
    def active(self) -> bool: return self.kind is not None

    def elapsed(self) -> float: return time.perf_counter() - self.t0 if self.active else 0.0

    def start(self, kind: str, path: str) -> bool:
        if self.active: return False
        self.kind, self.path, self.t0, self._last, self.dropped, self.count = kind, path, time.perf_counter(), 0.0, 0, 0
        self._frames = []
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if kind == "mp4":
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), MP4_FPS, MP4_SIZE)
            if not vw.isOpened():                                   # no mp4 encoder in this OpenCV build: fall back to AVI
                self.path = path = os.path.splitext(path)[0] + ".avi"
                vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), MP4_FPS, MP4_SIZE)
            if not vw.isOpened():
                self.kind = None; return False
            self._q = queue.Queue(maxsize=10)
            self._th = threading.Thread(target=self._pump, args=(vw,), daemon=True); self._th.start()
        return True

    def _pump(self, vw):
        try:
            while True:
                item = self._q.get()
                if item is None: break
                raw, w, h = item
                a = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
                vw.write(cv2.cvtColor(cv2.resize(a, MP4_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        finally:
            vw.release()

    def capture(self, screen: pygame.Surface):
        if not self.active: return
        now = time.perf_counter()
        if now - self.t0 > REC_MAX_S:
            self.stop(); return
        per = 1.0 / (MP4_FPS if self.kind == "mp4" else GIF_FPS)
        if now - self._last < per: return
        self._last = now
        w, h = screen.get_size()
        if self.kind == "mp4":
            try: self._q.put_nowait((pygame.image.tostring(screen, "RGB"), w, h)); self.count += 1
            except queue.Full: self.dropped += 1
        else:
            a = np.frombuffer(pygame.image.tostring(screen, "RGB"), np.uint8).reshape(h, w, 3)
            self._frames.append(cv2.resize(a, GIF_SIZE, interpolation=cv2.INTER_AREA)); self.count += 1
            if len(self._frames) >= GIF_MAX_FRAMES: self.stop()

    def stop(self):
        if not self.active: return
        kind, path, frames = self.kind, self.path, self._frames
        self.kind = None; self._frames = []
        if kind == "mp4":
            def fin():
                self._q.put(None)
                if self._th is not None: self._th.join(timeout=60)
                self._cb(path, os.path.exists(path) and os.path.getsize(path) > 0)
            self.last_thread = threading.Thread(target=fin, daemon=True); self.last_thread.start()
        else:
            def enc_gif():
                try: encode_gif(frames, path, GIF_FPS); self._cb(path, True)
                except Exception: self._cb(path, False)
            self.last_thread = threading.Thread(target=enc_gif, daemon=True); self.last_thread.start()


# ─────────────────────────────── games ───────────────────────────────
SCORES_FILE = "game_scores.json"


def _load_scores() -> dict:
    try:
        with open(os.path.join(PROJECT_DIR, SCORES_FILE), encoding="utf-8") as fh:
            d = json.load(fh)
        return {k: int(v) for k, v in d.items() if k in ("target", "draw", "bridge")}
    except Exception:
        return {}


def _save_scores(d: dict):
    try:
        os.makedirs(PROJECT_DIR, exist_ok=True)
        with open(os.path.join(PROJECT_DIR, SCORES_FILE), "w", encoding="utf-8") as fh: json.dump(d, fh)
    except Exception:
        pass


class GameBase:
    kind = ""
    title = ""
    help = ""
    def __init__(self, app): self.app = app; self.over = False; self.score = 0; self.lines: List[str] = []
    def update(self, dt: float): pass
    def render(self, s: pygame.Surface): pass
    def on_stroke(self, entry, dur: float): pass
    def on_launch(self): pass
    def next(self): pass
    def teardown(self): pass


class TargetGame(GameBase):
    kind, title = "target", "TARGET HUNT"
    help = "Draw a shape, SPACE (or open palm) to launch it, touch the glowing targets.  45 s."
    DUR, R = 45.0, 38

    def __init__(self, app):
        super().__init__(app)
        self.t = self.DUR; self.combo = 0; self.hits = 0; self.pulse = 0.0; self.targets: List[List[float]] = []
        for _ in range(3): self._spawn()

    def _spawn(self):
        for _ in range(30):
            p = [random.uniform(160, WIN_W - 160), random.uniform(190, WIN_H - 220)]
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 3 * self.R for q in self.targets): break
        if len(self.targets) >= 3: self.targets.pop(0)
        self.targets.append(p)

    def update(self, dt):
        self.pulse += dt
        if self.over: return
        self.t -= dt
        pts = [pt for pb in self.app.phys.bodies for pt in self.app._body_points(pb)]
        for tg in list(self.targets):
            if any(math.hypot(x - tg[0], y - tg[1]) < self.R + 8 for x, y in pts):
                self.combo += 1; self.hits += 1; self.score += 100 + 25 * min(self.combo - 1, 8)
                self.app.parts.burst(tg[0], tg[1], (255, 220, 80), 34)
                self.targets.remove(tg); self._spawn()
        if self.t <= 0:
            self.t = 0; self.over = True

    def render(self, s):
        for x, y in self.targets:
            r = int(self.R + 4 * math.sin(self.pulse * 4))
            for rr, a in ((r + 14, 40), (r + 6, 90)):
                g = pygame.Surface((rr * 2, rr * 2), pygame.SRCALPHA); pygame.draw.circle(g, (255, 210, 70, a), (rr, rr), rr)
                s.blit(g, (x - rr, y - rr))
            pygame.draw.circle(s, (255, 210, 70), (int(x), int(y)), r, 4)
            pygame.draw.circle(s, (255, 90, 90), (int(x), int(y)), max(4, r // 3))
        self.lines = ["TIME %4.1f s    SCORE %d    HITS %d" % (self.t, self.score, self.hits)] if not self.over else \
                     ["TIME UP!  Score %d  (%d hits)" % (self.score, self.hits), "NEXT = play again"]

    def next(self):
        self.__init__(self.app)


class DrawChallenge(GameBase):
    kind, title = "draw", "DRAW CHALLENGE"
    help = "Draw the named shape in ONE stroke.  Scored on accuracy, speed, smoothness, completion."
    ROUNDS = 5

    def __init__(self, app):
        super().__init__(app)
        order = list(CHALLENGES); random.shuffle(order)
        self.order = order[:self.ROUNDS]; self.i = 0; self.res = None; self.hold = 0.0; self.pts: List[int] = []

    @property
    def target(self): return self.order[self.i][0]

    def on_stroke(self, entry, dur):
        if self.over or self.res is not None: return
        r = score_challenge(self.target, entry[0], dur)
        w = 0.5 * r["accuracy"] + 0.2 * r["speed"] + 0.15 * r["smoothness"] + 0.15 * r["completion"] if r["ok"] else 0.0
        r["points"] = int(round(100 * w)); self.res = r; self.hold = 4.0
        self.pts.append(r["points"]); self.score = sum(self.pts)

    def _wipe(self):
        a = self.app
        a.canvas.fill((0, 0, 0, 0)); a.pending.clear(); a.undo.clear(); a.action_log.clear(); a._ai_undo.clear(); a.suggest = None; a.chip.hide()

    def update(self, dt):
        if self.res is not None and not self.over:
            self.hold -= dt
            if self.hold <= 0: self.next()

    def next(self):
        if self.over:
            self.__init__(self.app); self._wipe(); return
        if self.res is None: return
        self.res = None; self.i += 1; self._wipe()
        if self.i >= self.ROUNDS: self.over = True

    def render(self, s):
        if self.over:
            self.lines = ["DONE  total %d / %d" % (self.score, 100 * self.ROUNDS), "Rounds: " + "  ".join(str(p) for p in self.pts), "NEXT = play again"]
        elif self.res is None:
            self.lines = ["ROUND %d/%d   DRAW A  %s" % (self.i + 1, self.ROUNDS, self.order[self.i][1]), "Score so far: %d" % self.score]
        else:
            r = self.res
            self.lines = ["%s: %d points   (read as: %s)" % (self.order[self.i][1], r["points"], r["recognised"])] + r["lines"] + \
                         ["points = 50% accuracy + 20% speed + 15% smoothness + 15% completion"]


class BridgeChallenge(GameBase):
    kind, title = "bridge", "BRIDGE BUILDER"
    help = "Draw a bridge across the gap, press SPACE: it locks in place and the ball rolls to the flag."
    GAPS = [340, 430, 520]
    PLAT_Y, BALL_R, CLIFF = 470, 16, 330

    def __init__(self, app, level: int = 0, attempts: int = 1):
        super().__init__(app)
        self.level, self.attempts = level, attempts
        self.state = "build"; self.t_run = 0.0; self.ink = 0.0; self.won = False; self.msg = ""
        sp = app.phys.space
        gap = self.GAPS[min(level, len(self.GAPS) - 1)]
        self.x_l, self.x_r = self.CLIFF, self.CLIFF + gap
        sb = sp.static_body
        self.shapes: List[Any] = []
        for a, b in (((0, self.PLAT_Y), (self.x_l, self.PLAT_Y)), ((self.x_r, self.PLAT_Y), (WIN_W, self.PLAT_Y))):
            sh = pymunk.Segment(sb, a, b, 10); sh.friction = 0.8; sh.elasticity = 0.0; sp.add(sh); self.shapes.append(sh)
        self.ball = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, self.BALL_R))
        self.ball.position = (self.x_l - 120, self.PLAT_Y - 10 - self.BALL_R)
        bs = pymunk.Circle(self.ball, self.BALL_R); bs.friction = 0.5; bs.elasticity = 0.0
        sp.add(self.ball, bs); self.shapes.append(bs)
        self.flag_x = self.x_r + 110
        app.phys.space.gravity = GRAVITY_PRESETS[DEFAULT_GRAVITY]; app.phys._grav_idx = DEFAULT_GRAVITY

    def teardown(self):
        sp = self.app.phys.space
        try:
            for sh in self.shapes:
                if sh in sp.shapes: sp.remove(sh)
            if self.ball in sp.bodies: sp.remove(self.ball)
        except Exception:
            pass
        self.shapes = []

    def on_launch(self):
        a = self.app
        if self.state != "build": return
        if not a.phys.bodies:
            a.notify.show("Draw a bridge first, then launch"); return
        self.ink = 0.0
        for pb in a.phys.bodies:
            a.phys.set_frozen(pb, True)
            self.ink += sum(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(self._pts(pb), self._pts(pb)[1:]))
        self.ball.velocity = (300.0, 0.0)
        self.state = "run"; self.t_run = 0.0

    def _pts(self, pb): return self.app._body_points(pb)

    def update(self, dt):
        if self.state != "run": return
        if not self.app.phys.bodies: self.app.phys.space.step(dt)      # (phys.step handles it when bodies exist)
        self.t_run += dt
        x, y = self.ball.position
        if x >= self.flag_x and y < self.PLAT_Y + 20:
            self.state = "over"; self.won = True
            self.score = int(max(100, 1000 - 100 * (self.attempts - 1) - self.ink / 10.0))
            self.app.parts.burst(self.flag_x, self.PLAT_Y - 40, (120, 255, 160), 60)
            self.msg = "BRIDGE HELD!  %d pts  (1000 - 100 per retry - bridge length/10)" % self.score
        elif y > WIN_H + 30 or not (math.isfinite(x) and math.isfinite(y)):
            self.state = "over"; self.msg = "The ball fell.  NEXT = retry"
        elif self.t_run > 14.0:
            self.state = "over"; self.msg = "The ball stopped.  NEXT = retry"

    def next(self):
        a = self.app
        lvl, att = (self.level + 1, 1) if self.won and self.level + 1 < len(self.GAPS) else ((0, 1) if self.won else (self.level, self.attempts + 1))
        sc = self.score if self.won else 0
        self.teardown()
        a.phys.clear(); a.canvas.fill((0, 0, 0, 0)); a.pending.clear(); a.undo.clear(); a.action_log.clear(); a._ai_undo.clear()
        self.__init__(a, lvl, att)
        self.score = sc

    def render(self, s):
        for sh in self.shapes[:2]:
            pygame.draw.line(s, (120, 200, 255), sh.a, sh.b, 20)
        px = self.flag_x
        pygame.draw.line(s, (240, 240, 240), (px, self.PLAT_Y - 10), (px, self.PLAT_Y - 80), 4)
        pygame.draw.polygon(s, (120, 255, 160), [(px, self.PLAT_Y - 80), (px + 44, self.PLAT_Y - 66), (px, self.PLAT_Y - 52)])
        bx, by = self.ball.position
        if math.isfinite(bx) and math.isfinite(by):
            pygame.draw.circle(s, (255, 180, 60), (int(bx), int(by)), self.BALL_R)
            pygame.draw.circle(s, (255, 240, 200), (int(bx) - 4, int(by) - 4), 5)
        if self.state == "over": self.lines = [self.msg]
        else: self.lines = ["LEVEL %d   attempt %d   gap %d px" % (self.level + 1, self.attempts, self.x_r - self.x_l),
                            "Draw the bridge, then SPACE" if self.state == "build" else "Rolling...  %.1f s" % self.t_run]


GAME_CLASSES = {"target": TargetGame, "draw": DrawChallenge, "bridge": BridgeChallenge}


class GameManager:
    def __init__(self, app):
        self.app = app; self.g: Optional[GameBase] = None; self.backup = None; self.scores = _load_scores()
        self._f = pygame.font.SysFont("monospace", 13, bold=True); self._fb = pygame.font.SysFont("monospace", 20, bold=True)

    @property
    def active(self) -> bool: return self.g is not None

    def start(self, kind: str):
        a = self.app
        if kind not in GAME_CLASSES: return
        if self.g is not None: self._end_current(save=True)
        else:
            self.backup = a._snapshot()
        if self.backup is not None and self.g is None and (a.pending or a.phys.bodies or a.canvas.get_bounding_rect().w):
            a._clear()
        if a.mode_name != "GAME": a.set_mode("GAME", announce=False)
        a.play_panel.open = False
        self.g = GAME_CLASSES[kind](a)
        a.notify.show("%s: %s" % (self.g.title, self.g.help[:70]), (255, 220, 120))

    def _end_current(self, save: bool):
        g = self.g
        if g is None: return
        if save and g.score > self.scores.get(g.kind, 0) and (g.over or g.kind == "bridge"):
            self.scores[g.kind] = g.score; _save_scores(self.scores)
        g.teardown(); self.g = None

    def stop(self, restore: bool = True):
        if self.g is None: return
        self._end_current(save=True)
        if restore and self.backup is not None:
            try: self.app._restore(self.backup)
            except Exception: pass
            self.app.notify.show("Game ended - your drawing is back", (150, 220, 255))
        self.backup = None

    def next(self):
        if self.g is not None:
            self.g.next()

    def update(self, dt):
        if self.g is None: return
        self.g.update(dt)
        if self.g.over and self.g.score > self.scores.get(self.g.kind, 0) and self.g.kind != "bridge":
            self.scores[self.g.kind] = self.g.score; _save_scores(self.scores)
        if self.g.kind == "bridge" and self.g.over and self.g.won and self.g.score > self.scores.get("bridge", 0):
            self.scores["bridge"] = self.g.score; _save_scores(self.scores)

    def on_stroke(self, entry, dur):
        if self.g is not None: self.g.on_stroke(entry, dur)

    def on_launch(self):
        if self.g is not None: self.g.on_launch()

    def render(self, s):
        g = self.g
        if g is None: return
        g.render(s)
        lines = [g.title] + g.lines
        w = 740; h = 12 + 26 + 19 * (len(lines) - 1) + 8
        x = WIN_W // 2 - w // 2; y = 104
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 14, 28, 205), card.get_rect(), border_radius=12)
        pygame.draw.rect(card, (255, 210, 90, 230), card.get_rect(), 2, border_radius=12)
        s.blit(card, (x, y))
        s.blit(self._fb.render(g.title, True, (255, 210, 90)), (x + 14, y + 8))
        for i, ln in enumerate(g.lines):
            s.blit(self._f.render(ln[:90], True, (235, 240, 255)), (x + 14, y + 38 + 19 * i))


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 9 (part C) — ART / PLAY / FILE panels and the ArtMixin
# ══════════════════════════════════════════════════════════════════════════════
def _art_stepper(label: str, key: str, step, w: int = 330) -> RowSpec:
    return RowSpec(cells=[Cell(label, w=w - 12 - 2 * 38), Cell("-", key, -step, w=34, repeat=True), Cell("+", key, step, w=34, repeat=True)], h=28)


class ArtPanel(FlyoutPanel):
    def __init__(self):
        super().__init__("ART", pygame.Rect(232, 64, 100, 32), 420, (255, 140, 60))

    def specs(self, app) -> List[RowSpec]:
        rows = [RowSpec(header="DRAWING"),
                RowSpec(cells=[Cell("TWO-HAND  [W]", "twohand", None, on=app.two_hand), Cell("both index fingers draw", w=200)], h=30),
                RowSpec(header="SYMMETRY  [M]")]
        lab = {"off": "OFF", "horizontal": "HORIZ", "vertical": "VERT", "radial": "RADIAL", "kaleidoscope": "KALEID"}
        rows.append(RowSpec(cells=[Cell(lab[m], "sym", m, on=(app.sym_mode == m)) for m in SYM_MODES[:3]], h=30))
        rows.append(RowSpec(cells=[Cell(lab[m], "sym", m, on=(app.sym_mode == m)) for m in SYM_MODES[3:]], h=30))
        rows.append(_art_stepper("FOLDS: %d" % app.sym_n, "symn", 1))
        rows.append(RowSpec(cells=[Cell("CENTER: " + ("SCREEN" if app.sym_center is None else "%d,%d" % app.sym_center), w=100),
                                   Cell("SCREEN", "symc", "screen"), Cell("LAST", "symc", "last")], h=28))
        rows.append(RowSpec(header="GENERATIVE ART  [J]"))
        rows.append(RowSpec(cells=[Cell("AUTO: " + ("ON" if app.gen_on else "OFF") + " [J]", "gen", None, on=app.gen_on),
                                   Cell("GENERATE", "generate", None)], h=30))
        rows.append(_art_stepper("STRENGTH: %d" % app.gen_strength, "gstr", 1))
        rows.append(_art_stepper("PETALS/FOLDS: %d" % app.gen_folds, "gfold", 1))
        rows.append(_art_stepper("ROTATION: %d deg" % app.gen_rot, "grot", 15))
        rows.append(_art_stepper("PARTICLES: %d" % app.gen_density, "gdens", 1))
        rows.append(RowSpec(cells=[Cell("line=tree circle=mandala tri=sierpinski", w=306)], h=22))
        return rows


class PlayPanel(FlyoutPanel):
    def __init__(self):
        super().__init__("PLAY", pygame.Rect(126, 64, 100, 32), 420, (255, 210, 90))

    def specs(self, app) -> List[RowSpec]:
        g = app.game
        rows = [RowSpec(header="GAME MODE  [Ctrl+G]")]
        rows.append(RowSpec(cells=[Cell("TARGETS", "start", "target", w=98, on=(g.g is not None and g.g.kind == "target")),
                                   Cell("DRAW IT", "start", "draw", w=98, on=(g.g is not None and g.g.kind == "draw")),
                                   Cell("BRIDGE", "start", "bridge", w=98, on=(g.g is not None and g.g.kind == "bridge"))], h=30))
        rows.append(RowSpec(cells=[Cell("NEXT / RETRY", "next", None, w=150, enabled=g.active), Cell("STOP", "stop", None, w=90, enabled=g.active)], h=30))
        rows.append(RowSpec(cells=[Cell((g.g.help if g.g else "Pick a game. Your drawing is saved and restored.")[:46], w=306)], h=22))
        rows.append(RowSpec(cells=[Cell("BEST  targets %d  draw %d  bridge %d" % tuple(g.scores.get(k, 0) for k in ("target", "draw", "bridge")), w=306)], h=22))
        return rows


class FilePanel(FlyoutPanel):
    def __init__(self):
        super().__init__("FILE", pygame.Rect(20, 64, 100, 32), 420, (120, 230, 150))

    def specs(self, app) -> List[RowSpec]:
        rows = [RowSpec(header="EXPORT  (folder: %s/)" % EXPORT_DIR)]
        rows.append(RowSpec(cells=[Cell(t, "export", k) for t, k in (("PNG", "png"), ("JPG", "jpg"), ("SVG", "svg"))], h=30))
        rows.append(RowSpec(cells=[Cell("JSON", "export", "json"), Cell("BG: " + app.exp_bg.upper(), "expbg", None)], h=30))
        rows.append(RowSpec(header="PROJECT  (folder: %s/)" % PROJECT_DIR))
        rows.append(RowSpec(cells=[Cell("SAVE  Ctrl+S", "save", None), Cell("LOAD LATEST", "load", None)], h=30))
        for p in app._recent_projects():
            rows.append(RowSpec(cells=[Cell(os.path.basename(p)[:30], "load", p, w=306)], h=26))
        rows.append(RowSpec(header="RECORD"))
        rows.append(RowSpec(cells=[Cell("STOP REC" if app.rec.active and app.rec.kind == "mp4" else "MP4  [R]", "rec", "mp4", w=100, on=(app.rec.active and app.rec.kind == "mp4")),
                                   Cell("STOP GIF" if app.rec.active and app.rec.kind == "gif" else "GIF", "rec", "gif", w=80, on=(app.rec.active and app.rec.kind == "gif")),
                                   Cell("AUTOSAVE: " + ("ON" if app.autosave_on else "OFF"), "auto", None, w=112, on=app.autosave_on)], h=30))
        rows.append(RowSpec(cells=[Cell(app.file_msg[:44], w=306)], h=22))
        return rows


class ArtMixin:
    """Phase 9 behaviour of App: symmetry, generative art, game modes, export, recording, project files."""

    def _art_init(self):
        self.sym_mode, self.sym_n, self.sym_center = "off", 8, None
        self.gen_on, self.gen_strength, self.gen_folds, self.gen_rot, self.gen_density = False, 3, 8, 0, 2
        self._sym_live = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA); self._sym_used = False
        self._stroke_t0 = time.perf_counter()
        self.art_panel, self.play_panel, self.file_panel = ArtPanel(), PlayPanel(), FilePanel()
        self._ext += [(self.art_panel, self.art_apply), (self.play_panel, self.play_apply), (self.file_panel, self.file_apply)]
        self.exp_bg = "dark"; self.file_msg = "Ready"
        self._file_q: "queue.Queue" = queue.Queue()
        self.rec = Recorder(lambda p, ok: self._file_q.put(("Saved: " + os.path.basename(p) if ok else "Recording failed", ok)))
        self.game = GameManager(self)
        self.autosave_on = False; self._auto_t = 0.0; self._auto_sig = None
        self._last_frame = None; self._recent_cache = (0.0, [])
        self._rec_font = pygame.font.SysFont("monospace", 16, bold=True)

    # ---------------------------------------------------------------- symmetry
    def _sym_c(self): return self.sym_center if self.sym_center is not None else SYM_CENTER

    def _sym_live_seg(self, p1, p2):
        col = (*self.rgb, 235); w = max(2, self.pen_w)
        for cp in symmetry_copies([p1, p2], self.sym_mode, self.sym_n, self._sym_c()):
            pygame.draw.line(self._sym_live, col, cp[0], cp[1], w)
            pygame.draw.circle(self._sym_live, col, cp[1], w // 2)
        self._sym_used = True

    def _art_live_render(self, s):
        if self._sym_used: s.blit(self._sym_live, (0, 0))
        if self.sym_mode != "off" and not self.mode3d:                # faint guide lines (drawn directly: cheap)
            cx, cy = self._sym_c(); gc = (70, 100, 140)
            if self.sym_mode == "horizontal": pygame.draw.line(s, gc, (0, cy), (WIN_W, cy), 1)
            elif self.sym_mode == "vertical": pygame.draw.line(s, gc, (cx, 0), (cx, WIN_H), 1)
            else:
                n = self.sym_n * (2 if self.sym_mode == "kaleidoscope" else 1)
                for k in range(n):
                    a = math.tau * k / n
                    pygame.draw.line(s, gc, (cx, cy), (cx + 900 * math.cos(a), cy + 900 * math.sin(a)), 1)
        self.game.render(s)
        if self.rec.active:
            if int(time.perf_counter() * 2) % 2 == 0: pygame.draw.circle(s, (255, 40, 40), (WIN_W // 2 - 40, 26), 8)
            s.blit(self._rec_font.render("REC %s %02d:%02d" % (self.rec.kind.upper(), self.rec.elapsed() // 60, self.rec.elapsed() % 60), True, (255, 120, 120)), (WIN_W // 2 - 26, 17))

    # ---------------------------------------------------------------- commit hooks
    def _art_pre(self):
        if self.mode3d or not (self.sym_mode != "off" or self.gen_on): return None
        return (self._snapshot(), list(self.undo._s), list(self._redo))

    def _ink_poly(self, P, rgb, w, bk, deadline: float = 1e18) -> int:
        runs, cur = [], []
        for x, y in P:
            if -10 <= x < WIN_W + 10 and -10 <= y < WIN_H + 10: cur.append((int(x), int(y)))
            else:
                if len(cur) >= 2: runs.append(cur)
                cur = []
        if len(cur) >= 2: runs.append(cur)
        n = 0
        br = BRUSHES.get(bk) or BRUSHES[DEFAULT_BRUSH]
        for R in runs:
            if time.perf_counter() > deadline: break
            ctx = br.new_ctx(rgb, w, R[0], seed=random.getrandbits(32))
            for a, b in zip(R, R[1:]):
                for fp in self._densify(a, b): self.layers.draw(br.feed(ctx, fp, ctx.w), ctx.passes)
            self.layers.flatten_into(self.canvas, ctx.passes); self.layers.clear()
            self.pending.append((R, tuple(rgb), w, bk)); n += 1
        return n

    def _gen_polys(self, pts):
        if len(pts) < 8 or sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pts, pts[1:])) < 60: return []
        kind, outline = "unknown", None
        try:
            r = ShapeRecognizer.recognize(list(pts))
            if r is not None: kind, outline = r.kind, r.outline
        except Exception:
            pass
        return generate_pattern(pts, kind, self.gen_strength, self.gen_folds, float(self.gen_rot), outline)

    def _gen_ink(self, entry) -> int:
        dl = time.perf_counter() + 0.6                                   # never freeze a frame for long
        polys = self._gen_polys(entry[0]); n = 0
        for P, h in polys:
            n += self._ink_poly(P, hue_shift(entry[1], h), max(2, entry[2] - 1), entry[3], dl)
        if n and self.gen_density > 0 and polys:
            for _ in range(min(12, 3 * self.gen_density)):
                P, h = random.choice(polys); x, y = random.choice(P)
                self.parts.burst(x, y, hue_shift(entry[1], h), 5 * self.gen_density)
        return n

    def _art_after_commit(self, entry, pre):
        self.game.on_stroke(entry, time.perf_counter() - self._stroke_t0)
        if pre is None: return
        n = 0
        dl = time.perf_counter() + 0.8
        if self.sym_mode != "off":
            for P in symmetry_copies(entry[0], self.sym_mode, self.sym_n, self._sym_c()):
                n += self._ink_poly(P, entry[1], entry[2], entry[3], dl)
        if self.gen_on: n += self._gen_ink(entry)
        if n:
            self._ai_undo.append(pre)
            if len(self._ai_undo) > 20: del self._ai_undo[0]
            self.action_log.append("ai")

    def art_apply(self, key: str, arg=None):
        if key == "twohand": self.toggle_two_hand()
        elif key == "sym": self.set_symmetry(arg)
        elif key == "symn": self.sym_n = max(2, min(16, self.sym_n + arg))
        elif key == "symc":
            if arg == "screen": self.sym_center = None
            elif self.pending:
                xs = [p[0] for e in self.pending[-1:] for p in e[0]]; ys = [p[1] for e in self.pending[-1:] for p in e[0]]
                self.sym_center = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
            else: self.notify.show("Draw something first")
        elif key == "gen": self.toggle_gen()
        elif key == "gstr": self.gen_strength = max(1, min(5, self.gen_strength + arg))
        elif key == "gfold": self.gen_folds = max(2, min(16, self.gen_folds + arg))
        elif key == "grot": self.gen_rot = (self.gen_rot + arg) % 360
        elif key == "gdens": self.gen_density = max(0, min(5, self.gen_density + arg))
        elif key == "generate": self.generate_from_last()

    def set_symmetry(self, mode):
        if mode not in SYM_MODES: return
        self.sym_mode = mode
        self.notify.show("Symmetry: %s%s" % (mode.upper(), "" if mode in ("off", "horizontal", "vertical") else " x%d" % self.sym_n), (255, 160, 80))

    def cycle_symmetry(self): self.set_symmetry(SYM_MODES[(SYM_MODES.index(self.sym_mode) + 1) % len(SYM_MODES)])

    def toggle_gen(self):
        self.gen_on = not self.gen_on
        self.notify.show("Generative art " + ("ON - each stroke grows a pattern" if self.gen_on else "OFF"), (255, 160, 80))

    def generate_from_last(self) -> int:
        if not self.pending: self.notify.show("Draw a stroke first"); return 0
        entry = self.pending[-1]
        if not self._gen_polys(entry[0]): self.notify.show("That stroke is too short for a pattern"); return 0
        self._ai_push_undo()
        n = self._gen_ink(entry)
        if not n: self._ai_undo.pop(); return 0
        self._ai_done("Generative art")
        self.notify.show("Generated %d lines" % n, (255, 160, 80))
        return n

    # ---------------------------------------------------------------- games
    def play_apply(self, key: str, arg=None):
        if key == "start": self.game.start(arg)
        elif key == "next": self.game.next()
        elif key == "stop": self.game.stop()

    def _game_enter(self):
        if not self.game.active: self.play_panel.open = True
        self.notify.show("GAME MODE: pick a game in the PLAY tab", (255, 210, 120))

    # ---------------------------------------------------------------- per frame
    def _art_update(self, dt: float):
        while True:
            try: msg, ok = self._file_q.get_nowait()
            except queue.Empty: break
            self.file_msg = msg
            self.notify.show(msg, (120, 230, 150) if ok else (255, 120, 120))
        if self.game.active and self.mode_name != "GAME": self.game.stop()
        self.game.update(dt)
        HUD_STATE["hide"] = any(p.open for p in (self.brush_panel, self.fx_panel, self.edit_panel, self.layer_panel, self.time_panel)) or any(p.open for p, _ in self._ext)
        if self._sym_used and self.stroke_ctx is None:
            self._sym_live.fill((0, 0, 0, 0)); self._sym_used = False
        if self.autosave_on:
            self._auto_t += dt
            if self._auto_t >= AUTOSAVE_S:
                self._auto_t = 0.0
                last = self.hist.entries[-1] if self.hist.entries else None
                sig = (last.id if last is not None and hasattr(last, "id") else id(last), len(self.pending), len(self.phys.bodies))
                if sig != self._auto_sig and not self.cur_stroke:
                    self._auto_sig = sig; self.save_project(auto=True)

    def _rec_capture(self):
        if self.rec.active: self.rec.capture(self.screen)

    def _art_shutdown(self):
        try:
            if self.rec.active: self.rec.stop()
            if self.rec.last_thread is not None: self.rec.last_thread.join(timeout=20)
        except Exception:
            pass

    # ---------------------------------------------------------------- export
    def _stamp(self): return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _unique(self, d: str, stem: str, ext: str) -> str:
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, stem + ext); i = 2
        while os.path.exists(p): p = os.path.join(d, "%s_%d%s" % (stem, i, ext)); i += 1
        return p

    def compose(self, bg: str) -> pygame.Surface:
        """The artwork on its own (no toolbar / panels / HUD), rendered into a temporary surface."""
        saved = self.screen
        surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        try:
            self.screen = surf
            if bg == "theme":
                fr = self._last_frame if self._last_frame is not None else np.zeros((CAM_H, CAM_W, 3), np.uint8)
                self._draw_background(fr)
            elif bg == "dark": surf.fill((14, 16, 24, 255))
            for lid in self.lyr.order:
                if not self.lyr.visible(lid): continue
                if lid == "drawing": self._draw_drawing_layer()
                elif lid == "objects": self._draw_objects_layer()
                else: self._draw_effects_layer()
        finally:
            self.screen = saved
        return surf

    def export_image(self, ext: str) -> str:
        bg = "dark" if (ext == "jpg" and self.exp_bg == "clear") else self.exp_bg
        surf = self.compose(bg)
        rgb = np.ascontiguousarray(pygame.surfarray.array3d(surf).transpose(1, 0, 2))
        if ext == "png" and bg == "clear":
            a = np.ascontiguousarray(pygame.surfarray.array_alpha(surf).T)
            img = np.dstack([rgb[..., 2], rgb[..., 1], rgb[..., 0], a])
        else:
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        path = self._unique(EXPORT_DIR, "airdraw_" + self._stamp(), "." + ext)
        ok = cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == "jpg" else [])
        if not ok: raise IOError("could not write " + path)
        return path

    def _vector_items(self) -> List[dict]:
        items = [dict(pts=list(e[0]), rgb=e[1], width=e[2], brush=e[3]) for e in self.pending]
        for pb in self.phys.bodies:
            P = self._body_points(pb)
            if len(P) >= 2: items.append(dict(pts=P, rgb=pb.color, width=pb.width, brush=pb.brush, opacity=float(pb.opacity)))
        return items

    def export_svg(self) -> str:
        path = self._unique(EXPORT_DIR, "airdraw_" + self._stamp(), ".svg")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(svg_document(self._vector_items(), WIN_W, WIN_H, None if self.exp_bg == "clear" else (14, 16, 24)))
        return path

    def export_json(self) -> str:
        snap = self._snapshot()
        doc = dict(format="airdraw-strokes", version=1,
                   strokes=[dict(points=[list(p) for p in it["pts"]], color=list(it["rgb"]), width=it["width"], brush=it["brush"]) for it in self._vector_items()],
                   settings=snap["settings"], solids=len(snap["solids"]))
        path = self._unique(EXPORT_DIR, "airdraw_" + self._stamp(), ".json")
        with open(path, "w", encoding="utf-8") as fh: json.dump(enc(doc), fh, indent=1)
        return path

    def file_apply(self, key: str, arg=None):
        try:
            if key == "export":
                p = {"png": lambda: self.export_image("png"), "jpg": lambda: self.export_image("jpg"),
                     "svg": self.export_svg, "json": self.export_json}[arg]()
                self._file_q.put(("Exported: " + os.path.basename(p), True))
            elif key == "expbg": self.exp_bg = {"dark": "theme", "theme": "clear", "clear": "dark"}[self.exp_bg]
            elif key == "save": self.save_project()
            elif key == "load":
                p = arg or (self._recent_projects(1) or [None])[0]
                if p is None: self._file_q.put(("No saved projects yet", False))
                else: self.load_project(p)
            elif key == "rec": self.toggle_record(arg)
            elif key == "auto":
                self.autosave_on = not self.autosave_on; self._auto_t = 0.0
                self.notify.show("Autosave %s (every %d s, newest 10 kept)" % ("ON" if self.autosave_on else "OFF", AUTOSAVE_S))
        except Exception as e:
            self._file_q.put(("%s failed: %s" % (key, e), False))

    def toggle_record(self, kind: str = "mp4"):
        if self.rec.active: self.rec.stop(); self.notify.show("Recording stopped - saving..."); return
        ext = ".mp4" if kind == "mp4" else ".gif"
        path = self._unique(EXPORT_DIR, "airdraw_rec_" + self._stamp(), ext)
        if self.rec.start(kind, path): self.notify.show("Recording %s (max %d s) - press R to stop" % (kind.upper(), REC_MAX_S), (255, 110, 110))
        else: self._file_q.put(("Could not start recording", False))

    # ---------------------------------------------------------------- projects
    def _recent_projects(self, n: int = 3) -> List[str]:
        t, cache = self._recent_cache
        if time.perf_counter() - t > 2.0:
            try:
                fs = [os.path.join(PROJECT_DIR, f) for f in os.listdir(PROJECT_DIR) if f.endswith(PROJECT_EXT)]
                cache = sorted(fs, key=os.path.getmtime, reverse=True)
            except OSError:
                cache = []
            self._recent_cache = (time.perf_counter(), cache)
        return cache[:n]

    def save_project(self, auto: bool = False):
        snap = self._snapshot()
        meta = dict(theme=self.theme, hud=self.hud_mode, mode=self.mode_name, sym=[self.sym_mode, self.sym_n],
                    gen=[self.gen_on, self.gen_strength, self.gen_folds, self.gen_rot, self.gen_density])
        prev = None
        if not auto:
            try:
                th = cv2.resize(np.ascontiguousarray(pygame.surfarray.array3d(self.compose("dark")).transpose(1, 0, 2)), (320, 180), interpolation=cv2.INTER_AREA)
                prev = cv2.imencode(".png", cv2.cvtColor(th, cv2.COLOR_RGB2BGR))[1].tobytes()
            except Exception:
                prev = None
        path = self._unique(PROJECT_DIR, ("autosave_" if auto else "airdraw_") + self._stamp(), PROJECT_EXT)
        def work():
            try:
                write_airdraw(path, snap, prev, meta)
                if auto: self._prune_autosaves()
                self._file_q.put((("Autosaved: " if auto else "Project saved: ") + os.path.basename(path), True))
            except Exception as e:
                self._file_q.put(("Save failed: %s" % e, False))
        threading.Thread(target=work, daemon=True).start()
        self._recent_cache = (0.0, [])
        return path

    def _prune_autosaves(self, keep: int = 10):
        try:
            fs = sorted((os.path.join(PROJECT_DIR, f) for f in os.listdir(PROJECT_DIR) if f.startswith("autosave_") and f.endswith(PROJECT_EXT)),
                        key=os.path.getmtime, reverse=True)
            for f in fs[keep:]: os.remove(f)
        except OSError:
            pass

    def load_project(self, path: str) -> bool:
        try:
            snap, meta = read_airdraw(path, with_meta=True)
        except ValueError as e:
            self._file_q.put(("Cannot open: %s" % e, False)); return False
        self.game.stop(restore=False)
        pre = self._snapshot()
        try:
            self._restore(snap)
        except Exception as e:
            self._restore(pre)
            self._file_q.put(("Project damaged, nothing changed (%s)" % e.__class__.__name__, False)); return False
        self._ai_undo.clear()
        try:
            if meta.get("theme") in THEMES: self.theme = meta["theme"]
            if meta.get("hud") in HUD_MODES: self.hud_mode = meta["hud"]
            sm = meta.get("sym") or ["off", 8]
            if sm[0] in SYM_MODES: self.sym_mode, self.sym_n = sm[0], max(2, min(16, int(sm[1])))
        except Exception:
            pass
        self._log("Open project")
        self._file_q.put(("Opened: " + os.path.basename(path), True))
        return True




# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 10 — one clean UI skin · steady hover · smooth input · help overlay
#  All panels share the same grid: tabs 100x32, buttons 126x34 (steppers 40x34), toolbar 70x46.
# ══════════════════════════════════════════════════════════════════════════════
# ── PANEL SIZE: one knob.  1.0 = compact, 1.2 = default, 1.4 = large.  (or set env AIRDRAW_UI_SCALE=1.4)
try: UI_SCALE = max(0.8, min(1.6, float(os.environ.get("AIRDRAW_UI_SCALE", "1.2"))))
except ValueError: UI_SCALE = 1.2
UI_BTN_W, UI_BTN_H, UI_GAP, UI_PAD = int(126 * UI_SCALE), int(34 * UI_SCALE), 6, 12
UI_SMALL = ("-", "+", "<", ">", "UP", "DN", "CLR")
UI_TIP_TAU = 0.040                # fingertip low-pass time constant for UI pointing (s)
UI_LOST_GRACE = 0.18              # a lost / jittering fingertip keeps its hover this long (s)
UI_STICKY_PX = 12                 # a button stays hovered this far outside its edge
_SPRITES: Dict[tuple, pygame.Surface] = {}
_TEXTS: Dict[tuple, pygame.Surface] = {}


def _cached(cache: dict, key, build, limit: int = 700):
    v = cache.get(key)
    if v is None:
        if len(cache) > limit: cache.clear()
        v = cache[key] = build()
    return v


def ui_text(font, text: str, color) -> pygame.Surface:
    return _cached(_TEXTS, (id(font), text, tuple(color)), lambda: font.render(text, True, color), 900)


def ui_button(w: int, h: int, accent, state: str, prog: float, radius: int = 8) -> pygame.Surface:
    """state: off | on | dim | swatch.  prog 0..1 = dwell progress (drawn as a bar along the bottom)."""
    q = int(max(0.0, min(1.0, prog)) * 14 + 0.5)
    def build():
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        a = tuple(accent)
        if state == "dim":
            fill, edge, ew = (20, 24, 38, 170), (60, 66, 92, 120), 1
        elif state == "on":
            fill, edge, ew = (int(a[0] * .42 + 14), int(a[1] * .42 + 14), int(a[2] * .42 + 14), 245), (255, 255, 255, 235), 2
        else:
            fill, edge, ew = (28 + q, 34 + q, 54 + q, 238), (*(int(c * .75) for c in a), 170), 1
        pygame.draw.rect(s, fill, s.get_rect(), border_radius=radius)
        if q and state != "dim":
            pygame.draw.rect(s, (*a, 70 + 4 * q), s.get_rect(), border_radius=radius)
            bw = max(0, int((w - 8) * q / 14))
            if bw: pygame.draw.rect(s, (*a, 255), (4, h - 5, bw, 3), border_radius=2)
        pygame.draw.rect(s, edge, s.get_rect(), ew, border_radius=radius)
        return s
    return _cached(_SPRITES, ("btn", w, h, tuple(accent), state, q, radius), build)


def ui_panel_bg(w: int, h: int, accent, pulse: int) -> pygame.Surface:
    def build():
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(s, (9, 13, 24, 238), s.get_rect(), border_radius=14)
        pygame.draw.rect(s, (*accent, pulse), s.get_rect(), 2, border_radius=14)
        pygame.draw.rect(s, (*accent, 190), (14, 0, w - 28, 3), border_radius=2)
        return s
    return _cached(_SPRITES, ("pnl", w, h, tuple(accent), pulse // 8), build, 200)


def ui_tab(w: int, h: int, accent, open_: bool, prog: float) -> pygame.Surface:
    q = int(max(0.0, min(1.0, prog)) * 14 + 0.5)
    def build():
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(s, ((*(int(c * .5 + 10) for c in accent), 245) if open_ else (9, 13, 24, 225)), s.get_rect(), border_radius=10)
        if q: pygame.draw.rect(s, (*accent, 40 + 6 * q), s.get_rect(), border_radius=10)
        pygame.draw.rect(s, (*accent, 255 if open_ else 200), s.get_rect(), 2, border_radius=10)
        if q: pygame.draw.rect(s, (*accent, 255), (6, h - 5, int((w - 12) * q / 14), 3), border_radius=2)
        return s
    return _cached(_SPRITES, ("tab", w, h, tuple(accent), open_, q), build)


# ─────────────────────────────── FlyoutPanel: grid layout, steady hover, clean render ───────────────────────────────
def _cell_kind(c: Cell) -> str:
    if not c.key: return "label"
    if c.rgb: return "sw"
    if c.text in UI_SMALL: return "small"
    return "btn"


def _fp_row_h(r: RowSpec, compact: bool) -> int:
    if r.header: return int(28 * UI_SCALE)
    kinds = [_cell_kind(c) for c in r.cells]
    if all(k == "label" for k in kinds): return int(24 * UI_SCALE)
    return (int(30 * min(UI_SCALE, 1.15)) + 2) if compact else (UI_BTN_H + 4)


def _fp_layout(self, ctx):
    if getattr(self, "_ui_sc", None) != UI_SCALE:                       # first layout: widen the panel and grow its fonts once
        self._ui_sc = UI_SCALE
        self.width = min(int(self.width * UI_SCALE), WIN_W - 12)
        self._f = pygame.font.SysFont("monospace", int(14 * UI_SCALE), bold=True)
        self._s = pygame.font.SysFont("monospace", int(11 * UI_SCALE))
    rows = self.specs(ctx)
    avail = WIN_H - (self.TAB.bottom + 4) - 86                          # keep clear of the toolbar
    tall = sum(_fp_row_h(r, False) for r in rows) + 2 * UI_PAD
    compact = tall > avail
    bh = int(30 * min(UI_SCALE, 1.15)) if compact else UI_BTN_H
    h = sum(_fp_row_h(r, compact) for r in rows) + 2 * UI_PAD
    x = min(self.TAB.x, WIN_W - self.width - 6)
    self.panel_rect = pygame.Rect(x, self.TAB.bottom + 4, self.width, h)
    inner = self.width - 2 * UI_PAD
    y = self.panel_rect.y + UI_PAD
    self.hits, self._draw = [], []
    for r in rows:
        rh = _fp_row_h(r, compact)
        rr = pygame.Rect(x + UI_PAD, y, inner, rh)
        if r.header:
            self._draw.append((rr, r, [])); y += rh; continue
        kinds = [_cell_kind(c) for c in r.cells]
        n = len(r.cells)
        w = [0] * n
        for i, (c, k) in enumerate(zip(r.cells, kinds)):
            if k == "small": w[i] = int(40 * UI_SCALE)
            elif k == "sw": w[i] = int(30 * UI_SCALE)
            elif k == "btn": w[i] = inner if (c.w or 0) >= 250 and n == 1 else UI_BTN_W         # a wide cell on its own = full-width bar
            elif i > 0 or any(kk != "label" for kk in kinds[:i]): w[i] = max(c.w or 0, 84)
        flex = [i for i in range(n) if kinds[i] == "label" and w[i] == 0]
        if not flex and kinds.count("label") and all(k == "label" for k in kinds): flex = [0]; w[0] = 0
        if not flex and "label" in kinds and n > 1:
            flex = [kinds.index("label")]; w[flex[0]] = 0
        used = sum(w) + UI_GAP * (n - 1)
        if flex:
            fw = max(60, inner - used)
            for i in flex: w[i] = fw // len(flex)
        elif used > inner:                                              # too many buttons for one row: shrink them evenly
            nb = [i for i in range(n) if kinds[i] == "btn"]
            if nb:
                spare = inner - (sum(w) - sum(w[i] for i in nb)) - UI_GAP * (n - 1)
                for i in nb: w[i] = max(54, spare // len(nb))
        placed, cx = [], rr.x
        for c, cw in zip(r.cells, w):
            rect = pygame.Rect(cx, rr.y + (rh - bh) // 2 if _cell_kind(c) != "label" else rr.y, cw, bh if _cell_kind(c) != "label" else rh)
            placed.append((rect, c)); cx += cw + UI_GAP
            if c.enabled and c.key: self.hits.append((rect, c.key, c.arg, c.repeat))
        self._draw.append((rr, r, placed)); y += rh


def _fp_update(self, tip, dt: float):
    """0.5 s finger-hold fires (steppers repeat).  Hover is sticky and survives a 0.18 s tracking drop-out."""
    hit = self._hit(tip)
    cur = None if self._hk is None else self._hk
    rect_prev = getattr(self, "_hr", None)
    if cur is not None and rect_prev is not None and tip is not None and rect_prev.inflate(2 * UI_STICKY_PX, 2 * UI_STICKY_PX).collidepoint(tip):
        hit = getattr(self, "_hit_prev", hit) or hit                    # still on (or right next to) the same button
    grace = False
    if hit is None and cur is not None and getattr(self, "_lost", 0.0) < UI_LOST_GRACE:
        self._lost = getattr(self, "_lost", 0.0) + dt
        hit = getattr(self, "_hit_prev", None); grace = hit is not None
    elif hit is not None: self._lost = 0.0
    fired = None
    hov = None if hit is None else (hit[0], repr(hit[1]))
    if hov != self._hk:
        self._hk, self._ht, self._latched, self._rep = hov, 0.0, False, 0.0
    elif hov is not None and not grace:
        self._ht += dt
        if not self._latched and self._ht >= UI_HOVER_S:
            self._latched, self._rep = True, 0.0
            if hit[0] == "_tab": self.toggle()
            else: fired = (hit[0], hit[1])
        elif self._latched and hit[2] and hit[0] != "_tab":
            self._rep += dt
            if self._rep >= 0.25:
                self._rep = 0.0; fired = (hit[0], hit[1])
    if hit is not None:
        self._hit_prev = hit
        self._hr = self.TAB if hit[0] == "_tab" else next((rc for rc, k, a, rp in self.hits if k == hit[0] and repr(a) == repr(hit[1])), None)
    else:
        self._hit_prev = None; self._hr = None
    if self.open and hov is None:
        self._idle += dt
        if self._idle > 10.0: self.open = False
    else:
        self._idle = 0.0
    return fired


def _fp_render(self, surface: pygame.Surface, ctx):
    A = self.accent
    hov_tab = self._hk is not None and self._hk[0] == "_tab"
    prog = (self._ht / UI_HOVER_S) if (hov_tab and not self._latched) else 0.0
    surface.blit(ui_tab(self.TAB.w, self.TAB.h, A, bool(self.open), prog), self.TAB.topleft)
    lb = ui_text(self._f, self.title, (245, 247, 255))
    surface.blit(lb, (self.TAB.x + (self.TAB.w - lb.get_width()) // 2, self.TAB.y + (self.TAB.h - lb.get_height()) // 2))
    if not self.open: return
    self.layout(ctx)
    surface.blit(ui_panel_bg(self.panel_rect.w, self.panel_rect.h, A, ui_pulse(210, 30)), self.panel_rect.topleft)
    for rr, r, placed in self._draw:
        if r.header:
            surface.blit(ui_text(self._f, r.header, (*A,)), (rr.x + 2, rr.y + 6))
            pygame.draw.line(surface, (*(int(c * .45) for c in A),), (rr.x, rr.bottom - 2), (rr.right, rr.bottom - 2), 1)
            continue
        for ci, (rect, c) in enumerate(placed):
            kind = _cell_kind(c)
            if kind == "label":
                first = ci == 0 or all(_cell_kind(p[1]) == "label" for p in placed[:ci])
                font = self._f if len(r.cells) > 1 or ci else self._s
                txt = c.text
                while txt and font.size(txt)[0] > rect.w - 4 and len(txt) > 4: txt = txt[:-2]
                if txt != c.text: txt = txt.rstrip() + "…"
                col = (235, 240, 255) if len(r.cells) > 1 else (160, 172, 200)
                im = ui_text(font, txt, col)
                x = rect.x + 2 if first and len(r.cells) >= 1 and (len(placed) == 1 or ci == 0) else rect.x + (rect.w - im.get_width()) // 2
                surface.blit(im, (x, rect.y + (rect.h - im.get_height()) // 2))
                continue
            hovered = self._hk == (c.key, repr(c.arg))
            prog = (self._ht / UI_HOVER_S) if (hovered and not self._latched) else 0.0
            if kind == "sw":
                pygame.draw.rect(surface, c.rgb, rect, border_radius=7)
                pygame.draw.rect(surface, (255, 255, 255) if c.on else (90, 96, 120), rect, 3 if c.on else 1, border_radius=7)
                if prog: pygame.draw.rect(surface, (255, 255, 255), (rect.x + 3, rect.bottom - 5, int((rect.w - 6) * prog), 3))
                continue
            state = "dim" if not c.enabled else ("on" if c.on else "off")
            surface.blit(ui_button(rect.w, rect.h, A, state, prog), rect.topleft)
            txt = c.text; font = self._f
            if font.size(txt)[0] > rect.w - 8: font = self._s
            while txt and font.size(txt)[0] > rect.w - 8 and len(txt) > 3: txt = txt[:-2]
            if txt != c.text: txt = txt.rstrip() + "…"
            im = ui_text(font, txt, (235, 240, 255) if c.enabled else (110, 116, 140))
            surface.blit(im, (rect.x + (rect.w - im.get_width()) // 2, rect.y + (rect.h - im.get_height()) // 2))


FlyoutPanel.layout, FlyoutPanel.update, FlyoutPanel.render = _fp_layout, _fp_update, _fp_render


# ─────────────────────────────── Toolbar: pre-drawn buttons, big forgiving hit area, steady hover ───────────────────────────────
for _b in Toolbar.BTNS:
    if _b.key == "wind_l": _b.label = "<WIND"
    elif _b.key == "wind_r": _b.label = "WIND>"
    elif _b.key == "screenshot": _b.label = "PHOTO"
Toolbar.H, Toolbar.PAD, Toolbar.BW = 46, 6, 70


def _tb_init(self):
    total = len(self.BTNS) * (self.BW + self.PAD) - self.PAD
    x0, y0 = (WIN_W - total) // 2, WIN_H - self.H - 36
    self._rects, self._hit_rects = {}, {}
    for b in self.BTNS:
        r = pygame.Rect(x0, y0, self.BW, self.H)
        self._rects[b.key] = r
        self._hit_rects[b.key] = pygame.Rect(r.x - self.PAD // 2, r.y - 8, r.w + self.PAD, WIN_H - (r.y - 8))   # reaches the very bottom edge
        x0 += self.BW + self.PAD
    self._hk, self._ht, self._lost = None, 0.0, 0.0
    self._anim = {}
    self._font = pygame.font.SysFont("monospace", 13, bold=True)
    self.mode_label, self.cur_mode = "MODE 2D", "DRAW"


def _tb_update(self, tip, dt: float):
    hov = None
    if tip:
        if self._hk is not None and self._hit_rects[self._hk].inflate(2 * UI_STICKY_PX, 0).collidepoint(tip): hov = self._hk
        else:
            for k, r in self._hit_rects.items():
                if r.collidepoint(tip): hov = k; break
    grace = False
    if hov is None and self._hk is not None and self._lost < UI_LOST_GRACE:
        self._lost += dt; hov = self._hk; grace = True                   # tracking blip: keep the hover, don't advance the timer
    elif hov is not None: self._lost = 0.0
    fired = None
    if hov is not None and hov == self._hk:
        if not grace:
            self._ht += dt
            if self._ht >= UI_HOVER_S:
                fired = hov; self._ht = 0.0
    else:
        self._hk, self._ht = hov, 0.0
    for k in list(self._anim):
        tgt = min(self._ht / UI_HOVER_S, 1.0) if k == self._hk else 0.0
        v = self._anim[k]
        self._anim[k] = v + (tgt - v) * min(1.0, dt * (14.0 if tgt > v else 6.0))
        if self._anim[k] < 0.004 and k != self._hk: del self._anim[k]
    if self._hk is not None and self._hk not in self._anim: self._anim[self._hk] = 0.0
    return fired


def _tb_render(self, surface: pygame.Surface):
    for b in self.BTNS:
        r = self._rects[b.key]
        anim = self._anim.get(b.key, 0.0)
        on = False
        surface.blit(ui_button(r.w, r.h, b.color, "on" if on else "off", anim, 10), r.topleft)
        pygame.draw.rect(surface, b.color, (r.x + 10, r.y + 1, r.w - 20, 3), border_radius=2)             # coloured identity strip
        lab = self.mode_label if b.key == "mode3d" else (self.cur_mode if b.key == "mode" else b.label)
        im = ui_text(self._font, lab, (240, 244, 255))
        surface.blit(im, (r.x + (r.w - im.get_width()) // 2, r.y + 6 + (r.h - 6 - im.get_height()) // 2))


Toolbar.__init__, Toolbar.update, Toolbar.render = _tb_init, _tb_update, _tb_render


# ─────────────────────────────── BRUSH / FX list panels share the same look ───────────────────────────────
def _brush_render(self, surface: pygame.Surface, active: str, rgb):
    CY = (60, 220, 255)
    surface.blit(ui_tab(self.TAB.w, self.TAB.h, CY, bool(self.open), (self._ht / UI_HOVER_S) if self._hk == "_tab" else 0.0), self.TAB.topleft)
    lb = ui_text(self._f, "BRUSH", (245, 250, 255))
    surface.blit(lb, (self.TAB.x + (self.TAB.w - lb.get_width()) // 2, self.TAB.y + (self.TAB.h - lb.get_height()) // 2))
    if not self.open: return
    surface.blit(ui_panel_bg(self.PANEL.w, self.PANEL.h, CY, ui_pulse(210, 30)), self.PANEL.topleft)
    for i, k in enumerate(BRUSH_ORDER):
        r = self._rows[k]
        prog = (self._ht / UI_HOVER_S) if self._hk == k else 0.0
        surface.blit(ui_button(r.w, r.h, CY, "on" if k == active else "off", prog), r.topleft)
        surface.blit(brush_preview(k, rgb), (r.x + 4, r.y + 2))
        nm = ui_text(self._f, BRUSHES[k].name, (235, 245, 255))
        surface.blit(nm, (r.x + 100, r.y + (r.h - nm.get_height()) // 2))
        fk = ui_text(self._s, "F%d" % (i + 1), (140, 170, 190))
        surface.blit(fk, (r.right - fk.get_width() - 6, r.y + (r.h - fk.get_height()) // 2))


def _fx_render(self, surface: pygame.Surface, current: Optional[str]):
    MG = (190, 130, 255)
    surface.blit(ui_tab(self.TAB.w, self.TAB.h, MG, bool(self.open), (self._ht / UI_HOVER_S) if self._hk == "_tab" else 0.0), self.TAB.topleft)
    lb = ui_text(self._f, "FX", (245, 238, 255))
    surface.blit(lb, (self.TAB.x + (self.TAB.w - lb.get_width()) // 2, self.TAB.y + (self.TAB.h - lb.get_height()) // 2))
    if not self.open: return
    surface.blit(ui_panel_bg(self.PANEL.w, self.PANEL.h, MG, ui_pulse(210, 30)), self.PANEL.topleft)
    for k in EFFECT_ORDER:
        r = self._rows[k]
        sel = (k == current) or (k == "none" and current not in EFFECTS)
        prog = (self._ht / UI_HOVER_S) if self._hk == k else 0.0
        surface.blit(ui_button(r.w, r.h, MG, "on" if sel else "off", prog), r.topleft)
        col = EFFECTS[k].color if k in EFFECTS else (150, 150, 160)
        pygame.draw.circle(surface, col, (r.x + 18, r.centery), 7)
        pygame.draw.circle(surface, (255, 255, 255), (r.x + 18, r.centery), 7, 1)
        nm = ui_text(self._f, EFFECTS[k].name if k in EFFECTS else "No effect", (240, 232, 255))
        surface.blit(nm, (r.x + 36, r.y + (r.h - nm.get_height()) // 2))


BrushPanel.render, EffectPanel.render = _brush_render, _fx_render


Toolbar.blocks = lambda self, pt: bool(pt) and any(r.collidepoint(pt) for r in self._hit_rects.values())
ColorPalette.blocks = lambda self, pt: bool(pt) and any(r.inflate(10, 10).collidepoint(pt) for r in self._rects.values())


# ─────────────────────────────── classic HUD: compact, clear of the tab row and the toolbar ───────────────────────────────
HUD_STATE = {"hide": False}
_HUD_MODES = {"DRAW": ("DRAW", (90, 230, 90)), "LAUNCH": ("LAUNCH", (240, 195, 60)), "PALM": ("MENU", (150, 200, 240)), "GRAB": ("GRAB", (90, 190, 240)),
              "ERASE": ("ERASE", (240, 110, 70)), "PINCH": ("RESIZE", (210, 110, 240)), "PUSH": ("PUSH", (240, 150, 70)),
              "BLAST": ("BLAST", (240, 90, 60)), "IDLE": ("idle", (150, 155, 170))}


def _hud_render(self, surface, hands, color_key, pen_w, fps, n_bodies, grav_idx, show_trails, wind, extra=None):
    W, H = surface.get_size()
    x = 12
    for hd in sorted(hands, key=lambda h: h.label != "Right"):                   # hand badges: one tidy row above the tabs
        txt, col = _HUD_MODES.get(hd.gesture, _HUD_MODES["IDLE"])
        im = ui_text(self._med, "%s  %s" % ("R" if hd.label == "Right" else "L", txt), col)
        w = im.get_width() + 24
        self._panel(surface, x, 12, w, 34, (6, 10, 20), 190)
        pygame.draw.rect(surface, col, (x, 12, 4, 34))
        surface.blit(im, (x + 14, 12 + (34 - im.get_height()) // 2))
        x += w + 8
    g_cols = [(120, 175, 240), (240, 175, 120), (210, 120, 240)]
    lines = [("gravity  %s   [G]" % GRAVITY_LABELS[grav_idx], g_cols[grav_idx]),
             ("trails  %s   [T]" % ("ON" if show_trails else "off"), (90, 225, 90) if show_trails else (150, 155, 170)),
             ("pen  %dpx   [+ / -]" % pen_w, (215, 220, 230))]
    if wind != 0.0: lines.append(("wind  %s %d" % ("<" if wind < 0 else ">", abs(int(wind))), (60, 225, 215)))
    for txt, col in (extra or []): lines.append((txt, col))
    y0 = 108
    if not HUD_STATE["hide"]: self._panel(surface, 10, y0, 262, 12 + 23 * len(lines), (6, 10, 20), 170)
    for i, (txt, col) in enumerate(lines if not HUD_STATE["hide"] else []):
        surface.blit(ui_text(self._small, txt, col), (20, y0 + 8 + 23 * i))
    self._panel(surface, 0, H - 26, W, 26, (6, 10, 20), 200)
    bar = " FPS %3.0f   Objects %d   |   Space launch   C clear   Z undo   Y redo   S photo   Ctrl+/ guide   Q quit" % (fps, n_bodies)
    surface.blit(self._small.render(bar, True, (165, 172, 190)), (6, H - 21))


HUD.render = _hud_render
Toolbar_Y_OFFSET = 36


# ─────────────────────────────── layers panel: two tidy rows per layer ───────────────────────────────
def _layer_specs(self, app) -> List[RowSpec]:
    L = app.lyr; rows = [RowSpec(header="LAYERS  (top of list = drawn last)")]
    for i in L.display_rows():
        lay = L[i]; renaming = app.rename_id == i
        name = (app.rename_buf + "_") if renaming else lay.name
        editable = i in ("drawing", "objects", "effects")
        rows.append(RowSpec(cells=[Cell(name[:14], "select", i, on=(L.selected == i)),
                                   Cell("SHOW" if lay.visible else "HIDDEN", "vis", i, on=not lay.visible),
                                   Cell("LOCKED" if lay.locked else "unlocked", "lock", i, on=lay.locked, enabled=editable)]))
        movable = i in L.order
        rows.append(RowSpec(cells=[Cell("opacity  " + ("%d%%" % round(lay.opacity * 100) if i != "ui" else "--")),
                                   Cell("-", "op", (i, -0.1), repeat=True, enabled=i != "ui"), Cell("+", "op", (i, 0.1), repeat=True, enabled=i != "ui"),
                                   Cell("UP", "up", i, enabled=movable), Cell("DN", "down", i, enabled=movable),
                                   Cell("CLR", "clr", i, enabled=editable)]))
    rows.append(RowSpec(header="Ctrl+R renames the selected layer"))
    return rows


LayerPanel.specs = _layer_specs


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 10 — smooth UI pointer · two-hand drawing · quick-guide overlay
# ══════════════════════════════════════════════════════════════════════════════
HELP_LINES = [
    ("HANDS", None),
    ("Right index finger only", "draw"),
    ("Right thumb + index pinch", "pen size"),
    ("Right fist", "erase"),
    ("Right PEACE sign (hold 0.5 s)", "LAUNCH the drawing"),
    ("Right open palm (hold)", "radial menu"),
    ("Thumb + middle pinch (either hand)", "grab / move / throw"),
    ("Left fist", "push objects"),
    ("Left open palm", "blast"),
    ("BOTH index fingers (press W)", "two-hand drawing"),
    ("RIGHT-SIDE SWITCHES (hold fingertip)", "CLEANUP / 2D>3D / REAL WORLD / BOUNCE"),
    ("MENUS", None),
    ("Hold a fingertip 0.5 s on any button", "press it"),
    ("Top tabs: FILE PLAY ART UI AI TIME ...", "open one menu at a time"),
    ("KEYS", None),
    ("Space launch   C clear   Z undo   Y redo   S photo", ""),
    ("G gravity   T trails   E blast   B brushes   F effects", ""),
    ("W two-hand   M symmetry   J generative   R record", ""),
    ("A AI   / ask   K calculator   U ui   H history   L layers", ""),
    ("Ctrl+S save   Ctrl+G game   Ctrl+T theme   Ctrl+1..9 mode", ""),
    ("Ctrl+K cleanup   N 2D>3D   Ctrl+W real-world   , . bounce", ""),
    ("Ctrl+/ this guide      Q or Esc quit", ""),
]


class Polish10Mixin:
    def _p10_init(self):
        self.two_hand = False
        self.lh_layers = StrokeLayers((WIN_W, WIN_H))
        self.lh_ctx: Optional[StrokeCtx] = None
        self.lh_pts: List[Tuple[int, int]] = []
        self.lh_ema = DualEMA()
        self.lh_t0 = 0.0
        self._uts: Optional[Tuple[float, float]] = None
        self._uts_gap = 0.0
        self.help_on = False
        self._help_surf: Optional[pygame.Surface] = None

    # ---- steady pointer ----------------------------------------------------------------------------
    def _ui_smooth(self, tip, dt: float):
        if tip is None:
            self._uts_gap += dt
            if self._uts_gap > 0.5: self._uts = None
            return None
        self._uts_gap = 0.0
        if self._uts is None or math.hypot(tip[0] - self._uts[0], tip[1] - self._uts[1]) > 140:
            self._uts = (float(tip[0]), float(tip[1]))                          # a jump (other hand / re-acquired): no lag
        else:
            a = 1.0 - math.exp(-max(dt, 1e-3) / UI_TIP_TAU)
            nx = self._uts[0] + (tip[0] - self._uts[0]) * a
            ny = self._uts[1] + (tip[1] - self._uts[1]) * a
            if math.hypot(nx - self._uts[0], ny - self._uts[1]) >= 0.8: self._uts = (nx, ny)    # sub-pixel shimmer is ignored
        return (int(round(self._uts[0])), int(round(self._uts[1])))

    def _ui_blocked_at(self, pt) -> bool:
        if pt is None: return False
        if (self.brush_panel.blocks(pt) or self.fx_panel.blocks(pt) or self.edit_panel.blocks(pt) or self.layer_panel.blocks(pt)
                or self.time_panel.blocks(pt) or self.chip.blocks(pt) or self.toolbar.blocks(pt) or self.palette.blocks(pt) or self.switches.blocks(pt)):
            return True
        return any(p.blocks(pt) for p, _ in self._ext)

    # ---- two-hand drawing ----------------------------------------------------------------------------
    def _lh_rgb(self):
        h, s, v = colorsys.rgb_to_hsv(self.rgb[0] / 255.0, self.rgb[1] / 255.0, self.rgb[2] / 255.0)
        if s < 0.25: return (255, 165, 60) if v > 0.5 else (90, 200, 255)
        r, g, b = colorsys.hsv_to_rgb((h + 0.5) % 1.0, max(s, 0.65), max(v, 0.9))
        return (int(r * 255), int(g * 255), int(b * 255))

    def _hand_col(self, label: str):
        if label == "Right": return self.rgb
        return self._lh_rgb() if self.two_hand else (80, 160, 230)

    def toggle_two_hand(self):
        self.two_hand = not self.two_hand
        if not self.two_hand: self._lh_commit()
        self.notify.show("TWO-HAND DRAWING " + ("ON - both index fingers draw (left = complementary colour)" if self.two_hand else "OFF"), (120, 230, 255))

    def _lh_reset(self):
        self.lh_layers.clear(); self.lh_ctx = None; self.lh_pts = []; self.lh_ema.reset()

    def _lh_update(self, lg: str, ltip, dt: float):
        want = (self.two_hand and not self.mode3d and lg == "DRAW" and ltip is not None and not self._ui_blocked_at(ltip)
                and not self._mode_blocks_ink() and not self.lyr.locked("drawing"))
        if not want:
            if self.lh_pts or self.lh_ctx is not None: self._lh_commit()
            return
        sx, sy = self.lh_ema.update(float(ltip[0]), float(ltip[1]))
        p = (int(sx), int(sy))
        if not self.lh_pts:
            self.lh_ema.reset(); sx, sy = self.lh_ema.update(float(ltip[0]), float(ltip[1])); p = (int(sx), int(sy))
            self.lh_pts = [p]; self.lh_t0 = time.perf_counter(); self.suggest = None; self.chip.hide()
            self.lh_layers.clear()
            self.lh_ctx = self.brush.new_ctx(self._lh_rgb(), self.pen_w, p, seed=random.getrandbits(32))
            return
        lx, ly = self.lh_pts[-1]
        if math.hypot(p[0] - lx, p[1] - ly) >= MIN_MOVE_PX:
            br = BRUSHES[self.lh_ctx.brush_key]
            for fp in self._densify((lx, ly), p):
                if self.sym_mode != "off": self._sym_live_seg(self.lh_pts[-1], fp)
                self.lh_layers.draw(br.feed(self.lh_ctx, fp, self.lh_ctx.w), self.lh_ctx.passes)
                self.lh_pts.append(fp)
        br = BRUSHES[self.lh_ctx.brush_key]
        if br.emits and self._fx_rng.random() < 0.5:
            br.emit(self.fx, self.lh_pts[-1], self.lh_ctx.rgb, self.lh_ctx.w, self._fx_rng, 0.0)

    def _lh_commit(self):
        ctx, pts = self.lh_ctx, self.lh_pts
        entry, pre = None, None
        if len(pts) >= 2 and ctx is not None:
            pre = self._art_pre()
            self.undo.push(self.canvas)
            self.lh_layers.flatten_into(self.canvas, ctx.passes)
            entry = (list(pts), ctx.rgb, ctx.w, ctx.brush_key)
            self.pending.append(entry); self.action_log.append("ink"); self._ai_note_stroke(entry)
        self.lh_layers.clear(); self.lh_ctx = None; self.lh_pts = []; self.lh_ema.reset()
        if entry is not None:
            t0, self._stroke_t0 = self._stroke_t0, self.lh_t0
            self._art_after_commit(entry, pre)
            self._stroke_t0 = t0
            self._log("Stroke (left hand)")
            if self.mode_name == "PHYSICS": self._launch()

    def _lh_render(self):
        if self.lh_ctx is not None: self.lh_layers.render(self.screen, self.lh_ctx.passes)

    # ---- quick guide overlay -----------------------------------------------------------------------------
    def toggle_help(self):
        self.help_on = not self.help_on

    def _help_render(self, s: pygame.Surface):
        if not self.help_on: return
        if self._help_surf is None:
            W, H = 760, 58 + 21 * len(HELP_LINES) + 14
            card = pygame.Surface((W, H), pygame.SRCALPHA)
            pygame.draw.rect(card, (8, 12, 24, 240), card.get_rect(), border_radius=16)
            pygame.draw.rect(card, (110, 220, 255, 255), card.get_rect(), 2, border_radius=16)
            tf, hf, bf, sf = (pygame.font.SysFont("monospace", z, bold=True) for z in (22, 15, 14, 14))
            card.blit(tf.render("QUICK GUIDE", True, (110, 220, 255)), (22, 14))
            card.blit(sf.render("Ctrl+/ to close", True, (150, 170, 200)), (W - 170, 20))
            y = 54
            for a, b in HELP_LINES:
                if b is None:
                    y += 4; card.blit(hf.render(a, True, (255, 210, 110)), (22, y)); y += 20; continue
                card.blit(bf.render(a, True, (235, 240, 255)), (34, y))
                if b: card.blit(sf.render("-> " + b, True, (150, 220, 190)), (400, y))
                y += 20
            self._help_surf = card
        r = self._help_surf.get_rect(midtop=(WIN_W // 2, 104))
        s.blit(self._help_surf, r.topleft)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 12 — right-side switches (cleanup · 2D>3D · real-world collide · bounce) · camera edges as obstacles
# ══════════════════════════════════════════════════════════════════════════════
BOUNCE_STEP = 0.05
_BOUNCE = {"v": None}                       # None = the app's original per-object defaults until the user touches the control
WORLD_EVERY_S = 0.25                        # how often the camera is re-read for obstacles
WORLD_MAX_SEGS = 150


class RealWorld:
    """Turns the camera picture into solid edges (table edge, book, screen, cup rim ...).  Hands are masked out, so your own
    hands never become obstacles.  Works on a 320x180 copy of the picture - cheap - and is only ever run while the switch is ON."""
    SW, SH = 320, 180

    def __init__(self):
        self.enabled = False
        self.segs: List[Tuple[float, float, float, float]] = []     # display pixels
        self._acc: Optional[np.ndarray] = None
        self._t = 0.0
        self._sig: set = set()
        self.version = 0

    def reset(self):
        self._acc, self._t, self._sig, self.segs = None, 0.0, set(), []
        self.version += 1

    def _hand_mask(self, hands) -> np.ndarray:
        SW, SH = self.SW, self.SH
        m = np.zeros((SH, SW), np.uint8)
        for h in hands or []:
            try:
                lms = getattr(h.landmarks, "landmark", h.landmarks)
                P = np.array([[l.x * SW, (1.0 - l.y) * SH] for l in lms], np.float32)
                cv2.fillConvexPoly(m, cv2.convexHull(P).astype(np.int32), 255)
                w, mc = P[0], P[9]                                    # wrist, middle knuckle -> extend the mask along the forearm
                d = w - mc; n = float(np.hypot(d[0], d[1])) or 1.0; d /= n
                perp = np.array([-d[1], d[0]], np.float32)
                half = max(8.0, float(np.hypot(*(P[5] - P[17]))) * 0.75)
                far = w + d * 260.0
                quad = np.array([w + perp * half, w - perp * half, far - perp * half * 1.6, far + perp * half * 1.6], np.float32)
                cv2.fillConvexPoly(m, quad.astype(np.int32), 255)
            except Exception:
                continue
        return cv2.dilate(m, np.ones((17, 17), np.uint8))

    def update(self, frame_bgr: Optional[np.ndarray], hands, now: float) -> bool:
        """Re-read the camera every WORLD_EVERY_S.  True when the set of obstacle edges changed enough to rebuild physics."""
        if not self.enabled or frame_bgr is None or now - self._t < WORLD_EVERY_S: return False
        self._t = now
        small = cv2.flip(cv2.resize(frame_bgr, (self.SW, self.SH), interpolation=cv2.INTER_AREA), 0)   # vertical flip only = same space as the picture on screen
        g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        e = (cv2.Canny(g, 45, 120) > 0).astype(np.float32)
        self._acc = 0.5 * e if self._acc is None else 0.5 * self._acc + 0.5 * e         # an edge must persist -> camera noise never becomes a wall
        stable = ((self._acc > 0.6).astype(np.uint8)) * 255
        stable[self._hand_mask(hands) > 0] = 0
        stable[:3, :] = stable[-3:, :] = 0; stable[:, :3] = stable[:, -3:] = 0
        lines = cv2.HoughLinesP(stable, 1, np.pi / 180, threshold=12, minLineLength=16, maxLineGap=4)
        segs: List[Tuple[float, float, float, float]] = []
        if lines is not None:
            L = sorted((tuple(int(v) for v in q) for q in np.asarray(lines).reshape(-1, 4)), key=lambda q: -math.hypot(q[2] - q[0], q[3] - q[1]))[:WORLD_MAX_SEGS]
            kx, ky = WIN_W / self.SW, WIN_H / self.SH
            segs = [(float(a * kx), float(b * ky), float(c * kx), float(d * ky)) for a, b, c, d in L]
        sig = {(int(a // 10), int(b // 10), int(c // 10), int(d // 10)) for a, b, c, d in segs}
        diff = len(sig ^ self._sig)
        if diff <= 0.2 * max(len(sig), len(self._sig), 1) and self.segs: return False     # nearly the same picture: leave physics alone
        self._sig, self.segs = sig, segs
        self.version += 1
        return True

    def render(self, surface: pygame.Surface):
        if not self.enabled: return
        for a, b, c, d in self.segs:
            pygame.draw.line(surface, (70, 255, 215), (int(a), int(b)), (int(c), int(d)), 2)


_orig_add_stroke = PhysicsWorld.add_stroke
def _add_stroke12(self, pts, color, width, brush=DEFAULT_BRUSH):
    r = _orig_add_stroke(self, pts, color, width, brush)
    v = _BOUNCE["v"]
    if v is not None and self.bodies:
        pb = self.bodies[-1]; pb.elast = v
        for sh in pb.shapes: sh.elasticity = v
    return r
PhysicsWorld.add_stroke = _add_stroke12

_orig_scene_add = Scene3D.add
def _scene_add12(self, s, morph: bool = True):
    r = _orig_scene_add(self, s, morph)
    if _BOUNCE["v"] is not None: s.bounce = _BOUNCE["v"]
    return r
Scene3D.add = _scene_add12

_orig_scene_step = Scene3D.step
def _scene_step12(self, dt: float):
    _orig_scene_step(self, dt)
    sg = getattr(self, "world_segs", None)
    if sg is None or len(sg) == 0: return
    view, e = self.view, float(getattr(self, "world_bounce", 0.5))
    A, B = sg[:, 0:2], sg[:, 2:4]; AB = B - A; L2 = np.maximum((AB ** 2).sum(1), 1e-6)
    for s in self.solids:
        if s.building or s.held or not s.active or s.frozen or s.morph < 1.0 or len(s.V) == 0: continue
        csx, csy, k, _ = self.center_screen(s)
        r = max(6.0, s.radius * s.scale * 0.75 * k)
        P = np.array([csx, csy])
        t = np.clip(((P - A) * AB).sum(1) / L2, 0.0, 1.0)
        D = P - (A + AB * t[:, None]); dist = np.sqrt((D ** 2).sum(1))
        i = int(dist.argmin())
        if dist[i] >= r: continue
        n = D[i] / dist[i] if dist[i] > 1e-6 else np.array([0.0, -1.0])
        s.pos += view.R.T @ np.array([n[0] * (r - dist[i]) / k, n[1] * (r - dist[i]) / k, 0.0])
        vv = view.R @ s.vel
        vn = float(vv[0] * n[0] + vv[1] * n[1])
        if vn < 0:
            f = 1.0 if abs(vn) < 40.0 else (1.0 + e)                    # resting contact: just stop, no micro-bounce
            vv[0] -= f * vn * n[0]; vv[1] -= f * vn * n[1]
            s.vel = view.R.T @ vv
            s.ang *= 0.95
Scene3D.step = _scene_step12


class SwitchPanel:
    """Always-visible switches on the right edge.  Hold a fingertip on one for 0.5 s (or click / press its key)."""
    ACC = {"cleanup": (190, 120, 255), "to3d": (170, 130, 255), "world": (60, 230, 200), "bounce": (255, 190, 70)}

    def __init__(self):
        S = min(UI_SCALE, 1.2)
        self.W, self.ROW, self.GAP, self.PAD = int(152 * S), int(34 * S), 6, 10
        self.x = WIN_W - self.W - 10; self.y = 112
        self.visible = True
        self._font = pygame.font.SysFont("monospace", int(13 * S), bold=True)
        self._small = pygame.font.SysFont("monospace", int(11 * S))
        px, py, W, R, G = self.x + self.PAD, self.y + self.PAD + 18, self.W - 2 * self.PAD, self.ROW, self.GAP
        self.rects: Dict[str, pygame.Rect] = {}
        for i, k in enumerate(("cleanup", "to3d", "world")):
            self.rects[k] = pygame.Rect(px, py + i * (R + G), W, R)
        by = py + 3 * (R + G)
        self.rects["bounce-"] = pygame.Rect(px, by, 42, R)
        self.rects["bounce+"] = pygame.Rect(px + W - 42, by, 42, R)
        self.rects["_bval"] = pygame.Rect(px + 42, by, W - 84, R)
        self.status_y = by + R + 4
        self.panel_rect = pygame.Rect(self.x, self.y, self.W, self.status_y + 18 + self.PAD - self.y)
        self._hk, self._ht, self._lost, self._anim = None, 0.0, 0.0, {}
        self._labels = {"cleanup": "CLEANUP", "to3d": "2D > 3D", "world": "REAL WORLD"}

    def blocks(self, pt) -> bool:
        return bool(self.visible and pt is not None and self.panel_rect.inflate(8, 8).collidepoint(pt))

    def _hit(self, pt):
        if pt is None or not self.visible: return None
        if self._hk in self.rects and self._hk != "_bval" and self.rects[self._hk].inflate(2 * UI_STICKY_PX, 8).collidepoint(pt): return self._hk
        for k, r in self.rects.items():
            if k != "_bval" and r.inflate(6, 6).collidepoint(pt): return k
        return None

    def click(self, pos) -> Optional[str]:
        if not self.visible: return None
        for k, r in self.rects.items():
            if k != "_bval" and r.collidepoint(pos): return k
        return None

    def update(self, tip, dt: float) -> Optional[str]:
        hov = self._hit(tip)
        grace = False
        if hov is None and self._hk is not None and self._lost < UI_LOST_GRACE and self.visible:
            self._lost += dt; hov = self._hk; grace = True
        elif hov is not None: self._lost = 0.0
        fired = None
        if hov is not None and hov == self._hk:
            if not grace:
                self._ht += dt
                if self._ht >= UI_HOVER_S:
                    fired = hov
                    self._ht = (UI_HOVER_S - 0.28) if hov.startswith("bounce") else 0.0      # -/+ keep repeating while held
        else:
            self._hk, self._ht = hov, 0.0
        for k in list(self._anim):
            tgt = min(self._ht / UI_HOVER_S, 1.0) if k == self._hk else 0.0
            v = self._anim[k]; self._anim[k] = v + (tgt - v) * min(1.0, dt * (14.0 if tgt > v else 6.0))
            if self._anim[k] < 0.004 and k != self._hk: del self._anim[k]
        if self._hk is not None and self._hk not in self._anim: self._anim[self._hk] = 0.0
        return fired

    def render(self, surface: pygame.Surface, st: dict):
        if not self.visible: return
        A = self.ACC["world"]
        surface.blit(ui_panel_bg(self.panel_rect.w, self.panel_rect.h, A, ui_pulse(210, 30)), self.panel_rect.topleft)
        t = ui_text(self._small, "QUICK SWITCHES", (170, 205, 215))
        surface.blit(t, (self.panel_rect.centerx - t.get_width() // 2, self.y + self.PAD))
        for k in ("cleanup", "to3d", "world"):
            r, on = self.rects[k], bool(st.get(k))
            surface.blit(ui_button(r.w, r.h, self.ACC[k], "on" if on else "off", self._anim.get(k, 0.0)), r.topleft)
            lb = ui_text(self._font, self._labels[k], (240, 244, 255) if on else (190, 198, 215))
            surface.blit(lb, (r.x + 10, r.centery - lb.get_height() // 2))
            pw, ph = 34, 18; pr = pygame.Rect(r.right - pw - 8, r.centery - ph // 2, pw, ph)
            pygame.draw.rect(surface, (46, 200, 120) if on else (62, 68, 88), pr, border_radius=ph // 2)
            pygame.draw.circle(surface, (250, 252, 255), (pr.right - ph // 2 if on else pr.x + ph // 2, pr.centery), ph // 2 - 2)
        A = self.ACC["bounce"]
        for k, sym in (("bounce-", "-"), ("bounce+", "+")):
            r = self.rects[k]
            surface.blit(ui_button(r.w, r.h, A, "off", self._anim.get(k, 0.0)), r.topleft)
            im = ui_text(self._font, sym, (245, 245, 250)); surface.blit(im, (r.centerx - im.get_width() // 2, r.centery - im.get_height() // 2))
        r = self.rects["_bval"]
        v1 = ui_text(self._small, "BOUNCE", (200, 190, 150)); v2 = ui_text(self._font, "%d%%" % round(st.get("bounce", 0.5) * 100), (255, 225, 150))
        surface.blit(v1, (r.centerx - v1.get_width() // 2, r.y + 2)); surface.blit(v2, (r.centerx - v2.get_width() // 2, r.bottom - v2.get_height() - 2))
        if st.get("world"):
            msg = "%d edges solid" % st.get("edges", 0)
            im = ui_text(self._small, msg, (120, 245, 215))
        else:
            im = ui_text(self._small, "camera edges: off", (130, 140, 160))
        surface.blit(im, (self.panel_rect.centerx - im.get_width() // 2, self.status_y))


class World12Mixin:
    def _w12_init(self):
        self.switches = SwitchPanel()
        self.world = RealWorld()
        self.bounce_lvl = 0.5
        self._world_shapes: list = []

    def _switch_state(self) -> dict:
        return {"cleanup": self.ai_auto_clean, "to3d": self.auto3d, "world": self.world.enabled,
                "bounce": self.bounce_lvl, "edges": len(self.world.segs)}

    def _switches_visible(self) -> bool:
        pn = (self.brush_panel, self.fx_panel, self.edit_panel, self.layer_panel, self.time_panel)
        return not (any(p.open for p in pn) or any(p.open for p, _ in self._ext) or self.help_on)

    def switch_do(self, key: str):
        if key == "cleanup":
            self.ai_auto_clean = not self.ai_auto_clean
            self.notify.show("Cleanup ON - shapes tidy themselves" if self.ai_auto_clean else "Cleanup OFF", (190, 120, 255))
        elif key == "to3d":
            self.auto3d = not self.auto3d
            self.notify.show("2D > 3D ON - drawings become 3D" if self.auto3d else "2D > 3D OFF", (170, 130, 255))
        elif key == "world": self.toggle_world()
        elif key == "bounce-": self.set_bounce(self.bounce_lvl - BOUNCE_STEP)
        elif key == "bounce+": self.set_bounce(self.bounce_lvl + BOUNCE_STEP)

    def set_bounce(self, v: float):
        v = max(0.0, min(1.0, round(v / BOUNCE_STEP) * BOUNCE_STEP))
        self.bounce_lvl = v; _BOUNCE["v"] = v
        for sh in self.phys.space.static_body.shapes: sh.elasticity = v
        for pb in self.phys.bodies:
            pb.elast = v
            for sh in pb.shapes: sh.elasticity = v
        for s in self.scene3d.solids: s.bounce = v
        self.scene3d.world_bounce = v
        self.notify.show("Bounce %d%%" % round(v * 100), (255, 190, 70))

    def toggle_world(self, on: Optional[bool] = None):
        on = (not self.world.enabled) if on is None else bool(on)
        self.world.enabled = on
        self.world.reset()
        self._world_apply()
        self.notify.show("REAL WORLD ON - objects bounce off edges the camera sees" if on else "Real world OFF", (60, 230, 200))

    def _world_apply(self):
        sp = self.phys.space
        if self._world_shapes:
            sp.remove(*self._world_shapes); self._world_shapes = []
        segs = self.world.segs if self.world.enabled else []
        v = self.bounce_lvl
        sb = sp.static_body
        for a, b, c, d in segs:
            sh = pymunk.Segment(sb, (a, b), (c, d), 3.0); sh.elasticity = v; sh.friction = 0.6
            self._world_shapes.append(sh)
        if self._world_shapes: sp.add(*self._world_shapes)
        self.scene3d.world_segs = np.array(segs, float).reshape(-1, 4) if segs else None
        self.scene3d.world_bounce = v

    def _world_update(self, frame, hands):
        if self.world.enabled and self.world.update(frame, hands, time.perf_counter()):
            self._world_apply()


class App(AIMixin, UIMixin, ArtMixin, Polish10Mixin, World12Mixin):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.NOFRAME)
        pygame.display.set_caption("Air Draw Physics v10")
        self.clock  = pygame.time.Clock()

        self.cam = WebcamStream(WEBCAM_INDEX, CAM_W, CAM_H)
        self.clf = GestureClassifier()
        self.tracker = TrackerThread(self.cam, self.clf) if (THREADED_TRACKING and hasattr(self.cam, "read_with_seq")) else None
        self._bg_frame = None
        self._bg_surf = None
        self._dt_s = 1.0 / 60.0
        self._next_t = time.perf_counter()
        self.phys = PhysicsWorld(WIN_W, WIN_H)
        self.parts = ParticleSystem()
        self.hud = HUD()
        self.notify = Notifier()
        self.palette = ColorPalette()
        self.toolbar = Toolbar()
        self.radial = RadialMenu()
        self.cursor = GlowCursor()
        self.undo = UndoStack()
        self.brush_key = DEFAULT_BRUSH
        self.brush_panel = BrushPanel()
        self.fx = FXSystem()
        self.layers = StrokeLayers((WIN_W, WIN_H))
        self.stroke_ctx: Optional[StrokeCtx] = None
        self._fx_rng = random.Random()
        self.grab = GrabController(self.phys)
        self.scene3d = Scene3D(self.phys)
        self.grab3d = Grab3DController(self.scene3d)
        self.depth = DepthEstimator()
        self.mode3d = False
        self.auto3d = AUTO_3D_DEFAULT
        self.chip = ConvertChip()
        self.suggest: Optional[Suggestion] = None
        self.action_log: List[str] = []
        self.stroke3d: Optional[Solid3D] = None
        self._stroke_z: List[float] = []
        self._z_now = 0.0
        self._z_s = 0.0
        self._orbit_prev = None
        self._ink_lock = None
        self.fx_panel = EffectPanel()
        self.pending_fx: Dict[int, dict] = {}                # id(pts) -> {key, pts, ring, geom, cen}
        self.pending_shape: Dict[int, tuple] = {}            # id(pts) -> (ShapeResult, pts)
        self.fx_last = None                                  # ("pending"|"body"|"solid", obj): default effect target
        self._panels_open = (False,) * 5
        self.lyr = LayerManager()
        self.hist = HistoryLog()
        self.edit_panel, self.layer_panel, self.time_panel = EditPanel(), LayerPanel(), TimelinePanel()
        self._ai_init()
        self._ext = [(self.ai_panel, self.ai_apply)]          # extra flyout panels (Phase 7+): (panel, action handler)
        self._ui_init()
        self._art_init()
        self._p10_init()
        self._w12_init()
        self.rename_id: Optional[str] = None
        self.rename_buf = ""
        self._redo: List[tuple] = []
        self._lock_warn_t = 0.0
        self.debug_overlay = DebugOverlay()
        self.debug = False
        self.ema: Dict[str, DualEMA] = {"Right": DualEMA(), "Left": DualEMA()}
        self.vel: Dict[str, VelocityTracker] = {"Right": VelocityTracker(),"Left":  VelocityTracker()}
        self.canvas = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        self.canvas.fill((0, 0, 0, 0))

        self.color_key: str = DEFAULT_COLOR_KEY
        self.pen_w: int = DEFAULT_PEN_W
        self.cur_stroke: List[Tuple[int,int]] = []
        self.pending: List[Tuple[List, Tuple, int, str]] = []   # (points, rgb, width, brush)
        self.show_trails = True
        self._wind = 0.0
        self.prev_g: Dict[str, str] = {"Right": "IDLE", "Left": "IDLE"}
        self.fps: float = 0.0
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    @property
    def rgb(self) -> Tuple[int,int,int]:
        return PALETTE[self.color_key]["rgb"]

    @property
    def brush(self) -> Brush:
        return BRUSHES[self.brush_key]

    def _select_brush(self, key: str):
        if key in BRUSHES:
            self.brush_key = key
            self.notify.show(f"Brush: {BRUSHES[key].name}", self.rgb)
            self._log(f"Brush: {BRUSHES[key].name}")

    def _cycle_brush(self, step: int = 1):
        i = BRUSH_ORDER.index(self.brush_key)
        self._select_brush(BRUSH_ORDER[(i + step) % len(BRUSH_ORDER)])

    def _begin_stroke(self, p: Tuple[int, int]):
        self.layers.clear()
        self._stroke_t0 = time.perf_counter()
        self.stroke_ctx = self.brush.new_ctx(self.rgb, self.pen_w, p, seed=random.getrandbits(32))

    def _paint(self, p1: Tuple[int,int], p2: Tuple[int,int]):
        """Paint the next stroke point with the active brush (layered passes + particles)."""
        ctx = self.stroke_ctx
        if ctx is None:
            self._begin_stroke(p1)
            ctx = self.stroke_ctx
        br = BRUSHES[ctx.brush_key]
        if self.sym_mode != "off": self._sym_live_seg(p1, p2)
        self.layers.draw(br.feed(ctx, p2, ctx.w), ctx.passes)
        if br.emits and self._fx_rng.random() < 0.5:
            br.emit(self.fx, p2, ctx.rgb, ctx.w, self._fx_rng, 0.0)

    def _densify(
        self,
        p1: Tuple[int,int],
        p2: Tuple[int,int],
    ) -> List[Tuple[int,int]]:
        dist  = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        steps = max(1, int(dist / DENSIFY_STEP))
        result = []
        for k in range(1, steps + 1):
            t = k / steps
            result.append((
                int(p1[0] + (p2[0]-p1[0]) * t),
                int(p1[1] + (p2[1]-p1[1]) * t),
            ))
        return result

    def _erase(self, cx: int, cy: int):
        if self.lyr.locked("drawing"):
            self._warn_locked("Drawing"); return
        r    = ERASE_RADIUS
        mask = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 255))
        pygame.draw.circle(mask, (0, 0, 0, 0), (r, r), r)
        self.canvas.blit(mask, (cx-r, cy-r),
                         special_flags=pygame.BLEND_RGBA_MIN)

    def _commit(self):
        if self.stroke3d is not None:
            self._commit3d()
            return
        ctx = self.stroke_ctx
        entry = None
        _pre = None
        if len(self.cur_stroke) >= 2 and ctx is not None:
            _pre = self._art_pre()
            self.undo.push(self.canvas)                     # snapshot BEFORE the stroke lands
            self.layers.flatten_into(self.canvas, ctx.passes)
            entry = (list(self.cur_stroke), ctx.rgb, ctx.w, ctx.brush_key)
            self.pending.append(entry)
            self.action_log.append("ink")
            self._ai_note_stroke(entry)
        self.layers.clear()
        self.stroke_ctx = None
        self.cur_stroke.clear()
        self.ema["Right"].reset()
        if entry is not None:
            self._art_after_commit(entry, _pre)
            self._log("Stroke")
            self._offer_conversion(entry[0], entry[1], entry[2], entry[3], "2d", 0.0, entry[0])
            if self.mode_name == "PHYSICS" and not self.mode3d: self._launch()

    def _launch(self):
        """Convert all pending ink strokes into physics rigid bodies."""
        self._commit()
        self._lh_commit()
        if self.mode3d and ENABLE_3D:
            self._pending_to_solids()                         # 3-D mode: drawings become solids (smooth morph)
        else:
            for pts, col, wid, bk in self.pending:
                cx, cy = self.phys.add_stroke(pts, col, wid, bk)
                self._give_effect_to_body(pts)
                self.parts.burst(cx, cy, col, P_LAUNCH)
        self.pending.clear()
        self.undo.clear()
        self.canvas.fill((0, 0, 0, 0))
        self.action_log = [a for a in self.action_log if a != "ink"]
        self.suggest = None
        self.chip.hide()
        n3 = self.scene3d.launch()                            # smooth 3-D launch (gravity fades in)
        self.notify.show(f"🚀 Launched!  Bodies: {len(self.phys.bodies)}" + (f"   3D: {n3}" if n3 else ""))
        self._log("Launch")
        self.game.on_launch()

    def _clear(self):
        self._lh_reset()
        if hasattr(self, "_ai_undo"):
            self._ai_undo.clear(); self.ai_rec = None; self.calc_strokes.clear(); self.calc_res = None; self._ai_pending = False
        self.phys.clear()
        self.canvas.fill((0, 0, 0, 0))
        self.cur_stroke.clear()
        self.pending.clear()
        self.layers.clear()
        self.stroke_ctx = None
        self.fx.clear()
        self.undo.clear()
        for ema in self.ema.values(): ema.reset()
        self.grab.reset()
        self.pending_fx.clear()
        self.pending_shape.clear()
        self.fx_last = None
        self.scene3d.clear()
        self.grab3d.reset()
        self.stroke3d = None
        self._stroke_z = []
        self.action_log.clear()
        self.suggest = None
        self.chip.hide()
        self.notify.show("🗑 Cleared")
        self._log("Clear all")

    def _screenshot(self):
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.png")
        pygame.image.save(self.screen, path)
        self.notify.show(f"📸 Saved: {os.path.basename(path)}", (80, 220, 120))

    def _do_action(self, key: str):
        """Central dispatcher for toolbar / radial / keyboard actions."""
        if   key == "launch":     self._launch()
        elif key == "clear":      self._clear()
        elif key == "undo":
            self._undo_any()
        elif key == "redo":
            self._redo_any()
        elif key == "gravity":
            self.notify.show(f"Gravity: {self.phys.cycle_gravity()}")
        elif key == "trails":
            self.show_trails = not self.show_trails
            self.notify.show(f"Trails {'ON' if self.show_trails else 'OFF'}")
        elif key == "screenshot":
            self._screenshot()
        elif key == "brushes":
            self.brush_panel.toggle()
        elif key == "effects":
            self.fx_panel.toggle()
        elif key == "to3d":
            self._convert_all()
        elif key == "mode3d":
            self._toggle_mode()
        elif key == "mode":
            self.cycle_mode()
        elif key == "theme":
            self.cycle_theme()
        elif key == "hud":
            self.cycle_hud()
        elif key == "explode":
            self.phys.explode(WIN_W/2, WIN_H/2)
            self.scene3d.explode(WIN_W/2, WIN_H/2)
            self.parts.burst(WIN_W/2, WIN_H/2, (255,200,50), P_EXPLODE)
            self.notify.show("Explosion!")
        elif key == "wind_l":
            self._wind = -1800 if self._wind >= 0 else 0.0
            self.notify.show("Wind ON" if self._wind else "Wind OFF")
        elif key == "wind_r":
            self._wind =  1800 if self._wind <= 0 else 0.0
            self.notify.show("Wind ON" if self._wind else "Wind OFF")
        self.phys.set_wind(self._wind)

    # ───────────── PHASE 6: 3-D mode ─────────────
    def _toggle_mode(self, on: Optional[bool] = None):
        if not ENABLE_3D:
            self.notify.show("3D is disabled (ENABLE_3D)"); return
        if self.cur_stroke: self._commit()
        self.mode3d = (not self.mode3d) if on is None else on
        if self.mode3d:
            self.depth.reset()
            self.notify.show("3D MODE: move hand toward camera = closer", (170, 130, 255))
        else:
            self.notify.show("2D MODE", (200, 200, 200))
        self.mode_name = "3D" if self.mode3d else ("DRAW" if self.mode_name == "3D" else self.mode_name)
        self._log("3D mode ON" if self.mode3d else "2D mode")

    def _begin_any(self, p):
        self.suggest = None
        self.chip.hide()
        if self.mode3d and ENABLE_3D:
            self.stroke3d = self.scene3d.begin_stroke(self.rgb, self.brush_key, self.pen_w)
            self._stroke_z = []
            self._z_s = self._z_now
            self._paint_any(p, p)
        else:
            self._begin_stroke(p)

    def _paint_any(self, p1, p2):
        if self.stroke3d is not None:
            self._z_s += 0.35 * (self._z_now - self._z_s)
            if self.scene3d.add_point(self.stroke3d, p2[0], p2[1], self._z_s):
                self._stroke_z.append(self._z_s)
        else:
            self._paint(p1, p2)

    def _commit3d(self):
        s, self.stroke3d = self.stroke3d, None
        pts = list(self.cur_stroke)
        zs = self._stroke_z
        self.cur_stroke.clear()
        self._stroke_z = []
        self.ema["Right"].reset()
        if s is None: return
        z = float(np.mean(zs)) if zs else 0.0
        if self.scene3d.finish_stroke(s):
            self.fx_last = ("solid", s)
            self.action_log.append("solid")
            self._log("Stroke (3D)")
            self._offer_conversion(pts, s.rgb, s.width, s.brush, "3d", z, s.uid)

    def _offer_conversion(self, pts, rgb, w, brush, src, z, ref):
        """A stroke just finished -> recognise it (once, geometry only) and offer / do the 3-D conversion."""
        self.suggest = None
        self.chip.hide()
        if src == "2d": self.fx_last = ("pending", ref)
        try:
            res = ShapeRecognizer.recognize(pts)
        except Exception:
            res = None
        if res is not None and src == "2d": self.pending_shape[id(ref)] = (res, ref)
        if not ENABLE_3D or res is None: return
        self.suggest = Suggestion(res, list(pts), rgb, w, brush, src, z, ref)
        if self.auto3d:
            self.convert_suggestion()
        else:
            self.chip.show(res, pts)

    def convert_suggestion(self) -> bool:
        """Turn the stroke that was just drawn into its 3-D solid (circle->sphere, square->cube, ...)."""
        s = self.suggest
        if s is None: return False
        self.suggest = None
        self.chip.hide()
        eff = None
        try:
            if s.src == "2d":
                if self.pending and self.pending[-1][0] is s.ref:
                    eff = self._pending_effect_key(s.ref)
                    self.undo.pop(self.canvas)               # lift the ink off the canvas (exact snapshot restore)
                    self.pending.pop()
                    if self.action_log and self.action_log[-1] == "ink": self.action_log.pop()
                else:
                    self.notify.show("That stroke is no longer editable")
                    return False
            else:
                old = next((o for o in self.scene3d.solids if o.uid == s.ref), None)
                eff = getattr(old, "effect", None)
                self.scene3d.remove_uid(s.ref)
                if self.action_log and self.action_log[-1] == "solid": self.action_log.pop()
            solid = build_solid(s.result, s.rgb, s.brush, s.w, self.scene3d.view, s.z)
            self.scene3d.add(solid, morph=True)
            if eff: solid.effect = eff                       # the effect follows the drawing into 3D
            self.fx_last = ("solid", solid)
            self.action_log.append("solid")
            self.notify.show(f"{s.result.shape_name}  ->  {s.result.solid_name}", s.rgb)
            self._log(f"{s.result.shape_name} -> {s.result.solid_name}")
            return True
        except Exception:
            self.notify.show("Couldn't convert that drawing", (230, 120, 80))
            return False

    def _pending_to_solids(self) -> int:
        n = 0
        for pts, col, wid, bk in self.pending:
            try:
                res = ShapeRecognizer.recognize(pts) or ShapeRecognizer.generic(pts)
                if res is None: continue
                sol = self.scene3d.add(build_solid(res, col, bk, wid, self.scene3d.view, 0.0), morph=True)
                sol.effect = self._pending_effect_key(pts)
                self.action_log.append("solid")
                n += 1
            except Exception:
                continue
        return n

    def _convert_all(self):
        """TO 3D button: every not-yet-launched drawing becomes a 3-D solid (stays put until LAUNCH)."""
        if self.cur_stroke: self._commit()
        n = self._pending_to_solids()
        self.pending.clear()
        self.undo.clear()
        self.canvas.fill((0, 0, 0, 0))
        self.action_log = [a for a in self.action_log if a != "ink"]
        self.suggest = None
        self.chip.hide()
        self.notify.show(f"Converted {n} drawing(s) to 3D" if n else "Draw something first, then TO 3D", (170, 130, 255))
        if n: self._log(f"Converted {n} to 3D")

    def _undo_any(self):
        if self.action_log and self.action_log[-1] == "ai":
            if self._ai_undo_step(): return
            self.action_log.pop()                          # stale marker (history was restored)
        if self.action_log and self.action_log[-1] == "solid":
            gone = self.scene3d.remove_last()
            if gone is not None:
                self.action_log.pop()
                self._redo.append(("solid", gone))
                self.suggest = None
                self.chip.hide()
                self.notify.show("Undone (3D)")
                self._log("Undo (3D)", clear_redo=False)
                return
            self.action_log.pop()
        after = pack_canvas(self.canvas)
        entry = self.pending[-1] if self.pending else None
        meta = None
        if entry is not None:
            meta = (self.pending_fx.get(id(entry[0])), self.pending_shape.get(id(entry[0])))
        ok = self.undo.pop(self.canvas)
        if ok and self.pending:
            self.pending.pop()                               # the undone stroke must not launch
        if ok and self.action_log and self.action_log[-1] == "ink":
            self.action_log.pop()
        if ok and entry is not None:
            self._redo.append(("ink", after, entry, meta))
        self.suggest = None
        self.chip.hide()
        self.notify.show("Undone" if ok else "Nothing to undo")
        if ok: self._log("Undo", clear_redo=False)

    def _redo_any(self):
        if not self._redo:
            self.notify.show("Nothing to redo"); return
        rec = self._redo.pop()
        if rec[0] == "ai":
            self._ai_redo_step(rec); return
        if rec[0] == "solid":
            if rec[1] not in self.scene3d.solids: self.scene3d.solids.append(rec[1])
            self.action_log.append("solid")
            self.notify.show("Redone (3D)")
        else:
            _, after, entry, meta = rec
            self.undo.push(self.canvas)                      # so Z can undo the redo again
            unpack_canvas(self.canvas, after)
            self.pending.append(entry)
            self.action_log.append("ink")
            if meta:
                fxs, shp = meta
                if fxs: self.pending_fx[id(entry[0])] = fxs
                if shp: self.pending_shape[id(entry[0])] = shp
            self.fx_last = ("pending", entry[0])
            self.notify.show("Redone")
        self.suggest = None
        self.chip.hide()
        self._log("Redo", clear_redo=False)

    def _orbit(self, tip):
        if self._orbit_prev is not None:
            self.scene3d.view.orbit((tip[0] - self._orbit_prev[0]) * ORBIT_GAIN,
                                    -(tip[1] - self._orbit_prev[1]) * ORBIT_GAIN)
        self._orbit_prev = tip

    def _hud_extra(self):
        ex = [(f"mode: {'3D' if self.mode3d else '2D'}  [X]", (170, 130, 255) if self.mode3d else (200, 200, 200))]
        if self.mode3d:
            ex.append((f"depth z: {self._z_now:+4.0f}  [0] reset", (150, 200, 255)))
        if self.two_hand: ex.append(("two-hand draw: ON  [W]", (120, 230, 255)))
        ex.append((f"auto-3D: {'ON' if self.auto3d else 'OFF'}  [N]", (120, 230, 160) if self.auto3d else (150, 150, 150)))
        n_fx = (sum(1 for e in self.pending_fx.values()) + sum(1 for b in self.phys.bodies if b.effect)
                + sum(1 for s in self.scene3d.solids if getattr(s, "effect", None)))
        if n_fx: ex.append((f"effects: {n_fx} active  [F]", (190, 130, 255)))
        return ex

    def _debug_extra(self):
        v = self.scene3d.view
        ex = [f"3D MODE: {'ON' if self.mode3d else 'off'}   solids: {len(self.scene3d.solids)}",
              f"view yaw {math.degrees(v.yaw):+.0f}  pitch {math.degrees(v.pitch):+.0f}",
              f"depth z {self._z_now:+.0f}   hand-size ratio x{self.depth.ratio:.2f}"]
        if self.tracker is not None:
            ex.append(f"tracking: {self.tracker.fps:.0f} fps, {self.tracker.proc_ms:.0f} ms/frame (own thread)")
        if self.grab3d.active: ex += self.grab3d.info
        return ex

    def _render_3d_ui(self):
        if not self.mode3d: return
        x, y0, y1 = WIN_W - 16, 330, 600
        pn = pygame.Surface((12, y1 - y0), pygame.SRCALPHA); pn.fill((8, 14, 30, 170))
        self.screen.blit(pn, (x - 2, y0))
        pygame.draw.rect(self.screen, (170, 130, 255), (x - 2, y0, 12, y1 - y0), 1, border_radius=4)
        f = (self._z_now + Z_RANGE) / (2 * Z_RANGE)
        my = int(y1 - f * (y1 - y0))
        pygame.draw.rect(self.screen, (255, 255, 255), (x - 6, my - 3, 20, 6), border_radius=3)
        font = self.chip._s
        self.screen.blit(font.render("NEAR", True, (190, 170, 255)), (x - 30, y0 - 14))
        self.screen.blit(font.render("FAR", True, (190, 170, 255)), (x - 22, y1 + 3))

    # ───────────── PHASE 4: particle effects / energy ring ─────────────
    def _pending_effect_key(self, pts) -> Optional[str]:
        sp = self.pending_fx.get(id(pts))
        return sp["key"] if sp and sp["pts"] is pts else None

    def _give_effect_to_body(self, pts):
        """A pending stroke was just launched as phys.bodies[-1]: hand over its shape + effect."""
        if not self.phys.bodies: return
        pb = self.phys.bodies[-1]
        shp = self.pending_shape.get(id(pts))
        if shp and shp[1] is pts: pb.shape = shp[0].kind
        sp = self.pending_fx.get(id(pts))
        if sp and sp["pts"] is pts:
            pb.effect, pb.ring = sp["key"], bool(sp["ring"])

    def _fx_target(self):
        """What an effect applies to: held > hovered > last drawn / grabbed."""
        for kind, obj in (("solid", self.grab3d.body), ("body", self.grab.body),
                          ("solid", self.grab3d.hover), ("body", self.grab.hover)):
            if obj is not None: return (kind, obj)
        t = self.fx_last
        if t is None: return None
        kind, obj = t
        if kind == "pending": return t if any(e[0] is obj for e in self.pending) else None
        if kind == "body":    return t if obj in self.phys.bodies else None
        if kind == "solid":   return t if obj in self.scene3d.solids else None
        return None

    def _fx_current(self) -> Optional[str]:
        t = self._fx_target()
        if not t: return None
        kind, obj = t
        if kind == "pending": return self._pending_effect_key(obj)
        return getattr(obj, "effect", None)

    def apply_effect(self, key: str) -> bool:
        """Attach (or with 'none', remove) a particle effect on the target.  A drawn circle becomes an energy ring."""
        if self.lyr.locked("effects"):
            self._warn_locked("Effects"); return False
        t = self._fx_target()
        if t is None:
            self.notify.show("Draw or grab something first, then pick an effect")
            return False
        kind, obj = t
        key = key if key in EFFECTS else None
        ring = False
        if kind == "pending":
            if key is None:
                self.pending_fx.pop(id(obj), None)
            else:
                shp = self.pending_shape.get(id(obj))
                ring = bool(shp and shp[1] is obj and shp[0].kind == "circle")
                geom = (shp[0].center[0], shp[0].center[1], shp[0].params["r"]) if ring else None
                cen = (sum(p[0] for p in obj) / len(obj), sum(p[1] for p in obj) / len(obj))
                self.pending_fx[id(obj)] = {"key": key, "pts": obj, "ring": ring, "geom": geom, "cen": cen}
        elif kind == "body":
            obj.effect = key
            obj.ring = bool(key and obj.shape == "circle")
            ring = obj.ring
        else:
            obj.effect = key
        if key is None:
            self.notify.show("Effect removed")
            self._log("Effect removed")
        else:
            self.notify.show(f"{EFFECTS[key].name}" + ("  ->  ENERGY RING" if ring else ""), EFFECTS[key].color)
            self._log(f"Effect: {EFFECTS[key].name}" + (" ring" if ring else ""))
        return True

    @staticmethod
    def _body_ring_geom(pb):
        P = [pb.body.local_to_world(sh.a) for sh in pb.shapes if isinstance(sh, pymunk.Segment)]
        if len(P) < 6: return None
        cx = sum(p.x for p in P) / len(P); cy = sum(p.y for p in P) / len(P)
        return cx, cy, sum(math.hypot(p.x - cx, p.y - cy) for p in P) / len(P)

    def _effects_update(self, dt: float):
        """Emit particles for every drawing / body / solid that carries an effect (capped, never unbounded)."""
        rng, budget = self._fx_rng, EFFECT_BUDGET
        live = {id(e[0]): e[0] for e in self.pending}
        for k in [k for k, v in self.pending_fx.items() if live.get(k) is not v["pts"]]: self.pending_fx.pop(k, None)
        for k in [k for k, v in self.pending_shape.items() if live.get(k) is not v[1]]: self.pending_shape.pop(k, None)
        def due(d: EffectDef, px: float) -> bool:
            return rng.random() < d.rate * min(4.0, max(0.4, px / 200.0)) * dt
        for pts, col, wid, bk in self.pending:
            sp = self.pending_fx.get(id(pts))
            if not sp or sp["pts"] is not pts or budget <= 0: continue
            d = EFFECTS[sp["key"]]
            if sp["ring"] and sp["geom"]:
                budget -= emit_ring_particles(self.fx, sp["key"], *sp["geom"], rng, dt)
            elif due(d, len(pts) * DENSIFY_STEP):
                p = pts[rng.randrange(len(pts))]
                if 0 <= p[0] < WIN_W and 0 <= p[1] < WIN_H and self.canvas.get_at(p)[3] > 40:   # skip erased ink
                    nx, ny = p[0] - sp["cen"][0], p[1] - sp["cen"][1]; L = math.hypot(nx, ny) or 1.0
                    d.emit(self.fx, p, (nx / L, ny / L), rng); budget -= 1
        for pb in self.phys.bodies:
            d = EFFECTS.get(pb.effect) if pb.effect else None
            if d is None or budget <= 0: continue
            if pb.ring:
                g = self._body_ring_geom(pb)
                if g: budget -= emit_ring_particles(self.fx, pb.effect, g[0], g[1], g[2], rng, dt)
            elif pb.shapes and due(d, len(pb.shapes) * DENSIFY_STEP):
                sh = pb.shapes[rng.randrange(len(pb.shapes))]
                if isinstance(sh, pymunk.Segment):
                    w = pb.body.local_to_world(sh.a); bx, by = pb.body.position
                    nx, ny = w.x - bx, w.y - by; L = math.hypot(nx, ny) or 1.0
                    d.emit(self.fx, (w.x, w.y), (nx / L, ny / L), rng); budget -= 1
        for s in self.scene3d.solids:
            d = EFFECTS.get(getattr(s, "effect", None) or "")
            if d is None or budget <= 0 or s._psx is None or len(s._psx) == 0: continue
            if due(d, s.radius * s.scale * 6.0):
                i = rng.randrange(len(s._psx)); cx, cy, _, _ = self.scene3d.center_screen(s)
                px, py = float(s._psx[i]), float(s._psy[i]); nx, ny = px - cx, py - cy; L = math.hypot(nx, ny) or 1.0
                d.emit(self.fx, (px, py), (nx / L, ny / L), rng); budget -= 1

    def _render_rings(self):
        t = time.perf_counter(); n = 0
        for pts, col, wid, bk in self.pending:
            sp = self.pending_fx.get(id(pts))
            if sp and sp["pts"] is pts and sp["ring"] and sp["geom"] and n < FX_RING_MAX:
                if draw_energy_ring(self.screen, *sp["geom"], sp["key"], t, (id(pts) % 97) / 10.0, self.lyr.opacity("effects")): n += 1
        for pb in self.phys.bodies:
            if pb.effect and pb.ring and n < FX_RING_MAX:
                g = self._body_ring_geom(pb)
                if g and draw_energy_ring(self.screen, *g, pb.effect, t, pb.uid * 0.7, self.lyr.opacity("effects")): n += 1

    # ───────────── PHASE 5: layers · edit · history ─────────────
    def _warn_locked(self, name: str):
        now = time.perf_counter()
        if now - self._lock_warn_t > 1.2:
            self._lock_warn_t = now
            self.notify.show(f"{name} layer is locked", (230, 140, 80))

    def _drawing_locked(self) -> bool:
        lid = "objects" if (self.mode3d and ENABLE_3D) else "drawing"
        if self.lyr.locked(lid):
            self._warn_locked(self.lyr[lid].name); return True
        return False

    def _grab_hands(self, hands):
        if not self.lyr.locked("objects"): return hands
        return [dataclasses.replace(h, gesture=("IDLE" if h.gesture == "GRAB" else h.gesture), grab_pt=None) for h in hands]

    # ---- layered drawing ----
    def _draw_background(self, frame):
        bg = self._bg_surface(frame)
        if not self.lyr.visible("background"):
            self.screen.fill((14, 16, 24)); return
        if self._themed_background(frame): return
        op = self.lyr.opacity("background")
        if op < 0.999:
            self.screen.fill((14, 16, 24)); bg.set_alpha(int(255 * op)); self.screen.blit(bg, (0, 0)); bg.set_alpha(255)
        else:
            self.screen.blit(bg, (0, 0))

    def _draw_drawing_layer(self):
        op = self.lyr.opacity("drawing")
        self.canvas.set_alpha(255 if op >= 0.999 else int(255 * op))     # 255, NEVER None: None strips the per-pixel alpha
        self.screen.blit(self.canvas, (0, 0))
        if self.stroke_ctx is not None:
            self.layers.render(self.screen, self.stroke_ctx.passes)
        self._lh_render()

    def _draw_objects_layer(self):
        op = self.lyr.opacity("objects")
        self.phys.alpha_mul = self.scene3d.alpha_mul = op
        self.world.render(self.screen)
        self.phys.render(self.screen, self.show_trails)
        self.grab.render(self.screen)
        self.scene3d.render(self.screen, budget_ms=max(2.5, min(SOLID_REBUILD_MS, 14.0 - (time.perf_counter() - self._t_frame0) * 1000.0)))

    def _draw_effects_layer(self):
        self.fx.alpha_mul = self.lyr.opacity("effects")
        self._render_rings()
        self.parts.render(self.screen)
        self.fx.render(self.screen)

    # ---- selection ----
    def edit_target(self):
        return self._fx_target()

    def _render_selection(self):
        if not (self.edit_panel.open or self.fx_panel.open or self.mode_name in ("SELECT", "EDIT")): return
        t = self.edit_target()
        if t is None: return
        kind, obj = t
        if kind == "body" and obj.visible:
            cx, cy, r = self.phys.object_bounds(obj)
        elif kind == "solid" and obj.visible:
            cx, cy, ck, _ = self.scene3d.center_screen(obj); r = obj.radius * obj.scale * ck + 12
        else:
            return
        _holo_ring(self.screen, cx, cy, r + 6, (255, 170, 70), "EDITING #%d" % obj.uid, time.perf_counter() * 1.5, self.scene3d._font)

    # ---- edit panel actions ----
    def _cycle_effect(self, arg: int):
        t = self.edit_target()
        if t is None: return
        kind, obj = t
        cur = self._pending_effect_key(obj) if kind == "pending" else getattr(obj, "effect", None)
        i = EFFECT_ORDER.index(cur) if cur in EFFECT_ORDER else 0
        self.apply_effect(EFFECT_ORDER[(i + arg) % len(EFFECT_ORDER)])

    def edit_apply(self, key: str, arg):
        t = self.edit_target()
        if t is None:
            self.notify.show("Nothing selected - pinch a drawing first"); return
        kind, obj = t
        if kind == "pending":
            if key == "effect": self._cycle_effect(arg)
            return
        if self.lyr.locked("objects"):
            self._warn_locked(self.lyr["objects"].name); return
        body = (kind == "body")
        label = None
        clamp = lambda v, lo, hi: max(lo, min(hi, v))
        if key == "color":
            rgb = PALETTE[arg]["rgb"]
            if body: obj.color = rgb
            else: obj.rgb = rgb
            label = f"Edit: color {PALETTE[arg]['name']}"
        elif key == "brush":
            i = BRUSH_ORDER.index(obj.brush) if obj.brush in BRUSH_ORDER else 0
            obj.brush = BRUSH_ORDER[(i + arg) % len(BRUSH_ORDER)]
            label = f"Edit: brush {BRUSHES[obj.brush].name}"
        elif key == "size":
            obj.width = int(clamp(obj.width + arg, EDIT_MIN_W, EDIT_MAX_W)); label = "Edit: size"
        elif key == "opacity":
            obj.opacity = round(clamp(obj.opacity + 0.1 * arg, 0.1, 1.0), 2); label = "Edit: opacity"
        elif key == "glow":
            obj.glow = round(clamp(obj.glow + 0.25 * arg, 0.0, 1.0), 2); label = "Edit: glow"
        elif key == "effect":
            self._cycle_effect(arg); return                  # apply_effect logs + notifies
        elif key == "freeze":
            if body: self.phys.set_frozen(obj, not obj.frozen)
            else:
                obj.frozen = not obj.frozen
                if obj.frozen: obj.vel[:] = 0; obj.ang[:] = 0; obj.active = False
                else: obj.active = True; obj.launch_t = 0.0
            label = "Edit: frozen" if obj.frozen else "Edit: unfrozen"
        elif key == "mass":
            if body: obj.mass_mult = round(clamp(obj.mass_mult * 1.25 ** arg, 0.25, 4.0), 3); self.phys.apply_props(obj)
            else: obj.mass = round(clamp(obj.mass * 1.25 ** arg, 0.25, 4.0), 3)
            label = "Edit: mass"
        elif key == "bounce":
            if body: obj.elast = round(clamp(obj.elast + 0.1 * arg, 0.0, 1.2), 2); self.phys.apply_props(obj)
            else: obj.bounce = round(clamp(obj.bounce + 0.1 * arg, 0.0, 1.2), 2)
            label = "Edit: bounce"
        elif key == "fric":
            obj.fric = round(clamp(obj.fric + 0.1 * arg, 0.0, 1.5), 2)
            if body: self.phys.apply_props(obj)
            label = "Edit: friction"
        elif key == "gscale":
            obj.gscale = round(clamp(obj.gscale + 0.25 * arg, 0.0, 2.0), 2)
            if body: self.phys.apply_props(obj)
            label = "Edit: gravity"
        elif key == "visible":
            obj.visible = not obj.visible; label = "Edit: hidden" if not obj.visible else "Edit: shown"
        elif key == "dup":
            self._edit_duplicate(kind, obj); label = "Duplicated"
        elif key == "delete":
            if body:
                if self.grab.body is obj: self.grab.reset()
                self.phys.remove_body(obj)
            else:
                if self.grab3d.body is obj: self.grab3d.reset()
                self.scene3d.remove(obj)
            self.fx_last = None; label = "Deleted"
        if body: obj.sprite = None                           # re-bake the look
        else: obj._spr = None
        if label: self._log(label, merge=True)

    def _edit_duplicate(self, kind, obj):
        if kind == "body":
            P = []
            for sh in obj.shapes:
                if isinstance(sh, pymunk.Segment):
                    a, b = obj.body.local_to_world(sh.a), obj.body.local_to_world(sh.b)
                    P += [(a.x + 40, a.y + 40), (b.x + 40, b.y + 40)]
            if not P: return
            pts = [P[0]] + [P[i] for i in range(1, len(P), 2)]
            self.phys.add_stroke(pts, obj.color, obj.width, obj.brush)
            nb = self.phys.bodies[-1]
            for f in ("effect", "ring", "shape", "opacity", "glow", "visible", "mass_mult", "elast", "fric", "gscale"):
                setattr(nb, f, getattr(obj, f))
            self.phys.apply_props(nb)
            if obj.frozen: self.phys.set_frozen(nb, True)
            self.fx_last = ("body", nb)
        else:
            d = solid_to_dict(obj); d["pos"] = d["pos"] + np.array([40.0, 40.0, 0.0])
            ns = dict_to_solid(d); ns.vel[:] = 0
            self.scene3d.add(ns, morph=False)
            self.fx_last = ("solid", ns)

    # ---- layer panel actions ----
    def layer_apply(self, key: str, arg):
        L = self.lyr; label = None
        if key == "select":
            L.selected = arg; return
        if key == "vis":
            L.toggle_visible(arg); label = f"Layer: {L[arg].name} {'shown' if L.visible(arg) else 'hidden'}"
        elif key == "lock":
            L.toggle_lock(arg); label = f"Layer: {L[arg].name} {'locked' if L.locked(arg) else 'unlocked'}"
        elif key == "op":
            L.step_opacity(arg[0], arg[1]); label = f"Layer: {L[arg[0]].name} opacity"
        elif key in ("up", "down"):
            if L.move(arg, +1 if key == "up" else -1): label = f"Layer: {L[arg].name} moved {key}"
        elif key == "clr":
            if self._clear_layer(arg): label = f"Layer: {L[arg].name} cleared"
        if label: self._log(label, merge=True)

    def _clear_layer(self, lid: str) -> bool:
        if self.lyr.locked(lid):
            self._warn_locked(self.lyr[lid].name); return False
        if lid == "drawing":
            if self.cur_stroke: self._commit()
            self.pending.clear(); self.undo.clear(); self.canvas.fill((0, 0, 0, 0))
            self.pending_fx.clear(); self.pending_shape.clear()
            self.action_log = [a for a in self.action_log if a != "ink"]
        elif lid == "objects":
            self.grab.reset(); self.grab3d.reset(); self.phys.clear(); self.scene3d.clear()
            self.action_log = [a for a in self.action_log if a != "solid"]
        elif lid == "effects":
            self.pending_fx.clear()
            for pb in self.phys.bodies: pb.effect, pb.ring = None, False
            for s in self.scene3d.solids: s.effect = None
            self.fx.clear()
        else:
            return False
        self.suggest = None; self.chip.hide(); self.fx_last = None
        self.notify.show(f"{self.lyr[lid].name} cleared")
        return True

    def _begin_rename(self):
        self.rename_id = self.lyr.selected
        self.rename_buf = self.lyr[self.rename_id].name
        self._rename_fresh = True                            # first typed character replaces the old name
        self.notify.show("Type a name, Enter = OK, Esc = cancel")

    def _rename_key(self, ev, k: str):
        if k == "return" or k == "enter":
            if self.lyr.rename(self.rename_id, self.rename_buf):
                self._log(f"Layer renamed: {self.rename_buf.strip()[:16]}")
            self.rename_id = None
        elif k == "escape":
            self.rename_id = None
        elif k == "backspace":
            self.rename_buf = self.rename_buf[:-1]; self._rename_fresh = False
        else:
            ch = getattr(ev, "unicode", "")
            if ch and ch.isprintable():
                if getattr(self, "_rename_fresh", False): self.rename_buf = ""; self._rename_fresh = False
                if len(self.rename_buf) < 16: self.rename_buf += ch

    # ---- history ----
    def _log(self, label: str, clear_redo: bool = True, merge: bool = False):
        if clear_redo: self._redo.clear()
        try:
            snap = self._snapshot()
        except Exception:
            snap = None
        now = time.perf_counter() - self.hist.t0
        last = self.hist.entries[-1] if self.hist.entries else None
        if merge and last is not None and last.label == label and now - last.t < 2.0:
            last.snap, last.t = snap, now                    # a burst of +/- presses = one history entry
        else:
            self.hist.add(label, snap)

    def _snapshot(self) -> dict:
        bodies = []
        for pb in self.phys.bodies:
            segs = [sh for sh in pb.shapes if isinstance(sh, pymunk.Segment)]
            if not segs: continue
            pts = [tuple(pb.body.local_to_world(segs[0].a))] + [tuple(pb.body.local_to_world(sh.b)) for sh in segs]
            d = dict(pts=pts, vel=tuple(pb.body.velocity), angvel=pb.body.angular_velocity)
            d.update({f: getattr(pb, f) for f in ("color", "width", "brush", "effect", "ring", "shape", "opacity", "glow",
                                                  "visible", "mass_mult", "elast", "fric", "gscale", "frozen")})
            bodies.append(d)
        pend = []
        for entry in self.pending:
            pts = entry[0]
            sp = self.pending_fx.get(id(pts)); shp = self.pending_shape.get(id(pts))
            pend.append((entry, {k: sp[k] for k in ("key", "ring", "geom", "cen")} if sp and sp["pts"] is pts else None,
                         shp[0] if shp and shp[1] is pts else None))
        return dict(canvas=pack_canvas(self.canvas), pending=pend, bodies=bodies,
                    solids=[solid_to_dict(s) for s in self.scene3d.solids if not s.building],
                    settings=dict(color_key=self.color_key, pen_w=self.pen_w, brush_key=self.brush_key, mode3d=self.mode3d,
                                  auto3d=self.auto3d, grav=self.phys.gravity_idx, trails=self.show_trails, wind=self._wind),
                    layers=self.lyr.snapshot(), action_log=list(self.action_log))

    def _restore(self, snap: dict):
        self._lh_reset()
        self.grab.reset(); self.grab3d.reset()
        self.layers.clear(); self.stroke_ctx = None; self.cur_stroke.clear(); self.stroke3d = None; self._stroke_z = []
        unpack_canvas(self.canvas, snap["canvas"])
        self.pending.clear(); self.pending_fx.clear(); self.pending_shape.clear()
        for entry, fxs, shp in snap["pending"]:
            pts = list(entry[0]); e = (pts,) + tuple(entry[1:])
            self.pending.append(e)
            if fxs: self.pending_fx[id(pts)] = dict(fxs, pts=pts)
            if shp is not None: self.pending_shape[id(pts)] = (shp, pts)
        self.phys.clear()
        for d in snap["bodies"]:
            self.phys.add_stroke(d["pts"], d["color"], d["width"], d["brush"])
            pb = self.phys.bodies[-1]
            for f in ("effect", "ring", "shape", "opacity", "glow", "visible", "mass_mult", "elast", "fric", "gscale"):
                setattr(pb, f, d[f])
            self.phys.apply_props(pb)
            pb.body.angular_velocity = d["angvel"]; pb.body.velocity = d["vel"]
            if d["frozen"]: self.phys.set_frozen(pb, True)
        self.scene3d.clear()
        for d in snap["solids"]: self.scene3d.solids.append(dict_to_solid(d))
        st = snap["settings"]
        self.color_key, self.pen_w, self.brush_key = st["color_key"], st["pen_w"], st["brush_key"]
        self.mode3d, self.auto3d, self.show_trails = st["mode3d"], st["auto3d"], st["trails"]
        self.phys._grav_idx = st["grav"]; self.phys.space.gravity = GRAVITY_PRESETS[st["grav"]]
        self._wind = st["wind"]; self.phys.set_wind(self._wind)
        self.lyr.restore(snap["layers"]); self.action_log = list(snap["action_log"])
        self.undo.clear(); self._redo.clear(); self.suggest = None; self.chip.hide(); self.fx_last = None; self.fx.clear()

    def time_apply(self, key: str, arg):
        H = self.hist
        if key == "undo": self._undo_any()
        elif key == "redo": self._redo_any()
        elif key == "older": H.scroll = min(H.scroll + 5, max(0, len(H.entries) - TimelinePanel.VISIBLE))
        elif key == "newer": H.scroll = max(0, H.scroll - 5)
        elif key == "goto":
            self._ai_undo.clear()
            e = H.get(arg)
            if e is None or e.snap is None: return
            self._restore(e.snap)
            self.notify.show(f"Restored: {e.label}", (120, 190, 255))
            self._log(f"Restored: {e.label}")

    def _ambient_fx(self, dt: float):
        """Pending (not yet launched) and launched strokes keep emitting their brush effect."""
        rng, budget = self._fx_rng, FX_AMBIENT_BUDGET
        def due(br: Brush, n_pts: int) -> bool:
            rate = br.amb_rate * (n_pts * DENSIFY_STEP / 200.0)
            return rng.random() < rate * dt
        for pts, col, wid, bk in self.pending[-8:]:
            br = BRUSHES.get(bk)
            if budget <= 0: break
            if br and br.emits and due(br, len(pts)):
                p = pts[rng.randrange(len(pts))]
                if 0 <= p[0] < WIN_W and 0 <= p[1] < WIN_H and self.canvas.get_at(p)[3] > 40:
                    br.emit(self.fx, p, col, wid, rng, 0.0); budget -= 1       # skips erased ink
        for pb in self.phys.bodies:
            br = BRUSHES.get(pb.brush)
            if budget <= 0: break
            if br and br.emits and pb.shapes and due(br, len(pb.shapes)):
                sh = pb.shapes[rng.randrange(len(pb.shapes))]
                if isinstance(sh, pymunk.Segment):
                    w = pb.body.local_to_world(sh.a)
                    br.emit(self.fx, (w.x, w.y), pb.color, pb.width, rng, 0.0); budget -= 1

    @staticmethod
    def _cursor_pt(hd: HandData):
        return hd.grab_pt if (hd.gesture == "GRAB" and hd.grab_pt) else hd.tip

    @staticmethod
    def _bgr_to_surf(bgr: np.ndarray) -> pygame.Surface:
        if bgr.shape[1] == WIN_W and bgr.shape[0] == WIN_H:       # camera == window: skip the transpose (measured ~2x faster)
            rgb = cv2.cvtColor(cv2.flip(bgr, 0), cv2.COLOR_BGR2RGB)   # vertical flip only (== np.flipud) - NOT a mirror
            if CAM_SHARPEN > 0:
                rgb = cv2.addWeighted(rgb, 1.0 + CAM_SHARPEN, cv2.GaussianBlur(rgb, (0, 0), 1.2), -CAM_SHARPEN, 0)
            return pygame.image.frombuffer(rgb.tobytes(), (WIN_W, WIN_H), "RGB").convert()
        rgb  = cv2.cvtColor(cv2.flip(bgr, 0), cv2.COLOR_BGR2RGB)   # vertical flip only - NOT a mirror
        if rgb.shape[1] != WIN_W or rgb.shape[0] != WIN_H:
            interp = cv2.INTER_AREA if rgb.shape[1] > WIN_W else CAM_UPSCALE   # smooth resize instead of blocky nearest-neighbour
            rgb = cv2.resize(rgb, (WIN_W, WIN_H), interpolation=interp)
        if CAM_SHARPEN > 0:
            rgb = cv2.addWeighted(rgb, 1.0 + CAM_SHARPEN, cv2.GaussianBlur(rgb, (0, 0), 1.2), -CAM_SHARPEN, 0)
        return pygame.image.frombuffer(rgb.tobytes(), (WIN_W, WIN_H), "RGB").convert()

    def _bg_surface(self, frame: np.ndarray) -> pygame.Surface:
        """Convert a camera frame once, however many render frames show it."""
        if frame is not self._bg_frame:
            self._bg_surf = self._bgr_to_surf(frame)
            self._bg_frame = frame
        return self._bg_surf

    def _pace(self):
        """Precise frame limiter: sleep most of the gap, spin the last 2 ms (OS sleep alone drifts +-1-15 ms)."""
        period = 1.0 / FPS_CAP
        now = time.perf_counter()
        if self._next_t < now - 2 * period:
            self._next_t = now                               # fell behind: don't try to catch up in a burst
        slack = self._next_t - now
        if slack > 0.002:
            time.sleep(slack - 0.002)
        while time.perf_counter() < self._next_t:
            pass
        self._next_t += period
        self.clock.tick()                                    # (uncapped) just keeps get_fps() honest

    def run(self):
        print("✅  Air Draw Physics v10  (two-hand, zero-mirror)")
        print("   Right hand: ☝ index-only DRAW  🤌 RESIZE  👊 ERASE  ✌ LAUNCH  🖐 radial menu")
        print("   Right-side switches: CLEANUP / 2D>3D / REAL WORLD collide / BOUNCE  (hold fingertip 0.5 s)")
        print("   Either hand: 🤏 thumb+middle pinch = GRAB / move / throw")
        print("   Left  hand: 👊 PUSH    🖐 BLAST      [D] gesture debug overlay")
        print("   3D: [X] 2D/3D mode  [V] turn last drawing into 3D  [N] auto-3D  [0] reset depth/view")
        print("       3D mode: hand toward camera = closer; left fist drag = orbit view; arrows = orbit/rotate")
        print("   Effects: [F] menu (or FX tab) -> Fire/Smoke/Sparks/Snow/Stars/Electricity/Plasma/Dust/Magic/Confetti/Bubbles")
        print("       applies to what you hold / hover / just drew; on a drawn CIRCLE it becomes an energy ring")
        print("   Phase 10: [W] two-hand drawing  Ctrl+/ quick guide  (cleaner UI, steadier buttons)")
        print("   Phase 9: [M] symmetry  [J] generative art  [R] record MP4  Ctrl+S save project  Ctrl+G game mode  (ART / PLAY / FILE tabs)")
        print("   Phase 8: [U] UI panel (themes/HUD/modes)  [`] HUD classic/system/off  Ctrl+T theme  Ctrl+1..9 mode")
        print("   Phase 7: [A] AI panel  [/] ask the assistant  [K] air calculator (K again = air math)")
        print("   Phase 5: [I] edit selected  [L] layers (Ctrl+R rename)  [H] history timeline  [Y]/Ctrl+Y redo  Ctrl+Z undo")
        print("   Hover finger over colour/toolbar 0.5 s to activate.")
        print("   Hold right palm 1 s for radial menu.\n")

        if not self.hist.entries: self._log("Start")
        self.notify.show("Ctrl+/ = quick guide   |   W = two-hand drawing", (150, 220, 255))
        prev_t = time.perf_counter()
        if self.tracker is not None:
            self.tracker.start()
        gc.collect()
        gc.freeze()                                          # long-lived objects are never re-scanned
        gc.set_threshold(100_000, 50, 100)                   # no collector pauses in the middle of a frame

        while True:
            now   = time.perf_counter()
            dt    = min(now - prev_t, 0.05)
            prev_t = now
            self._t_frame0 = now

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._quit()
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    bk = self.brush_panel.click(ev.pos)
                    if bk: self._select_brush(bk)
                    fxk = self.fx_panel.click(ev.pos)
                    if fxk: self.apply_effect(fxk)
                    if self.chip.click(ev.pos): self.convert_suggestion()
                    swk = self.switches.click(ev.pos)
                    if swk: self.switch_do(swk)
                elif ev.type == pygame.KEYDOWN:
                    k = pygame.key.name(ev.key)
                    if self.rename_id is not None:                      # typing a layer name: nothing else may fire
                        self._rename_key(ev, k); continue
                    if self._ai_key(ev, k): continue                    # typing into the ASK box: nothing else may fire
                    ctrl = bool(ev.mod & pygame.KMOD_CTRL)
                    if ctrl and k == "z": self._do_action("undo"); continue
                    if ctrl and k == "y": self._do_action("redo"); continue
                    if ctrl and len(k) == 1 and k in "123456789": self.set_mode(MODE_ORDER[int(k) - 1]); continue
                    if ctrl and k == "t": self.cycle_theme(); continue
                    if ctrl and k == "k": self.switch_do("cleanup"); continue
                    if ctrl and k == "w": self.switch_do("world"); continue
                    if ctrl and k == "s": self.save_project(); continue
                    if ctrl and k == "/": self.toggle_help(); continue
                    if ctrl and k == "g": self.set_mode("GAME") if self.mode_name != "GAME" else self.set_mode("DRAW"); continue
                    if ctrl and k == "r" and self.layer_panel.open: self._begin_rename(); continue
                    if k == "escape" and self.help_on: self.help_on = False
                    elif k in ("q", "escape"): self._quit()
                    elif k == "space": self._do_action("launch")
                    elif k == "c": self._do_action("clear")
                    elif k == "z": self._do_action("undo")
                    elif k == "g": self._do_action("gravity")
                    elif k == "t": self._do_action("trails")
                    elif k == "e": self._do_action("explode")
                    elif k == "s": self._do_action("screenshot")
                    elif k == "b": self.brush_panel.toggle()
                    elif k == "f": self.fx_panel.toggle()
                    elif k == "y": self._do_action("redo")
                    elif k == "i": self.edit_panel.toggle()
                    elif k == "l": self.layer_panel.toggle()
                    elif k == "h": self.time_panel.toggle()
                    elif k == "a": self.ai_panel.toggle()
                    elif k == "u": self.ui_panel.toggle()
                    elif k == "m": self.cycle_symmetry()
                    elif k == "w": self.toggle_two_hand()
                    elif k == "j": self.toggle_gen()
                    elif k == "r": self.toggle_record("mp4")
                    elif k == "`": self.cycle_hud()
                    elif k == "/": self.ai_apply("ask")
                    elif k == "k": self._calc_set_mode({None: "calc", "calc": "math", "math": "math"}[self.calc_mode])
                    elif k == "x": self._toggle_mode()
                    elif k == "v":
                        if not self.convert_suggestion():
                            self.notify.show("Nothing to convert - draw a shape first")
                    elif k == "n":
                        self.auto3d = not self.auto3d
                        self.notify.show(f"Auto 3D {'ON' if self.auto3d else 'OFF'}", (170, 130, 255))
                    elif k == "0":
                        self.depth.reset(); self.scene3d.view.reset()
                        self.notify.show("Depth + view reset")
                    elif k == ",": self.switch_do("bounce-")
                    elif k == ".": self.switch_do("bounce+")
                    elif k == "tab": self._cycle_brush(-1 if ev.mod & pygame.KMOD_SHIFT else 1)
                    elif k.startswith("f") and k[1:].isdigit() and 1 <= int(k[1:]) <= len(BRUSH_ORDER):
                        self._select_brush(BRUSH_ORDER[int(k[1:]) - 1])
                    elif k == "d":
                        self.debug = not self.debug
                        self.notify.show(f"Debug {'ON' if self.debug else 'OFF'}")
                    elif k in PALETTE:
                        self.color_key = k
                        self.notify.show(f"🎨 {PALETTE[k]['name']}", PALETTE[k]['rgb'])
                        self._log(f"Color: {PALETTE[k]['name']}")
                    elif k in ("+", "="):
                        self.pen_w = min(self.pen_w + 1, MAX_PEN_W)
                        self.notify.show(f"✏ Pen: {self.pen_w}px")
                    elif k == "-":
                        self.pen_w = max(self.pen_w - 1, MIN_PEN_W)
                        self.notify.show(f"✏ Pen: {self.pen_w}px")

            if self.tracker is not None:
                res = self.tracker.latest()
                if res is None:
                    self._pace()
                    continue
                _seq, frame, hands = res
            else:
                frame = self.cam.read()
                if frame is None:
                    self._pace()
                    continue
                rgb_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands: List[HandData] = self.clf.process(rgb_raw)
                for hd in hands:
                    self.clf.draw_skeleton(frame, hd.landmarks, hd.label)
            self._dt_s += 0.15 * (dt - self._dt_s)
            mdt = self._dt_s                                 # smoothed step for motion: a slow frame can't make things jump

            rh   = next((h for h in hands if h.label == "Right"), None)
            lh   = next((h for h in hands if h.label == "Left"),  None)
            rtip = rh.tip if rh else None
            ltip = lh.tip if lh else None
            rg   = rh.gesture if rh else "IDLE"
            lg   = lh.gesture if lh else "IDLE"
            self._ui_track(dt, rtip, ltip)
            self._select_update(dt, rtip, rg)

            ui_tip = (rtip if rg != "GRAB" else None) or (ltip if lg != "GRAB" else None)
            ui_tip = self._ui_smooth(ui_tip, dt)
            cf = self.palette.update(ui_tip, dt)
            if cf:
                self.color_key = cf
                self.notify.show(f"{PALETTE[cf]['name']}", PALETTE[cf]['rgb'])
                self._log(f"Color: {PALETTE[cf]['name']}")

            bf = self.brush_panel.update(ui_tip, dt)
            if bf: self._select_brush(bf)
            fxf = self.fx_panel.update(ui_tip, dt)
            if fxf: self.apply_effect(fxf)
            ef = self.edit_panel.update(ui_tip, dt)
            if ef: self.edit_apply(*ef)
            lf = self.layer_panel.update(ui_tip, dt)
            if lf: self.layer_apply(*lf)
            tf = self.time_panel.update(ui_tip, dt)
            if tf: self.time_apply(*tf)
            for _p, _h in self._ext:
                _f = _p.update(ui_tip, dt)
                if _f: _h(*_f)
            pnls = (self.brush_panel, self.fx_panel, self.edit_panel, self.layer_panel, self.time_panel) + tuple(p for p, _ in self._ext)
            opens = tuple(bool(p.open) for p in pnls)                       # only one menu open at a time
            prev_open = self._panels_open if len(self._panels_open) == len(pnls) else (False,) * len(pnls)
            if sum(opens) > 1:
                new = [i for i, o in enumerate(opens) if o and not prev_open[i]]
                keep = new[0] if new else max(i for i, o in enumerate(opens) if o)
                for i, p in enumerate(pnls): p.open = (i == keep)
                opens = tuple(i == keep for i in range(len(pnls)))
            self._panels_open = opens
            self.switches.visible = self._switches_visible()
            swf = self.switches.update(ui_tip, dt)
            if swf: self.switch_do(swf)
            self._world_update(frame, hands)
            if self.chip.update(ui_tip, dt):
                self.convert_suggestion()
                if rg == "DRAW" and rtip is not None:
                    self._ink_lock = rtip                    # the pointing finger that just clicked must not leave a stray dot
            if self._ink_lock is not None and (rg != "DRAW" or rtip is None or
                    math.hypot(rtip[0] - self._ink_lock[0], rtip[1] - self._ink_lock[1]) > 60):
                self._ink_lock = None
            ui_block = (self.brush_panel.blocks(rtip) or self.fx_panel.blocks(rtip) or self.edit_panel.blocks(rtip)
                        or self.layer_panel.blocks(rtip) or self.time_panel.blocks(rtip) or self.chip.blocks(rtip)
                        or any(p.blocks(rtip) for p, _ in self._ext) or self._mode_blocks_ink() or self.toolbar.blocks(rtip) or self.palette.blocks(rtip) or self.switches.blocks(rtip)
                        or self._ink_lock is not None
                        or (self.grab3d.active and self.grab3d.hand != "Right"))   # left hand holds a 3D object: right finger spins it, not draws
            self.toolbar.mode_label = f"MODE {'3D' if self.mode3d else '2D'}"
            self.toolbar.cur_mode = self.mode_name

            tf = self.toolbar.update(ui_tip, dt)
            if tf:
                self._do_action(tf)
            if self.clf.radial_ready("Right") and not self.radial.visible:
                cx, cy = rtip if rtip else (WIN_W//2, WIN_H//2)
                self.radial.open(cx, cy)
                self.clf.reset_radial("Right")
            if self.radial.visible and rg != "PALM":
                self.radial.close()
            rf = self.radial.update(rtip, dt)
            if rf:
                self._do_action(rf)
                self.radial.close()
            if (rg == "LAUNCH"
                    and self.clf.launch_ready("Right")
                    and not self.radial.visible):
                self._do_action("launch")
                self.clf.reset_launch("Right")

            if rh is not None and self.mode3d and ENABLE_3D:
                self._z_now = self.depth.update(rh.size, dt)
            elif not self.mode3d:
                self._z_now = 0.0
            self._dgap = 0.0 if (rg == "DRAW" and rtip is not None) else getattr(self, "_dgap", 0.0) + dt
            if rg == "DRAW" and rtip is not None and not ui_block and not self._drawing_locked():
                sx, sy = self.ema["Right"].update(float(rtip[0]), float(rtip[1]))
                sx, sy = int(sx), int(sy)

                if not self.cur_stroke:                      # start of a new stroke
                    self.ema["Right"].reset()
                    sx, sy = self.ema["Right"].update(float(rtip[0]), float(rtip[1]))
                    sx, sy = int(sx), int(sy)
                    self.cur_stroke = [(sx, sy)]
                    self._begin_any((sx, sy))
                else:
                    if self.cur_stroke:
                        lx, ly = self.cur_stroke[-1]
                        dist = math.hypot(sx - lx, sy - ly)
                        if dist >= MIN_MOVE_PX:
                            for fp in self._densify((lx, ly), (sx, sy)):
                                self._paint_any(self.cur_stroke[-1], fp)
                                self.cur_stroke.append(fp)

                if self.stroke_ctx is not None:               # keep flames burning while you hold still
                    sbr = BRUSHES[self.stroke_ctx.brush_key]
                    if sbr.emits:
                        sbr.emit(self.fx, (sx, sy), self.stroke_ctx.rgb, self.stroke_ctx.w, self._fx_rng, 0.0)
                ccx = int(rtip[0] * CAM_W / WIN_W)
                ccy = int(rtip[1] * CAM_H / WIN_H)
                bgr = (self.rgb[2], self.rgb[1], self.rgb[0])
                cv2.circle(frame, (ccx, ccy), CURSOR_OUTER, bgr, 2)
                cv2.circle(frame, (ccx, ccy), CURSOR_INNER, bgr, -1)


            elif rg == "PINCH" and rh and rh.feats is not None:
                t_ = max(0.0, min(1.0,
                    (rh.feats.ti - RESIZE_MIN_D) / (RESIZE_MAX_D - RESIZE_MIN_D)))
                new_w = int(MIN_PEN_W + t_ * (MAX_PEN_W - MIN_PEN_W))
                if new_w != self.pen_w:
                    self.pen_w = new_w
                if self.cur_stroke: self._commit()

            elif rg == "ERASE" and rtip is not None:
                if self.cur_stroke: self._commit()
                self._erase(*rtip)
                ecx = int(rtip[0] * CAM_W / WIN_W)
                ecy = int(rtip[1] * CAM_H / WIN_H)
                cv2.circle(frame, (ecx, ecy),
                           int(ERASE_RADIUS * CAM_W / WIN_W), (80, 80, 230), 2)

            else:
                near = (rtip is None or not self.cur_stroke or math.hypot(rtip[0] - self.cur_stroke[-1][0], rtip[1] - self.cur_stroke[-1][1]) < 140)
                flicker = (not ui_block and self._dgap < DRAW_GRACE_S and near and (rtip is None or rg in ("LAUNCH", "PALM")))
                if self.cur_stroke and not flicker and (self.prev_g["Right"] == "DRAW" or self._dgap >= DRAW_GRACE_S):
                    self._commit()

            self._lh_update(lg, ltip, dt)
            # THUMB+MIDDLE GRAB (either hand) -- centralised
            if rg == "GRAB" or lg == "GRAB":
                if self.cur_stroke: self._commit()
            keys = pygame.key.get_pressed()
            self.grab.key_scale = int(keys[pygame.K_RIGHTBRACKET]) - int(keys[pygame.K_LEFTBRACKET])
            self.grab.key_force = 1 if keys[pygame.K_p] else (-1 if keys[pygame.K_o] else 0)
            self.grab3d.key_scale = self.grab.key_scale
            self.grab3d.key_yaw = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
            self.grab3d.key_pitch = int(keys[pygame.K_DOWN]) - int(keys[pygame.K_UP])
            self.grab3d.key_depth = int(keys[pygame.K_PAGEUP]) - int(keys[pygame.K_PAGEDOWN])
            if DEPTH_METHOD == "keyboard" and self.mode3d:
                self.depth.nudge(300.0 * dt * self.grab3d.key_depth)
            if not self.grab3d.active and (self.grab3d.key_yaw or self.grab3d.key_pitch):
                self.scene3d.view.orbit(self.grab3d.key_yaw * 1.6 * dt, -self.grab3d.key_pitch * 1.6 * dt)
            gh = self._grab_hands(hands)
            ev3 = self.grab3d.update(gh, dt=dt)
            if ev3 == "grab":
                self.notify.show("GRAB 3D", (170, 130, 255))
            elif ev3 == "catch":
                self.notify.show("CAUGHT", (170, 130, 255))
            elif ev3 == "handoff":
                self.notify.show(f"Passed to {self.grab3d.hand} hand", (170, 130, 255))
            elif ev3 == "release":
                self.notify.show("Released 3D", (200, 200, 200))
                self._log("Moved 3D object")
            if ev3 in ("grab", "catch", "handoff") and self.grab3d.body is not None:
                self.fx_last = ("solid", self.grab3d.body)
            ev = self.grab.update(self.grab3d.mask_hands(gh), dt=dt)
            if ev == "grab":
                if self.grab.body is not None: self.fx_last = ("body", self.grab.body)
                self.notify.show("GRAB", (90, 220, 255))
            elif ev == "release":
                self.notify.show("Released", (200, 200, 200))
                self._log("Moved object")

            # LEFT-HAND PUSH / BLAST (never while the left hand is the grabber)
            if lg == "PUSH" and ltip is not None:
                if self.mode3d and ENABLE_3D:
                    self._orbit(ltip)                       # 3D mode: left fist drag = orbit the view
                elif self.prev_g["Left"] != "PUSH":
                    self.phys.push_near(ltip)
            elif lg == "BLAST" and ltip is not None:
                if self.prev_g["Left"] != "BLAST":
                    self.phys.explode(float(ltip[0]), float(ltip[1]))
                    self.scene3d.explode(float(ltip[0]), float(ltip[1]))
                    self.parts.burst(float(ltip[0]), float(ltip[1]),
                                     (255, 200, 50), P_EXPLODE)
                    self.notify.show("Hand Blast!")

            if lg != "PUSH": self._orbit_prev = None
            self.prev_g["Right"] = rg
            self.prev_g["Left"]  = lg

            if self.phys.bodies:
                self.phys.step(mdt)
            self.scene3d.view.update(mdt)
            self.scene3d.step(mdt)
            self.scene3d.emit_fx(self.fx, self._fx_rng, mdt)
            self._effects_update(mdt)
            self.parts.update(mdt)
            self._ambient_fx(mdt)
            self.fx.update(mdt)
            self.notify.update(dt)
            self._ai_update(dt)
            self._art_update(dt)
            for hd in hands:
                col = self._hand_col(hd.label)
                self.cursor.update(hd.label, self._cursor_pt(hd), col)
            self._last_frame = frame
            self._draw_background(frame)
            if self.mode3d and ENABLE_3D and GRID3D:
                self.scene3d.render_grid(self.screen)
            for lid in self.lyr.order:                       # content layers, bottom -> top (re-orderable)
                if not self.lyr.visible(lid): continue
                if   lid == "drawing": self._draw_drawing_layer()
                elif lid == "objects": self._draw_objects_layer()
                else:                  self._draw_effects_layer()
            self._render_selection()
            self._art_live_render(self.screen)
            self.radial.render(self.screen)
            self.toolbar.render(self.screen)
            self.switches.render(self.screen, self._switch_state())
            self.palette.render(self.screen, self.color_key)
            self.brush_panel.render(self.screen, self.brush_key, self.rgb)
            self.fx_panel.render(self.screen, self._fx_current())
            self.edit_panel.render(self.screen, self)
            self.layer_panel.render(self.screen, self)
            self.time_panel.render(self.screen, self)
            for _p, _h in self._ext: _p.render(self.screen, self)
            self.chip.render(self.screen)
            self._ai_render(self.screen)
            self._help_render(self.screen)
            self._render_3d_ui()

            for hd in hands:
                col = self._hand_col(hd.label)
                self.cursor.render(self.screen, hd.label, self._cursor_pt(hd), col, hd.gesture)
            self._theme_hand_glow(hands)

            self.fps = self.clock.get_fps()
            if self.lyr.visible("ui") and self.hud_mode == "system":
                draw_hud_system(self.screen, self, hands)
            if self.lyr.visible("ui") and self.hud_mode == "classic":
                self.hud.render(
                    self.screen, hands, self.color_key, self.pen_w,
                    self.fps, len(self.phys.bodies) + len(self.scene3d.solids),
                    self.phys.gravity_idx, self.show_trails, self._wind,
                    extra=self._hud_extra(),
                )
            if self.debug and self.lyr.visible("ui"):
                self.debug_overlay.render(self.screen, hands, self.grab, self.fps, extra=self._debug_extra())
            self.notify.render(self.screen)

            self._rec_capture()
            pygame.display.flip()
            self._pace()

    def _quit(self):
        print("\n🛑  Shutting down…")
        self._art_shutdown()
        if self.tracker is not None:
            self.tracker.stop()
        self.cam.stop()
        self.clf.close()
        pygame.quit()
        sys.exit(0)
if __name__ == "__main__":
    App().run()
