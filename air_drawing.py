
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  AIR DRAW PHYSICS  v10  ─  Two-Hand · Zero-Mirror · Velocity-Throw               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  INSTALL:  pip install opencv-python mediapipe pymunk pygame numpy               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  RIGHT HAND  (drawing)                                                           ║
║   ☝  Index up, middle DOWN    → DRAW  (ink locked to fingertip, no mirror)     ║
║   🤌  Pinch index+thumb       → RESIZE (pen width in real-time)                 ║
║   👊  Fist                    → ERASE (circular rubber eraser)                 ║
║   🖐  Open palm (hold 0.3 s)  → LAUNCH (drawings fall with physics)            ║
║                                                                                ║
║  LEFT HAND  (physics sandbox)                                                  ║
║   ✌  Index+middle up          → GRAB  (pick up, move, & throw objects)        ║
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
║   E  explode   S  screenshot   +/-  pen   1-8  color   Q/ESC  quit           ║
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
import datetime
import math
import os
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
import pygame
import pymunk


WEBCAM_INDEX = 1
CAM_W,  CAM_H = 1280, 720
WIN_W,  WIN_H = 1280, 720
FPS_CAP = 60
DETECT_CONF = 0.55
TRACK_CONF  = 0.45
MAX_HANDS = 2
EMA_FAST = 0.55
EMA_SLOW = 0.30
MIN_MOVE_PX = 2
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

LAUNCH_HOLD_S = 0.28
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

class WebcamStream:
    def __init__(self, idx: int, w: int, h: int):
        self.cap = cv2.VideoCapture("http://10.42.103.71:4747/video")
        if not self.cap.isOpened():
            sys.exit(f"\n❌  Camera {idx} not found.  Edit WEBCAM_INDEX.\n")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = f

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

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
        #Stage 1
        if self._x1 is None:
            self._x1, self._y1 = float(x), float(y)
        else:
            self._x1 = self._af * x + (1 - self._af) * self._x1
            self._y1 = self._af * y + (1 - self._af) * self._y1
        #Stage 2 over stage-1 output
        if self._x2 is None:
            self._x2, self._y2 = self._x1, self._y1
        else:
            self._x2 = self._as * self._x1 + (1 - self._as) * self._x2
            self._y2 = self._as * self._y1 + (1 - self._as) * self._y2
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
class HandData:
    label: str
    gesture: str
    tip: Optional[Tuple[int, int]]
    mid: Optional[Tuple[int, int]]
    thumb: Optional[Tuple[int, int]]
    pinch_d: Optional[float]
    landmarks: object 


class GestureClassifier:
    _F = [(8, 6), (12, 10), (16, 14), (20, 18)]

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
        self._launch_t: Dict[str, float] = {"Right": 0.0, "Left": 0.0}
        self._radial_t: Dict[str, float] = {"Right": 0.0, "Left": 0.0}

    @staticmethod
    def _up(lm, tip: int, pip: int, thresh: float = 0.03) -> bool:
        """True when fingertip is clearly above (smaller y) its PIP knuckle."""
        return (lm[pip].y - lm[tip].y) > thresh

    def _classify_right(self, lm, pinch_d: float, t: float) -> str:
        i = self._up(lm, *self._F[0])
        m = self._up(lm, *self._F[1])
        r = self._up(lm, *self._F[2])
        p = self._up(lm, *self._F[3])
        
        if i and m and r and p:
            if not self._launch_t["Right"]: self._launch_t["Right"] = t
            if not self._radial_t["Right"]: self._radial_t["Right"] = t
            print(f"i={i}, m={m}, r={r}, p={p}")
            return "LAUNCH"
        self._launch_t["Right"] = self._radial_t["Right"] = 0.0
        if not i and not m and not r and not p:            
            return "ERASE"
        if pinch_d < PINCH_OPEN and not m and not r and not p: 
            return "PINCH"
        if i and not m:                                    
            return "DRAW"
        return "IDLE"
    

    def _classify_left(self, lm, t: float) -> str:
        i = self._up(lm, *self._F[0])
        m = self._up(lm, *self._F[1])
        r = self._up(lm, *self._F[2])
        p = self._up(lm, *self._F[3])
        if i and m and r and p:
            if not self._radial_t["Left"]: self._radial_t["Left"] = t
            return "BLAST"
        self._radial_t["Left"] = 0.0
        if i and not m:
            return "DRAW"
        if i and m and not r and not p:
            return "GRAB"
        if not i and not m and not r and not p:
            return "PUSH"
        return "IDLE"
    

    def process(self, rgb_frame: np.ndarray) -> List[HandData]:
        result = self._hands.process(rgb_frame)
        out: List[HandData] = []

        if not result.multi_hand_landmarks:
            self._launch_t = {"Right": 0.0, "Left": 0.0}
            self._radial_t = {"Right": 0.0, "Left": 0.0}
            return out

        t = time.perf_counter()

        for lm_obj, handedness in zip( 
            result.multi_hand_landmarks,
            result.multi_handedness,
        ):
            lm    = lm_obj.landmark
            label = handedness.classification[0].label
            if label == "Left":
                label = "Right"
            elif label == "Right":
                label = "Left"

            def px(i: int) -> Tuple[int, int]:
                return (        int(lm[i].x * WIN_W),
        int((1.0 - lm[i].y) * WIN_H))

            tip   = px(8)
            mid   = px(12)
            thumb = px(4)

            pinch_d = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)

            if label == "Right":
                gesture = self._classify_right(lm, pinch_d, t)
            else:
                gesture = self._classify_left(lm, t)

            out.append(HandData(label, gesture, tip, mid, thumb, pinch_d, lm_obj))

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
        self._sel: Optional[PhysBody] = None
        self._build_walls()

    def _build_walls(self):
        sb = self.space.static_body
        for a, b in [
            ((0,      self.H), (self.W, self.H)),
            ((0,      0     ), (0,      self.H)),
            ((self.W, 0     ), (self.W, self.H)),
            ((0,      0     ), (self.W, 0     )),
        ]:
            seg = pymunk.Segment(sb, a, b, WALL_W)
            seg.elasticity = ELASTICITY
            seg.friction   = FRICTION
            self.space.add(seg)
    def add_stroke(
        self,
        pts:   List[Tuple[int, int]],
        color: Tuple[int, int, int],
        width: int,
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
        self.bodies.append(PhysBody(body, shapes, color, width))
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
        tx, ty  = tip
        best, bd2 = None, radius ** 2
        for pb in self.bodies:
            bx, by = pb.body.position
            d2 = (bx - tx) ** 2 + (by - ty) ** 2
            if d2 < bd2:
                best, bd2 = pb, d2
        return best

    def start_drag(self, tip: Tuple[int, int]):
        pb = self.find_nearest(tip)
        if pb is None: return
        pb.selected = True
        self._sel = pb
        mb = pymunk.Body(body_type=pymunk.Body.STATIC)
        mb.position = tip
        j = pymunk.PivotJoint(mb, pb.body, tip)
        j.max_force  = DRAG_MAX_FORCE
        j.error_bias = DRAG_ERR_BIAS
        self.space.add(mb, j)
        self._mb = mb
        self._dj = j

    def update_drag(self, tip: Tuple[int, int]):
        if self._mb:
            self._mb.position = tip

    def end_drag(self, throw_vel: Tuple[float, float] = (0.0, 0.0)):
        """Release the grabbed body.
           If throw_vel magnitude > 50 px/s, apply it directly to the body so
           the throw feels natural and physically accurate."""
        if self._dj:
            self.space.remove(self._dj, self._mb)
        if self._sel:
            vx, vy = throw_vel
            if abs(vx) + abs(vy) > 50:
                self._sel.body.velocity = (vx, vy)
            self._sel.selected = False
        self._mb = self._dj = self._sel = None

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
        if self._dj:
            self.space.remove(self._dj, self._mb)
        self._mb = self._dj = self._sel = None

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
            self.add_stroke(pts, old.color, old.width)

    #step
    def step(self, dt: float):
        if self._wind != 0.0:
            for pb in self.bodies:
                if not pb.frozen:
                    pb.body.apply_force_at_local_point(
                        (self._wind * BODY_MASS, 0.0), (0.0, 0.0))
        sub = dt / PHYS_STEPS
        for _ in range(PHYS_STEPS):
            self.space.step(sub)
        for pb in self.bodies:
            bx, by = pb.body.position
            pb.trail.append((int(bx), int(by)))

    def clear(self):
        self.end_drag()
        for pb in self.bodies:
            self.space.remove(pb.body, *pb.shapes)
        self.bodies.clear()

    def render(self, surface: pygame.Surface, trails: bool):
        for pb in self.bodies:
            # Optional fading centroid trail
            if trails and len(pb.trail) > 1:
                pts = list(pb.trail)
                tmp = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
                for i in range(1, len(pts)):
                    a = int(TRAIL_ALPHA * i / len(pts))
                    pygame.draw.line(tmp, (*pb.color, a),
                                     pts[i-1], pts[i], max(pb.width-1, 1))
                surface.blit(tmp, (0, 0))

            lw = max(pb.width, 2)
            for sh in pb.shapes:
                if not isinstance(sh, pymunk.Segment): continue
                aw = pb.body.local_to_world(sh.a)
                bw = pb.body.local_to_world(sh.b)
                ax, ay = int(aw.x), int(aw.y)
                bx, by = int(bw.x), int(bw.y)
                if pb.selected:
                    pygame.draw.line(surface, (255, 255, 80),
                                     (ax, ay), (bx, by), lw*2 + 8)
                elif pb.frozen:
                    pygame.draw.line(surface, (80, 140, 255),
                                     (ax, ay), (bx, by), lw*2 + 6)
                pygame.draw.line(surface, pb.color, (ax, ay), (bx, by), lw*2)
                pygame.draw.circle(surface, pb.color, (ax, ay), lw)
                pygame.draw.circle(surface, pb.color, (bx, by), lw)

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
    H = 48; PAD = 10; BW = 110
    BTNS = [
        TBtn("launch",     "LAUNCH",  (230, 180,  50)),
        TBtn("clear",      "CLEAR",   (220,  80,  80)),
        TBtn("undo",       "UNDO",     (100, 180, 230)),
        TBtn("gravity",    "GRAVITY", (130, 100, 230)),
        TBtn("trails",     "TRAILS",  ( 80, 210, 130)),
        TBtn("wind_l",     "WIND",    ( 40, 200, 200)),
        TBtn("wind_r",     "WIND",    ( 40, 200, 200)),
        TBtn("screenshot", "SAVE",   (200, 200,  80)),
        TBtn("explode",    "BLAST",  (230, 120,  50)),
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
            lb = self._font.render(b.label, True, (255, 255, 255))
            surface.blit(lb, (r.x + (r.w - lb.get_width())//2,
                               r.y + (r.h - lb.get_height())//2))

RADIAL_OPT = [
    {"key": "launch",     "icon": "🚀", "label": "Launch"},
    {"key": "clear",      "icon": "🗑", "label": "Clear"},
    {"key": "gravity",    "icon": "🌍", "label": "Gravity"},
    {"key": "screenshot", "icon": "📸", "label": "Save"},
    {"key": "explode",    "icon": "💥", "label": "Explode"},
    {"key": "trails",     "icon": "✨", "label": "Trails"},
]

class RadialMenu:
    RAD = 130; BTN_R = 38; DUR = 0.25

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

class HUD:
    def __init__(self):
        pygame.font.init()
        self._big   = pygame.font.SysFont("monospace", 26, bold=True)
        self._med   = pygame.font.SysFont("monospace", 18, bold=True)
        self._small = pygame.font.SysFont("monospace", 15)

    @staticmethod
    def _panel(surface, x, y, w, h, col=(0,0,0), alpha=155):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill((*col, alpha))
        surface.blit(s, (x, y))

    def render(self, surface: pygame.Surface,
               hands:       List[HandData],
               color_key:   str,
               pen_w:       int,
               fps:         float,
               n_bodies:    int,
               grav_idx:    int,
               show_trails: bool,
               wind:        float):

        W, H = surface.get_size()

        MODES = {
            "DRAW":   ("✏  DRAW",    (80,  225,  80)),
            "LAUNCH": ("🌍 LAUNCH",  (230, 185,  50)),
            "GRAB":   ("✌  GRAB",   ( 80, 180, 230)),
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

        self._panel(surface, 0, H-28, W, 28)
        bar = (f" FPS:{fps:4.0f}  Bodies:{n_bodies}"
               "   SPACE launch  C clear  Z undo  E explode  G gravity  S save  Q quit")
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
            "PUSH":   (230, 140,  60),
            "BLAST":  (230,  80,  50),
            "IDLE":   (150, 150, 150),
        }.get(gesture, (200, 200, 200))

        pygame.draw.circle(surface, ring_col, tip, CURSOR_OUTER, 2)
        pygame.draw.circle(surface, ring_col, tip, CURSOR_INNER)

class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.NOFRAME)
        pygame.display.set_caption("Air Draw Physics v10")
        self.clock  = pygame.time.Clock()

        self.cam = WebcamStream(WEBCAM_INDEX, CAM_W, CAM_H)
        self.clf = GestureClassifier()
        self.phys = PhysicsWorld(WIN_W, WIN_H)
        self.parts = ParticleSystem()
        self.hud = HUD()
        self.notify = Notifier()
        self.palette = ColorPalette()
        self.toolbar = Toolbar()
        self.radial = RadialMenu()
        self.cursor = GlowCursor()
        self.undo = UndoStack()
        self.ema: Dict[str, DualEMA] = {"Right": DualEMA(), "Left": DualEMA()}
        self.vel: Dict[str, VelocityTracker] = {"Right": VelocityTracker(),"Left":  VelocityTracker()}
        self.canvas = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        self.canvas.fill((0, 0, 0, 0))

        self.color_key: str = DEFAULT_COLOR_KEY
        self.pen_w: int = DEFAULT_PEN_W
        self.cur_stroke: List[Tuple[int,int]] = []
        self.pending: List[Tuple[List, Tuple, int]] = []
        self.show_trails = True
        self._dragging = False
        self._wind = 0.0
        self.prev_g: Dict[str, str] = {"Right": "IDLE", "Left": "IDLE"}
        self.fps: float = 0.0
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    @property
    def rgb(self) -> Tuple[int,int,int]:
        return PALETTE[self.color_key]["rgb"]

    def _paint(self, p1: Tuple[int,int], p2: Tuple[int,int]):
        """Thick segment + round end-caps on the ink canvas."""
        c = (*self.rgb, INK_ALPHA)
        r = max(self.pen_w // 2, 1)
        pygame.draw.line(self.canvas, c, p1, p2, self.pen_w)
        pygame.draw.circle(self.canvas, c, p1, r)
        pygame.draw.circle(self.canvas, c, p2, r)

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
        r    = ERASE_RADIUS
        mask = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 255))
        pygame.draw.circle(mask, (0, 0, 0, 0), (r, r), r)
        self.canvas.blit(mask, (cx-r, cy-r),
                         special_flags=pygame.BLEND_RGBA_MIN)

    def _commit(self):
        if len(self.cur_stroke) >= 2:
            self.undo.push(self.canvas)
            self.pending.append((list(self.cur_stroke), self.rgb, self.pen_w))
        self.cur_stroke.clear()
        self.ema["Right"].reset()

    def _launch(self):
        """Convert all pending ink strokes into physics rigid bodies."""
        self._commit()
        for pts, col, wid in self.pending:
            cx, cy = self.phys.add_stroke(pts, col, wid)
            self.parts.burst(cx, cy, col, P_LAUNCH)
        self.pending.clear()
        self.canvas.fill((0, 0, 0, 0))
        self.notify.show(f"🚀 Launched!  Bodies: {len(self.phys.bodies)}")

    def _clear(self):
        self.phys.clear()
        self.canvas.fill((0, 0, 0, 0))
        self.cur_stroke.clear()
        self.pending.clear()
        for ema in self.ema.values(): ema.reset()
        self._dragging = False
        self.notify.show("🗑 Cleared")

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
            ok = self.undo.pop(self.canvas)
            self.notify.show("Undone" if ok else "Nothing to undo")
        elif key == "gravity":
            self.notify.show(f"Gravity: {self.phys.cycle_gravity()}")
        elif key == "trails":
            self.show_trails = not self.show_trails
            self.notify.show(f"Trails {'ON' if self.show_trails else 'OFF'}")
        elif key == "screenshot":
            self._screenshot()
        elif key == "explode":
            self.phys.explode(WIN_W/2, WIN_H/2)
            self.parts.burst(WIN_W/2, WIN_H/2, (255,200,50), P_EXPLODE)
            self.notify.show("Explosion!")
        elif key == "wind_l":
            self._wind = -1800 if self._wind >= 0 else 0.0
            self.notify.show("Wind ON" if self._wind else "Wind OFF")
        elif key == "wind_r":
            self._wind =  1800 if self._wind <= 0 else 0.0
            self.notify.show("Wind ON" if self._wind else "Wind OFF")
        self.phys.set_wind(self._wind)

    @staticmethod
    def _bgr_to_surf(bgr: np.ndarray) -> pygame.Surface:
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb  = np.flipud(rgb)
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        if surf.get_size() != (WIN_W, WIN_H):
            surf = pygame.transform.scale(surf, (WIN_W, WIN_H))
        return surf

    def run(self):
        print("✅  Air Draw Physics v10  (two-hand, zero-mirror)")
        print("   Right hand: ☝ DRAW  🤌 RESIZE  👊 ERASE  🖐 LAUNCH")
        print("   Left  hand: ✌ GRAB  👊 PUSH    🖐 BLAST")
        print("   Hover finger over colour/toolbar 0.5 s to activate.")
        print("   Hold right palm 1 s for radial menu.\n")

        prev_t = time.perf_counter()

        while True:
            now   = time.perf_counter()
            dt    = min(now - prev_t, 0.05)
            prev_t = now

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._quit()
                elif ev.type == pygame.KEYDOWN:
                    k = pygame.key.name(ev.key)
                    if   k in ("q", "escape"): self._quit()
                    elif k == "space": self._do_action("launch")
                    elif k == "c": self._do_action("clear")
                    elif k == "z": self._do_action("undo")
                    elif k == "g": self._do_action("gravity")
                    elif k == "t": self._do_action("trails")
                    elif k == "e": self._do_action("explode")
                    elif k == "s": self._do_action("screenshot")
                    elif k in PALETTE:
                        self.color_key = k
                        self.notify.show(f"🎨 {PALETTE[k]['name']}", PALETTE[k]['rgb'])
                    elif k in ("+", "="):
                        self.pen_w = min(self.pen_w + 1, MAX_PEN_W)
                        self.notify.show(f"✏ Pen: {self.pen_w}px")
                    elif k == "-":
                        self.pen_w = max(self.pen_w - 1, MIN_PEN_W)
                        self.notify.show(f"✏ Pen: {self.pen_w}px")

            frame = self.cam.read()
            if frame is None:
                self.clock.tick(FPS_CAP)
                continue

            rgb_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands: List[HandData] = self.clf.process(rgb_raw)
            for hd in hands:
                self.clf.draw_skeleton(frame, hd.landmarks, hd.label)
                dbg = {
                    "DRAW":   (50, 220,  50),
                    "LAUNCH": (50, 180, 230),
                    "GRAB":   (80, 160, 230),
                    "ERASE":  (230, 100, 60),
                    "PINCH":  (200, 100, 230),
                    "PUSH":   (230, 140, 60),
                    "BLAST":  (230,  80, 50),
                    "IDLE":   (140, 140, 140),
                }
                x_dbg = 10 if hd.label == "Right" else WIN_W - 140
                cv2.putText(
                    frame, f"{hd.label[0]}:{hd.gesture}",
                    (x_dbg, frame.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    dbg.get(hd.gesture, (180,180,180)), 2,
                )

            rh   = next((h for h in hands if h.label == "Right"), None)
            lh   = next((h for h in hands if h.label == "Left"),  None)
            rtip = rh.tip if rh else None
            ltip = lh.tip if lh else None
            rg   = rh.gesture if rh else "IDLE"
            lg   = lh.gesture if lh else "IDLE"

            ui_tip = rtip or ltip
            cf = self.palette.update(ui_tip, dt)
            if cf:
                self.color_key = cf
                self.notify.show(f"{PALETTE[cf]['name']}", PALETTE[cf]['rgb'])

            tf = self.toolbar.update(ui_tip, dt)
            if tf:
                self._do_action(tf)
            if self.clf.radial_ready("Right") and not self.radial.visible:
                cx, cy = rtip if rtip else (WIN_W//2, WIN_H//2)
                self.radial.open(cx, cy)
                self.clf.reset_radial("Right")
            if self.radial.visible and rg != "LAUNCH":
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

            if rg == "DRAW" and rtip is not None:
                sx, sy = self.ema["Right"].update(float(rtip[0]), float(rtip[1]))
                sx, sy = int(sx), int(sy)

                if self.prev_g["Right"] != "DRAW":
                    self.ema["Right"].reset()
                    sx, sy = self.ema["Right"].update(float(rtip[0]), float(rtip[1]))
                    sx, sy = int(sx), int(sy)
                    self.cur_stroke = [(sx, sy)]
                else:
                    if self.cur_stroke:
                        lx, ly = self.cur_stroke[-1]
                        dist = math.hypot(sx - lx, sy - ly)
                        if dist >= MIN_MOVE_PX:
                            for fp in self._densify((lx, ly), (sx, sy)):
                                self._paint(self.cur_stroke[-1], fp)
                                self.cur_stroke.append(fp)

                ccx = int(rtip[0] * CAM_W / WIN_W)
                ccy = int(rtip[1] * CAM_H / WIN_H)
                bgr = (self.rgb[2], self.rgb[1], self.rgb[0])
                cv2.circle(frame, (ccx, ccy), CURSOR_OUTER, bgr, 2)
                cv2.circle(frame, (ccx, ccy), CURSOR_INNER, bgr, -1)

                if self._dragging:
                    self.phys.end_drag()
                    self._dragging = False

            elif rg == "PINCH" and rh and rh.pinch_d is not None:
                t_ = max(0.0, min(1.0,
                    (rh.pinch_d - PINCH_CLOSE) / (PINCH_OPEN - PINCH_CLOSE)))
                new_w = int(MIN_PEN_W + t_ * (MAX_PEN_W - MIN_PEN_W))
                if new_w != self.pen_w:
                    self.pen_w = new_w
                if self.cur_stroke: self._commit()
                if self._dragging:
                    self.phys.end_drag()
                    self._dragging = False

            elif rg == "ERASE" and rtip is not None:
                if self.cur_stroke: self._commit()
                self._erase(*rtip)
                ecx = int(rtip[0] * CAM_W / WIN_W)
                ecy = int(rtip[1] * CAM_H / WIN_H)
                cv2.circle(frame, (ecx, ecy),
                           int(ERASE_RADIUS * CAM_W / WIN_W), (80, 80, 230), 2)
                if self._dragging:
                    self.phys.end_drag()
                    self._dragging = False

            else:
                if self.prev_g["Right"] == "DRAW" and self.cur_stroke:
                    self._commit()

            if lg == "GRAB" and ltip is not None:
                self.vel["Left"].update(float(ltip[0]), float(ltip[1]))

                if not self._dragging:
                    self.phys.start_drag(ltip)
                    self._dragging = True
                else:
                    self.phys.update_drag(ltip)

                dcx = int(ltip[0] * CAM_W / WIN_W)
                dcy = int(ltip[1] * CAM_H / WIN_H)
                cv2.circle(frame, (dcx, dcy), 22, (80, 160, 230), 2)
                cv2.circle(frame, (dcx, dcy),  5, (80, 160, 230), -1)

            elif lg == "PUSH" and ltip is not None:
                if self.prev_g["Left"] != "PUSH":
                    self.phys.push_near(ltip)
                if self._dragging:
                    vx, vy = self.vel["Left"].velocity()
                    self.phys.end_drag((vx, vy))
                    self._dragging = False
                    self.vel["Left"].reset()

            elif lg == "BLAST" and ltip is not None:
                if self.prev_g["Left"] != "BLAST":
                    self.phys.explode(float(ltip[0]), float(ltip[1]))
                    self.parts.burst(float(ltip[0]), float(ltip[1]),
                                     (255, 200, 50), P_EXPLODE)
                    self.notify.show("Hand Blast!")
                if self._dragging:
                    vx, vy = self.vel["Left"].velocity()
                    self.phys.end_drag((vx, vy))
                    self._dragging = False
                    self.vel["Left"].reset()

            else:
                if self._dragging and lg != "GRAB":
                    vx, vy = self.vel["Left"].velocity()
                    self.phys.end_drag((vx, vy))
                    self._dragging = False
                    self.vel["Left"].reset()

            self.prev_g["Right"] = rg
            self.prev_g["Left"]  = lg

            if self.phys.bodies:
                self.phys.step(dt)
            self.parts.update(dt)
            self.notify.update(dt)
            for hd in hands:
                col = self.rgb if hd.label == "Right" else (80, 160, 230)
                self.cursor.update(hd.label, hd.tip, col)
            self.screen.blit(self._bgr_to_surf(frame), (0, 0))
            self.screen.blit(self.canvas, (0, 0))
            self.phys.render(self.screen, self.show_trails)
            self.parts.render(self.screen)
            self.radial.render(self.screen)
            self.toolbar.render(self.screen)
            self.palette.render(self.screen, self.color_key)

            for hd in hands:
                col = self.rgb if hd.label == "Right" else (80, 160, 230)
                self.cursor.render(self.screen, hd.label, hd.tip, col, hd.gesture)

            self.fps = self.clock.get_fps()
            self.hud.render(
                self.screen, hands, self.color_key, self.pen_w,
                self.fps, len(self.phys.bodies),
                self.phys.gravity_idx, self.show_trails, self._wind,
            )
            self.notify.render(self.screen)

            pygame.display.flip()
            self.clock.tick(FPS_CAP)

    def _quit(self):
        print("\n🛑  Shutting down…")
        self.cam.stop()
        self.clf.close()
        pygame.quit()
        sys.exit(0)
if __name__ == "__main__":
    App().run()
