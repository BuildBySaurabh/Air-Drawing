# Air-Drawing



"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  AIR DRAW PHYSICS  v10  ─  Two-Hand · Zero-Mirror · Velocity-Throw               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  INSTALL:  pip install opencv-python mediapipe pymunk pygame numpy               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  RIGHT HAND  (drawing)                                                           ║
║   ☝  Index up, middle DOWN    → DRAW  (ink locked to fingertip, no mirror)      ║
║   🤌  Pinch index+thumb       → RESIZE (pen width in real-time)                  ║
║   👊  Fist                    → ERASE (circular rubber eraser)                   ║
║   🖐  Open palm (hold 0.3 s)  → LAUNCH (drawings fall with physics)              ║
║                                                                                   ║
║  LEFT HAND  (physics sandbox)                                                     ║
║   ✌  Index+middle up          → GRAB  (pick up, move, & throw objects)          ║
║   👊  Fist                    → PUSH  (push nearest object away)                 ║
║   🖐  Open palm               → BLAST (area explosion at hand position)          ║
║                                                                                  ║
║  FINGER UI  (no keyboard required)                                               ║
║   Hover over COLOR SWATCH  (top-right)   0.5 s → change color                    ║
║   Hover over TOOLBAR BUTTON (bottom)     0.5 s → activate                        ║
║   Hold right open palm                   1.0 s → radial menu                     ║
║                                                                                  ║
║  KEYBOARD SHORTCUTS  (fallback / power-user)                                     ║
║   SPACE  launch    C  clear    Z  undo    G  gravity    T  trails                ║
║   E  explode   S  screenshot   +/-  pen   1-8  color   Q/ESC  quit               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  ── WHY DRAWING FOLLOWS YOUR FINGER EXACTLY (NO MIRROR) ─────────────────────    ║
║                                                                                  ║
║  The camera frame is captured RAW — never flipped.                               ║
║  MediaPipe receives the RAW frame and returns lm.x / lm.y in [0, 1]              ║
║  normalised to the image it was given.                                           ║
║                                                                                  ║
║  We map those to display pixels ONCE and ONLY ONCE:                              ║
║      disp_x = lm.x * WIN_W                                                       ║
║      disp_y = lm.y * WIN_H                                                       ║
║                                                                                  ║
║  The pygame background is also built from the same raw frame (no h-flip).        ║
║  So every layer — camera image, hand skeleton, ink, physics bodies, cursor —     ║
║  all share the exact same coordinate space.                                      ║
║  Result:  hand-right == screen-right.  Drawing "L" shows "L", never "⅃".         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
