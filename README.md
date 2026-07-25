# Air Drawing Physics

A real-time computer vision application that lets users draw in the air using hand gestures and interact with their drawings through a realistic 2D physics engine. Built with Python, OpenCV, MediaPipe, Pygame, and Pymunk, the project transforms hand movements captured by a webcam into an interactive drawing and physics simulation experience.

Unlike traditional drawing applications, this project supports gesture-based controls, object manipulation, realistic physics, and a zero-mirror coordinate system for a more natural drawing experience.

---

## Features

- Real-time hand tracking
- Air drawing using hand gestures
- Gesture recognition
- Realistic 2D physics simulation
- Object grabbing and throwing
- Explosion effects
- Brush size adjustment using pinch gestures
- Multi-color drawing
- Finger-controlled color palette
- Finger-controlled toolbar
- Motion trail effects
- Undo support
- Screenshot capture
- Gravity toggle
- Zero-mirror drawing system
- Fully interactive gesture-based UI
- Responsive and smooth rendering

---

## Hand Gesture Controls

### Right Hand

| Gesture | Action |
|---------|--------|
| Index Finger Up | Draw |
| Thumb + Index Pinch | Change Brush Size |
| Fist | Erase |
| Open Palm (Hold) | Launch Drawings into Physics |

### Left Hand

| Gesture | Action |
|---------|--------|
| Index + Middle Finger | Grab Physics Objects |
| Fist | Push Nearby Objects |
| Open Palm | Explosion Blast |

---

## Finger-Based Interface

No mouse or keyboard is required for normal operation.

### Color Palette

Hover over a color for approximately **0.5 seconds** to select it.

### Toolbar

Hover over toolbar buttons using your finger to activate them.

### Radial Menu

Hold your **right hand open** for about **1 second** to open the radial menu.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Launch Drawings |
| C | Clear Canvas |
| Z | Undo |
| G | Toggle Gravity |
| T | Toggle Motion Trails |
| E | Explosion |
| S | Save Screenshot |
| + / - | Increase or Decrease Brush Size |
| 1–8 | Change Drawing Color |
| Q / ESC | Quit |

---

## Camera Setup

### Option 1 — DroidCam (Default)

This project is configured to use **DroidCam** by default.

1. Install DroidCam on your Android phone.
2. Connect your phone and computer to the same Wi-Fi network.
3. Launch DroidCam and note the displayed IP address.
4. Update the stream URL if necessary.

```python
self.cap = cv2.VideoCapture("http://YOUR_PHONE_IP:4747/video")
```

Replace `YOUR_PHONE_IP` with your DroidCam IP address.

---

### Option 2 — Laptop Webcam

Replace

```python
self.cap = cv2.VideoCapture("http://YOUR_PHONE_IP:4747/video")
```

with

```python
self.cap = cv2.VideoCapture(0)
```

If the default camera is unavailable, try:

```python
cv2.VideoCapture(1)
```

or

```python
cv2.VideoCapture(2)
```

---

## Zero-Mirror Drawing

Unlike many webcam drawing applications, this project does **not** mirror the camera feed.

Benefits include:

- Natural left and right movement
- Correct handwriting orientation
- Accurate coordinate mapping
- More intuitive drawing experience

---

## How It Works

The application combines multiple technologies to create a seamless interactive experience.

- **OpenCV** captures webcam frames.
- **MediaPipe** detects hand landmarks.
- **Gesture Recognition** interprets hand poses.
- **Pygame** renders the user interface.
- **Pymunk** powers realistic physics simulation.
- **NumPy** performs mathematical calculations.

All components run together in real time to provide a smooth and responsive experience.

---

## Tech Stack

- Python
- OpenCV
- MediaPipe
- Pygame
- Pymunk
- NumPy

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/SauWagh/Air-Drawing.git
```

### Navigate to the Project

```bash
cd Air-Drawing
```

### Create a Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

Using requirements.txt

```bash
pip install -r requirements.txt
```

or

```bash
pip install opencv-python mediapipe pygame pymunk numpy
```

### Verify Installation

```bash
python -c "import cv2, mediapipe, pygame, pymunk, numpy; print('Everything Installed Successfully!')"
```

### Run the Application

```bash
python main.py
```

If your entry file has another name:

```bash
python air_drawing.py
```

---

## Python Version

Recommended versions:

- Python 3.10
- Python 3.11
- Python 3.12

Check your version:

```bash
python --version
```

---

## Required Libraries

| Package | Purpose |
|----------|----------|
| OpenCV | Webcam input and image processing |
| MediaPipe | Hand tracking and gesture recognition |
| Pygame | User interface and rendering |
| Pymunk | 2D physics engine |
| NumPy | Mathematical calculations |

Example `requirements.txt`

```text
opencv-python
mediapipe
pygame
pymunk
numpy
```

---

## Project Highlights

- Real-time computer vision
- AI-powered hand tracking
- Gesture recognition
- Physics-based drawing interaction
- Object grabbing and throwing
- Explosion effects
- Brush resizing
- Multi-color drawing
- Motion trails
- Undo support
- Screenshot capture
- Zero-mirror coordinate system
- Interactive finger-controlled interface

---

## Latest Features

- Two-hand gesture recognition
- Zero-mirror drawing system
- Finger-controlled toolbar
- Finger-controlled color palette
- Dynamic brush resizing
- Physics-based object launching
- Velocity-based object throwing
- Object grabbing
- Push nearby objects
- Explosion effects
- Gravity toggle
- Motion trails
- Undo support
- Screenshot capture
- Improved drawing precision
- Faster gesture detection
- Smoother rendering
- Enhanced collision handling
- More stable physics simulation

---

## Future Improvements

- Shape recognition
- AI gesture customization
- Multi-user support
- 3D physics simulation
- Additional brush styles
- Drawing layers
- Save and load drawings
- Object scaling
- Voice commands
- Touchscreen support
- Custom gesture training
- Improved UI animations

---

## Troubleshooting

### Camera Not Opening

- Close applications using the webcam (Zoom, Teams, OBS, Discord, etc.)
- Verify webcam permissions.
- Ensure DroidCam and your computer are connected to the same Wi-Fi network.
- Update the DroidCam IP address if it has changed.

### ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### MediaPipe Issues

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Reinstall MediaPipe:

```bash
pip install mediapipe
```

### Slow Performance

- Use good lighting.
- Close unnecessary background applications.
- Reduce webcam resolution if needed.
- Keep both hands clearly visible.

---

## Recommended Hardware

| Component | Recommendation |
|-----------|----------------|
| Webcam | 720p or 1080p |
| CPU | Intel i5 / AMD Ryzen 5 or higher |
| RAM | 8 GB or more |
| Operating System | Windows 10/11, Ubuntu 22.04+, macOS 12+ |

---

## Tested On

- Windows 11
- Python 3.11
- OpenCV 4.x
- MediaPipe
- Pygame 2.x
- Pymunk 7.x

---

## Tips for Best Experience

- Use good lighting.
- Keep both hands fully visible.
- Maintain a distance of approximately **50–80 cm** from the camera.
- Use a plain background for improved tracking.
- Keep the camera stable.

---

## Future Roadmap

- AI-powered gesture customization
- Advanced brush effects
- Collaborative drawing
- Cloud save support
- Drawing replay system
- Performance optimizations
- Enhanced UI and animations

---

## Contributing

Contributions are welcome.

If you have ideas for improvements, bug fixes, or new features, feel free to:

- Fork the repository
- Create a feature branch
- Submit a Pull Request
- Open an Issue

---

## Author

**Saurabh Waghamare**

Python Full-Stack Developer passionate about Computer Vision, Artificial Intelligence, Full-Stack Development, and building interactive real-time applications.
