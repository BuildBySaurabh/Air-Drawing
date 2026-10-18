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

<img width="1672" height="941" alt="Image" src="https://github.com/user-attachments/assets/e6604f2a-e85d-4290-b974-b8037404de80" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/c8b946ff-0054-4682-b4c4-b813d413ad99" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/2056c8d4-9aee-446d-ac8b-ef1db88ed334" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/5af31d4f-1465-46ef-b146-e59594ae9228" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/a9167f90-78ab-4eb6-9a70-018cc60c01ec" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/9339d20c-b47d-462a-88f5-cfe6a3e30131" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/40407b0b-ab6a-4e33-bc1c-1a8c50fb137b" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/e06b1df4-763f-4b90-b470-9e3012172175" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/b6713d5e-b47e-4af1-b32a-02dbd4ba058f" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/ba07a62b-bd98-4c4b-9ee2-31d7d318e355" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/d7cdcc87-b036-4a5d-b015-e47aaacf4b26" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/bd95de79-4ccc-44af-aed7-8d3a805bd44a" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/7ff0f0cc-f422-4af1-8bc3-51398f3cc2c0" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/039de42b-3e22-4cb6-9ac4-1ae87d46078a" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/32ab7f55-79c5-49b0-8e71-aff3c2619f26" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/a7be2b6d-a12e-42ed-ad78-0f3c0393b09d" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/c71e50c2-7cac-4e59-88a8-5d0c29e2f0e8" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/592b91d1-ac41-440f-8a23-cdd1c2fd58e9" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/85e7ecf3-f095-4b70-91a0-8926e48cbbb4" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/ddf1d06d-4779-4e21-8840-5e0d3d812d7d" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/76d2bac4-a92f-4d96-9f1d-8dc9340cad64" />
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/2faa8df5-5e2c-4e25-8b62-390f7b4f9f27" />
