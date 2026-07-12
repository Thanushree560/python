# Hand Tracking Project (Python + OpenCV + MediaPipe)

A webcam-based hand-tracking project built with **OpenCV** and **MediaPipe**.
It includes a reusable detector module and two demo apps.

## Files

| File | Description |
|---|---|
| `hand_tracking_module.py` | Reusable `HandDetector` class — landmark detection, finger-up detection, distance measurement. Run directly for a basic landmark viewer. |
| `finger_counter.py` | Counts how many fingers are raised and displays the number live. |
| `virtual_painter.py` | Draw in the air with your index finger; switch colors, clear canvas, save your drawing. |
| `requirements.txt` | Python dependencies. |

## Setup

1. Make sure you have **Python 3.9–3.11** (MediaPipe doesn't yet support every newest Python version — 3.10 is a safe bet).
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure your webcam is connected and not in use by another app.

## Running the demos

**Basic landmark viewer (sanity check):**
```bash
python hand_tracking_module.py
```

**Finger counter:**
```bash
python finger_counter.py
```
Hold up any number of fingers to the camera and watch the count update. Press `q` to quit.

**Virtual painter (air drawing):**
```bash
python virtual_painter.py
```
- ☝️ Index finger only → draw
- ✌️ Index + middle finger → move without drawing (selection mode)
- 🖐️ All 5 fingers → clear canvas
- `c` → cycle brush color
- `s` → save your drawing as a PNG
- `q` → quit

## How it works

MediaPipe's Hands solution detects **21 landmarks** per hand (wrist, finger joints,
fingertips) in real time from the webcam feed. `HandDetector`:

1. Converts each frame to RGB and runs it through MediaPipe's hand-landmark model.
2. Extracts pixel coordinates for each of the 21 landmarks.
3. Uses simple geometric rules (e.g., is a fingertip's y-coordinate above its
   knuckle's y-coordinate?) to determine which fingers are extended.
4. Exposes helper methods (`fingers_up`, `find_distance`) that the demo apps
   build gesture logic on top of.

## Ideas to extend this project

- **Volume/brightness control**: map the thumb–index finger distance to system volume using `pycaw` (Windows) or `osascript` (macOS).
- **Virtual mouse**: move the system cursor with your index finger and "click" with a pinch gesture, using `pyautogui`.
- **Sign language digit recognition**: extend `fingers_up` logic to recognize more complex gestures.
- **Two-hand gestures**: `max_hands=2` is already supported — try gestures that use both hands together (e.g., zoom by changing distance between two index fingers).

## Troubleshooting

- **Black/frozen webcam window**: try `cv2.VideoCapture(1)` instead of `0` if you have multiple cameras.
- **`ImportError` for mediapipe**: confirm your Python version is compatible (3.9–3.11 is safest as of mid-2026).
- **Low FPS**: lower the capture resolution (`cap.set(3, 640); cap.set(4, 480)`) or reduce `max_hands` to 1.
