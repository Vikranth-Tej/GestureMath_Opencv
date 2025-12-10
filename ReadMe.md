# GestureMath — CV Based Learning System

A real-time, camera-based interactive math system built using OpenCV + MediaPipe.
Basic arithmetics using **hand gestures** —
no buttons, no controllers… just **hands**! 🖐

---

##  Features

| Feature |
|--------|
| Face Detection |
| Finger Counting (both hands) |
| Automatic Addition (Left + Right hand) 
| Single-Hand Gesture Math (Add / Sub / Mul) |
| Gesture-based Evaluation (open palm) |
| FPS Counter + UI Overlay |


---
# Tech Stack

## Computer Vision

- **MediaPipe Hands & Face Detection** — Real-time landmarks & tracking
- **Hand Gesture Recognition** — Thumb, finger state analysis
- **State Machine Logic** — Gesture-based math operations
- **Stability Filtering** — Smoothing noisy predictions

## Vision Processing

- **OpenCV** — Camera input, frame processing, UI overlays
- **NumPy** — Geometric calculations & vector operations

---

##  Gesture Controls

| Gesture | Meaning | Example |
|--------|---------|---------|
| ✋ Show any number | Choose number | (2 fingers → 2) |
| 👍 Thumb up only | **+** operator | 2 ➕ ... |
| ✊ Fist (0 fingers) | **−** operator | 5 ➖ ... |
| ✌️ Two fingers | **×** operator | 3 ✖ ... |
| 🤚 4–5 fingers open | **Evaluate** (=) | Show result |

✔ Only **one hand required**  

---

##  Project Structure
```
Math_Gestures/
    │
    ├─ main_v1.py        # Face,both hands(addition)
    |
    ├─ main_v2.py        # Full gesture math engine
    |
    ├─ gesture_math.py   # Math logic state machine
    |
    └─ requirements.txt  

```

