import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, handedness_label):
    """
    Count raised fingers for one hand using MediaPipe landmarks.
    """
    lm = hand_landmarks.landmark

    TIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    PIP_IDS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

    fingers = {k: False for k in TIP_IDS.keys()}

    # Thumb: horizontal comparison (x) depends on Left/Right
    thumb_tip = lm[TIP_IDS["thumb"]]
    thumb_mcp = lm[2]
    if handedness_label == "Right":
        fingers["thumb"] = thumb_tip.x > thumb_mcp.x
    else:
        fingers["thumb"] = thumb_tip.x < thumb_mcp.x

    # Other fingers: tip above PIP joint (y smaller = higher)
    for f in ["index", "middle", "ring", "pinky"]:
        fingers[f] = lm[TIP_IDS[f]].y < lm[PIP_IDS[f]].y

    count = sum(fingers.values())
    return count, fingers


def stable_value(history: deque):
    """Return the most common value in the history buffer."""
    if not history:
        return None
    return max(set(history), key=history.count)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cv2.setUseOptimized(True)

    # Light-weight, real-time configuration
    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    face_detector = mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6,
    )

    last_time = time.time()

    # Rolling buffers for left/right smoothing
    left_history = deque(maxlen=8)
    right_history = deque(maxlen=8)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_results = hands.process(rgb)
            face_results = face_detector.process(rgb)
            rgb.flags.writeable = True

            # FACE overlay (optional visual)
            if face_results.detections:
                for det in face_results.detections:
                    mp_drawing.draw_detection(frame, det)

            # HANDS + finger counts
            if hand_results.multi_hand_landmarks:
                for hol, handed in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                ):
                    label = handed.classification[0].label  # "Left" / "Right"
                    count, _ = count_fingers(hol, label)

                    mp_drawing.draw_landmarks(frame, hol, mp_hands.HAND_CONNECTIONS)

                    cx = int(hol.landmark[0].x * w)
                    cy = int(hol.landmark[0].y * h)
                    cv2.putText(
                        frame,
                        f"{label}:{count}",
                        (cx - 40, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    if label == "Left":
                        left_history.append(count)
                    else:
                        right_history.append(count)

            left_count = stable_value(left_history)
            right_count = stable_value(right_history)

            # Equation text
            equation = "Raise both hands to add"
            if left_count is not None and right_count is not None:
                result = left_count + right_count
                equation = f"{left_count} + {right_count} = {result}"

            # FPS
            current_time = time.time()
            fps = int(1 / (current_time - last_time)) if current_time != last_time else 0
            last_time = current_time

            # HUD
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(
                frame, equation, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
            )
            cv2.putText(
                frame, f"FPS: {fps}", (w - 130, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
            )

            cv2.imshow("V1 - Finger Math Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        hands.close()
        face_detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
