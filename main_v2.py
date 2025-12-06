import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from gesture_math import GestureMathEngine

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, label):
    lm = hand_landmarks.landmark
    TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

    fingers = {k: False for k in TIP.keys()}

    thumb_tip = lm[TIP["thumb"]]
    thumb_mcp = lm[2]
    fingers["thumb"] = thumb_tip.x > thumb_mcp.x if label == "Right" else thumb_tip.x < thumb_mcp.x

    for f in ["index", "middle", "ring", "pinky"]:
        fingers[f] = lm[TIP[f]].y < lm[PIP[f]].y

    return sum(fingers.values()), fingers


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cv2.setUseOptimized(True)

    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    face_detector = mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.65,
    )

    engine = GestureMathEngine()
    last_time = time.time()

    # Stability buffers for finger counts
    finger_history_left = deque(maxlen=8)
    finger_history_right = deque(maxlen=8)

    # Gesture stability
    gesture_counter = {"add": 0, "sub": 0, "mul": 0, "eval": 0}

    # Clap stability
    last_clap_dist = None
    clap_movements = deque(maxlen=6)

    def stable_value(history: deque):
        if not history:
            return None
        return max(set(history), key=history.count)

    def confirm_gesture(g):
        if g is None:
            return None
        gesture_counter[g] += 1
        if gesture_counter[g] >= 5:  # needs 5 frames of agreement
            for k in gesture_counter:
                gesture_counter[k] = 0
            return g
        return None

    def detect_clap(dist):
        nonlocal last_clap_dist
        if last_clap_dist is not None:
            movement = dist - last_clap_dist
            clap_movements.append(movement)

            # approaching (negative movement) + close distance
            if len(clap_movements) >= 4 and all(m < -2 for m in list(clap_movements)[-4:]) and dist < 70:
                return "eval"

        last_clap_dist = dist
        return None

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

            left = right = None
            gesture = None

            # Optional: show face box (good for demo)
            if face_results.detections:
                for det in face_results.detections:
                    mp_drawing.draw_detection(frame, det)

            # Hands + per-frame finger counts
            if hand_results.multi_hand_landmarks:
                for hol, hnd in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                ):
                    label = hnd.classification[0].label  # "Left" / "Right"
                    count, _ = count_fingers(hol, label)

                    mp_drawing.draw_landmarks(frame, hol, mp_hands.HAND_CONNECTIONS)

                    cx = int(hol.landmark[0].x * w)
                    cy = int(hol.landmark[0].y * h)
                    cv2.putText(
                        frame,
                        f"{label}:{count}",
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    if label == "Left":
                        finger_history_left.append(count)
                    else:
                        finger_history_right.append(count)

            # Smoothed counts
            left = stable_value(finger_history_left)
            right = stable_value(finger_history_right)

            # Operator gestures (with confirmation)
            if left is not None and right is not None:
                if left == 5 and right == 5:
                    gesture = confirm_gesture("add")
                elif left == 0 or right == 0:
                    gesture = confirm_gesture("sub")
                elif abs(left - right) >= 4:
                    gesture = confirm_gesture("mul")

            # Clap → eval gesture
            if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
                lm0 = hand_results.multi_hand_landmarks[0].landmark[0]
                lm1 = hand_results.multi_hand_landmarks[1].landmark[0]
                dist = np.hypot((lm0.x - lm1.x) * w, (lm0.y - lm1.y) * h)
                clap = detect_clap(dist)
                if clap:
                    gesture = confirm_gesture(clap)
            else:
                last_clap_dist = None
                clap_movements.clear()

            # Update math engine
            engine.update(left, right, gesture)
            hud_text = engine.get_display()

            # FPS
            current_time = time.time()
            fps = int(1 / (current_time - last_time)) if current_time != last_time else 0
            last_time = current_time

            # HUD
            cv2.rectangle(frame, (0, 0), (w, 60), (10, 10, 10), -1)
            cv2.putText(
                frame,
                hud_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 240, 80),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (w - 130, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (80, 220, 255),
                2,
            )

            cv2.imshow("V2 - Stable Gesture Math Engine", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        hands.close()
        face_detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
