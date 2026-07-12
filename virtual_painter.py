"""
virtual_painter.py
--------------------
Draw on screen using just your index finger in the air.

Controls (gestures):
  - Index finger up only          -> Draw
  - Index + middle fingers up     -> Selection mode (move without drawing)
  - All 5 fingers up              -> Clear the canvas
Press 'c' to change color, 's' to save your drawing, 'q' to quit.
"""

import cv2
import numpy as np
import time
from hand_tracking_module import HandDetector

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 255)]
COLOR_NAMES = ["Red", "Green", "Blue", "Yellow", "White"]
BRUSH_THICKNESS = 12
ERASER_THICKNESS = 60


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detection_confidence=0.8, max_hands=1)

    canvas = None
    prev_x, prev_y = 0, 0
    color_index = 0
    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        if canvas is None:
            canvas = np.zeros_like(frame)

        frame = detector.find_hands(frame, draw=True)
        landmark_list = detector.find_positions(frame)

        if landmark_list:
            x1, y1 = landmark_list[8][1], landmark_list[8][2]   # index tip
            x2, y2 = landmark_list[12][1], landmark_list[12][2]  # middle tip

            fingers = detector.fingers_up(landmark_list)

            # All fingers up -> clear canvas
            if fingers == [1, 1, 1, 1, 1]:
                canvas = np.zeros_like(frame)
                prev_x, prev_y = 0, 0

            # Selection mode: index + middle up
            elif fingers[1] and fingers[2] and not fingers[3]:
                prev_x, prev_y = 0, 0
                cv2.rectangle(frame, (x1 - 15, y1 - 25), (x2 + 15, y2 + 25),
                              COLORS[color_index], cv2.FILLED)

            # Draw mode: only index finger up
            elif fingers[1] and not fingers[2]:
                cv2.circle(frame, (x1, y1), 8, COLORS[color_index], cv2.FILLED)
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1
                cv2.line(canvas, (prev_x, prev_y), (x1, y1),
                          COLORS[color_index], BRUSH_THICKNESS)
                prev_x, prev_y = x1, y1
            else:
                prev_x, prev_y = 0, 0

        # Merge canvas onto the live frame
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, mask)
        frame = cv2.bitwise_or(frame, canvas)

        # UI: color palette bar
        for i, col in enumerate(COLORS):
            cv2.rectangle(frame, (10 + i * 60, 10), (60 + i * 60, 60), col, cv2.FILLED)
            if i == color_index:
                cv2.rectangle(frame, (10 + i * 60, 10), (60 + i * 60, 60), (0, 0, 0), 3)
        cv2.putText(frame, COLOR_NAMES[color_index], (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1] - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Virtual Painter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            color_index = (color_index + 1) % len(COLORS)
        elif key == ord('s'):
            filename = f"drawing_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print(f"Saved drawing to {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
