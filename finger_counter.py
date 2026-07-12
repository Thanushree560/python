"""
finger_counter.py
------------------
Counts how many fingers are held up in front of the webcam using the
HandDetector module, and displays the count on screen.

Run with:  python finger_counter.py
Press 'q' to quit.
"""

import cv2
import time
from hand_tracking_module import HandDetector


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detection_confidence=0.75, max_hands=1)
    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read from webcam.")
            break

        frame = detector.find_hands(frame)
        landmark_list = detector.find_positions(frame)

        if landmark_list:
            hand_label = detector.get_hand_label() or "Right"
            fingers = detector.fingers_up(landmark_list, hand_label)
            count = fingers.count(1)

            # Big counter box
            cv2.rectangle(frame, (20, 20), (200, 160), (0, 200, 0), cv2.FILLED)
            cv2.putText(frame, str(count), (60, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 6)

            cv2.putText(frame, f'Hand: {hand_label}', (220, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1] - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Finger Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
