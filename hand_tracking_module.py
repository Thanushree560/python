"""
hand_tracking_module.py
------------------------
A reusable hand-detection wrapper around MediaPipe Hands.

Import HandDetector into your own scripts:

    from hand_tracking_module import HandDetector

    detector = HandDetector()
    frame = detector.find_hands(frame)
    landmark_list = detector.find_positions(frame)
"""

import cv2
import mediapipe as mp
import math


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.7,
                 tracking_confidence=0.5):
        """
        mode: static image mode if True, otherwise treats input as a video stream
        max_hands: maximum number of hands to detect
        detection_confidence: minimum confidence for initial detection
        tracking_confidence: minimum confidence for landmark tracking
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # Landmark indices for fingertips: thumb, index, middle, ring, pinky
        self.tip_ids = [4, 8, 12, 16, 20]
        self.results = None

    def find_hands(self, frame, draw=True):
        """Detects hands in a BGR frame and optionally draws landmarks on it."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style(),
                    )
        return frame

    def find_positions(self, frame, hand_index=0):
        """
        Returns a list of [id, x, y] for each of the 21 landmarks of the
        requested hand (0 = first hand found), plus a bounding box.
        Returns an empty list if no hand is found.
        """
        landmark_list = []
        self.bbox = None

        if self.results and self.results.multi_hand_landmarks:
            if hand_index >= len(self.results.multi_hand_landmarks):
                return landmark_list

            hand = self.results.multi_hand_landmarks[hand_index]
            h, w, _ = frame.shape
            x_list, y_list = [], []

            for lm_id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                landmark_list.append([lm_id, cx, cy])

            self.bbox = (min(x_list), min(y_list), max(x_list), max(y_list))

        return landmark_list

    def fingers_up(self, landmark_list, hand_label="Right"):
        """
        Given a landmark_list from find_positions(), returns a list of 5
        values (1 = finger up, 0 = finger down) in order:
        [thumb, index, middle, ring, pinky].

        hand_label: "Right" or "Left" (affects thumb direction check).
        """
        fingers = []
        if not landmark_list:
            return [0, 0, 0, 0, 0]

        # Thumb: compare x-coordinates (direction depends on hand)
        if hand_label == "Right":
            fingers.append(1 if landmark_list[self.tip_ids[0]][1] >
                            landmark_list[self.tip_ids[0] - 1][1] else 0)
        else:
            fingers.append(1 if landmark_list[self.tip_ids[0]][1] <
                            landmark_list[self.tip_ids[0] - 1][1] else 0)

        # Other four fingers: compare y-coordinates (tip above pip joint = up)
        for tip_id in self.tip_ids[1:]:
            fingers.append(1 if landmark_list[tip_id][2] <
                            landmark_list[tip_id - 2][2] else 0)

        return fingers

    def find_distance(self, p1, p2, landmark_list, frame=None, draw=True):
        """
        Distance between two landmark ids (e.g. thumb tip=4, index tip=8).
        Returns (distance, frame, [x1, y1, x2, y2, cx, cy]).
        """
        x1, y1 = landmark_list[p1][1], landmark_list[p1][2]
        x2, y2 = landmark_list[p2][1], landmark_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw and frame is not None:
            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, frame, [x1, y1, x2, y2, cx, cy]

    def get_hand_label(self, hand_index=0):
        """Returns 'Left' or 'Right' for the given detected hand, or None."""
        if self.results and self.results.multi_handedness:
            if hand_index < len(self.results.multi_handedness):
                return self.results.multi_handedness[hand_index].classification[0].label
        return None


def main():
    """Quick standalone test: shows the webcam feed with hand landmarks + FPS."""
    import time

    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detector.find_hands(frame)
        landmark_list = detector.find_positions(frame)

        if landmark_list:
            print(landmark_list[4])  # example: print thumb tip position

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
