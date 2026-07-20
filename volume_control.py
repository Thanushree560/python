import cv2
import mediapipe as mp
import numpy as np
import math

from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# -------------------------
# Initialize Windows Volume
# -------------------------
devices = AudioUtilities.GetSpeakers()

interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# -------------------------
# MediaPipe Hands
# -------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

# -------------------------
# Webcam
# -------------------------
cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

            h, w, c = img.shape

            lmList = []

            for idx, lm in enumerate(handLms.landmark):
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                lmList.append((idx, cx, cy))

            # Thumb tip
            x1, y1 = lmList[4][1], lmList[4][2]

            # Index finger tip
            x2, y2 = lmList[8][1], lmList[8][2]

            # Draw circles
            cv2.circle(img, (x1, y1), 12, (255, 0, 255), -1)
            cv2.circle(img, (x2, y2), 12, (255, 0, 255), -1)

            # Draw line
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Midpoint
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)

            # Distance
            length = math.hypot(x2 - x1, y2 - y1)

            # Convert distance to volume
            vol = np.interp(length, [30, 220], [minVol, maxVol])
            volBar = np.interp(length, [30, 220], [400, 150])
            volPer = np.interp(length, [30, 220], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

            # Volume Bar
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), -1)

            cv2.putText(
                img,
                f'{int(volPer)}%',
                (35, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3
            )

    cv2.imshow("Gesture Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()