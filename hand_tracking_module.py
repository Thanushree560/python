import cv2
import mediapipe as mp

class HandDetector():

    def __init__(self,
                 mode=False,
                 maxHands=1,
                 detectionCon=0.7,
                 trackCon=0.7):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:

            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0):

        lmList=[]

        if self.results.multi_hand_landmarks:

            myHand=self.results.multi_hand_landmarks[handNo]

            for id,lm in enumerate(myHand.landmark):

                h,w,c=img.shape

                cx=int(lm.x*w)
                cy=int(lm.y*h)

                lmList.append([id,cx,cy])

        return lmList

    def fingersUp(self):

        fingers=[]

        tipIds=[4,8,12,16,20]

        if len(self.results.multi_hand_landmarks)!=0:

            myHand=self.results.multi_hand_landmarks[0]

            lm=[]

            for point in myHand.landmark:
                lm.append(point)

            fingers.append(lm[4].x>lm[3].x)

            fingers.append(lm[8].y<lm[6].y)
            fingers.append(lm[12].y<lm[10].y)
            fingers.append(lm[16].y<lm[14].y)
            fingers.append(lm[20].y<lm[18].y)

        return fingers