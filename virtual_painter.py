import cv2
import numpy as np
from hand_tracking_module import HandDetector

brushThickness = 15
eraserThickness = 60

detector = HandDetector()

drawColor = (255,0,255)

cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)

xp, yp = 0,0

imgCanvas = np.zeros((720,1280,3),np.uint8)

while True:

    success,img = cap.read()

    img=cv2.flip(img,1)

    img=detector.findHands(img)

    lmList=detector.findPosition(img)

    if len(lmList)!=0:

        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]

        fingers=detector.fingersUp()

        # Selection Mode
        if len(fingers)>=2 and fingers[1] and fingers[2]:

            xp,yp=0,0

            if y1<100:

                if 50<x1<250:
                    drawColor=(255,0,255)

                elif 300<x1<500:
                    drawColor=(255,0,0)

                elif 550<x1<750:
                    drawColor=(0,255,0)

                elif 800<x1<1000:
                    drawColor=(0,0,255)

                elif 1050<x1<1250:
                    drawColor=(0,0,0)

            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)

        # Drawing Mode
        if len(fingers)>=2 and fingers[1] and not fingers[2]:

            cv2.circle(img,(x1,y1),10,drawColor,cv2.FILLED)

            if xp==0 and yp==0:
                xp,yp=x1,y1

            if drawColor==(0,0,0):

                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)

                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)

            else:

                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)

                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)

            xp,yp=x1,y1

    cv2.rectangle(img,(50,10),(250,90),(255,0,255),cv2.FILLED)
    cv2.putText(img,"Pink",(80,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.rectangle(img,(300,10),(500,90),(255,0,0),cv2.FILLED)
    cv2.putText(img,"Blue",(340,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.rectangle(img,(550,10),(750,90),(0,255,0),cv2.FILLED)
    cv2.putText(img,"Green",(570,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.rectangle(img,(800,10),(1000,90),(0,0,255),cv2.FILLED)
    cv2.putText(img,"Red",(850,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.rectangle(img,(1050,10),(1250,90),(0,0,0),cv2.FILLED)
    cv2.putText(img,"Erase",(1080,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)

    _,imgInv=cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)

    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)

    img=cv2.bitwise_and(img,imgInv)

    img=cv2.bitwise_or(img,imgCanvas)

    cv2.imshow("AI Virtual Painter",img)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()

cv2.destroyAllWindows()