# EyeBlinkDetection

# This is for my science research project for 24/25. This is SOLELY for eye blink detection. The other red-eye-sclera detection will be another repository.


import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

import pandas as pd
from fontTools.merge.util import first

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [0.5, 1.5], invert=True)

leftIdList = [466, 388, 387, 386, 385, 384, 398, 263, 249, 390, 373, 374, 380, 381, 382, 362]
rightIdList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
leftRectIds = [386, 374, 382, 467]
rightRectIds = [159, 23, 130, 243]

ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

from datetime import datetime

start = datetime.now()

STATE = None
ratioAvg = 0

# ratioTS = pd.DataFrame(columns=['ratio', 'ver', 'hor'])
while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    end = datetime.now()
    difference = end - start
    seconds = difference.total_seconds()
    # cv2.putText(faces, "Time elapsed %s seconds" % int(seconds), (100,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if faces:
        face = faces[0]
        for idList in [leftIdList, rightIdList]:
            # for id in idList:
            #     cv2.circle(img, face[id], 5, color, cv2.FILLED)

            bothRectIds = [leftRectIds]#, rightRectIds]
            for j in range(len(bothRectIds)):
                leftUp, leftDown, leftLeft, leftRight = [face[i] for i in bothRectIds[j]]
                lengthVer, _ = detector.findDistance(leftUp, leftDown)
                lengthHor, _ = detector.findDistance(leftLeft, leftRight)
                ratio = int((lengthVer / lengthHor) * 100)

                # ratioTS.loc[end, ['ratio', 'ver', 'hor']] = ratio, lengthVer, lengthHor
                ratioList.append(ratio)

                # df = ratioTS.copy()
                # for N1 in range(1,10):
                #     for N2 in range(1, 10):
                #         print(N1, N2)
                #         m1 = ratioTS['ratio'].rolling(N1, min_periods=N1).mean().shift(N1)
                #         m2 = ratioTS['ratio'].rolling(N2, min_periods=N2).mean()
                #         df['ratio-(%s, %s)' % (N1,N2)] = m2/m1
                #         df['diff-(%s, %s)' % (N1,N2)] = m2-m1
                # df.to_clipboard()
                N1 = 7
                N2 = 2
                if len(ratioList) > (N1+N2):
                    if STATE is None:
                        STATE = 'IDLE'
                    first_7 = sum((ratioList[-(N1+N2):-N2])) / len((ratioList[-(N1+N2):-N2]))
                    last_2 = sum((ratioList[-(N2):]))/len((ratioList[-(N2):]))
                    ratioAvg = last_2 / first_7

                if STATE == 'IDLE':
                    if ratioAvg > 1.05:
                        STATE = 'WAIT_FOR_REVERSION'

                if STATE == 'WAIT_FOR_REVERSION':
                    if ratioAvg < 0.8:
                        STATE = 'WAIT_FOR_RESET'
                        blinkCounter += 1

                if STATE == 'WAIT_FOR_RESET':
                    if ratioAvg >= 1:
                        STATE = 'IDLE'

                print(datetime.now(), ratio, lengthVer, lengthHor)

                # cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
                # cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        if seconds > 30:
            cvzone.putTextRect(img, f'Blink Count: {blinkCounter}' + ', %sbpm' % int(blinkCounter / seconds * 60), (50, 100), colorR=color)
        else:
            cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)


        imgPlot = plotY.update(ratioAvg, color)

        img = cv2.resize(img, (640, 480))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 480))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Image", imgStack)
    cv2.waitKey(1)
