import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
import pyautogui

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2) 
mpDraw = mp.solutions.drawing_utils

def fingers_spread(lmList):
    thumb_index_dist = hypot(lmList[4][1] - lmList[8][1], lmList[4][2] - lmList[8][2])
    index_middle_dist = hypot(lmList[8][1] - lmList[12][1], lmList[8][2] - lmList[12][2])
    middle_ring_dist = hypot(lmList[12][1] - lmList[16][1], lmList[12][2] - lmList[16][2])
    ring_pinky_dist = hypot(lmList[16][1] - lmList[20][1], lmList[16][2] - lmList[20][2])
    
    return thumb_index_dist > 50 and index_middle_dist > 50 and middle_ring_dist > 50 and ring_pinky_dist > 50

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList1 = []
    lmList2 = []
    
    if results.multi_hand_landmarks:
        for idx, handlandmark in enumerate(results.multi_hand_landmarks):
            lmList = []
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            if idx == 0:
                lmList1 = lmList
            elif idx == 1:
                lmList2 = lmList
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)
    
    if lmList1:
        x1, y1 = lmList1[4][1], lmList1[4][2]   # Thumb tip
        x2, y2 = lmList1[8][1], lmList1[8][2]   # Index tip
        length_thumb_index = hypot(x2 - x1, y2 - y1)

        # Brightness Control
        bright = np.interp(length_thumb_index, [15, 220], [0, 100])
        print(bright, length_thumb_index)
        sbc.set_brightness(int(bright))

    if lmList2:
        x1, y1 = lmList2[4][1], lmList2[4][2]   # Thumb tip
        x2, y2 = lmList2[8][1], lmList2[8][2]   # Index tip
        length_thumb_index = hypot(x2 - x1, y2 - y1)

        # Brightness Control
        bright = np.interp(length_thumb_index, [15, 220], [0, 100])
        print(bright, length_thumb_index)
        sbc.set_brightness(int(bright))
    
    if lmList1 and lmList2:
        if fingers_spread(lmList1) and fingers_spread(lmList2):
            cv2.putText(img, 'Screenshot Taken', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            pyautogui.screenshot('screenshot.png')
    
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
