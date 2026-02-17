import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Correct volume initialization for stable pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# MediaPipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            h, w, _ = img.shape

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            # Thumb tip = 4, Index tip = 8
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cv2.circle(img, (x1, y1), 10, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0,255,0), cv2.FILLED)
            cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 3)

            length = np.hypot(x2-x1, y2-y1)

            # Map distance to volume
            vol = np.interp(length, [30, 200], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Volume Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
