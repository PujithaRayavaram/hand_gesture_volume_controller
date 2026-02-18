import cv2
import mediapipe as mp
import time
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ================= AUDIO SETUP =================
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

pTime = 0
prevVol = 0
smoothFactor = 5

gesture_enabled = False

minDistance = 40
maxDistance = 160

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, c = frame.shape

    # ======= UI PANEL =======
    cv2.rectangle(frame, (0, 0), (260, h), (245, 245, 245), -1)

    # Toggle Status
    status_text = "ON" if gesture_enabled else "OFF"
    status_color = (0, 200, 0) if gesture_enabled else (0, 0, 255)

    cv2.putText(frame, f"Gesture Mode: {status_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2)

    # ======= HAND DETECTION =======
    if results.multi_hand_landmarks and gesture_enabled:
        for handLms in results.multi_hand_landmarks:

            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            if len(lmList) >= 9:

                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]

                length = math.hypot(x2 - x1, y2 - y1)
                length = max(minDistance, min(maxDistance, length))

# Calculate percentage first
                volPercent = np.interp(length,
                       [minDistance, maxDistance],
                       [0, 100])

# Smooth percentage
                volPercent = prevVol + (volPercent - prevVol) / smoothFactor
                prevVol = volPercent

# Clamp 0–100
                volPercent = max(0, min(100, volPercent))

# Set volume using SCALAR (correct method)
                try:
                    volume.SetMasterVolumeLevelScalar(volPercent / 100, None)
                except:
                    pass

# Volume bar
                volBar = np.interp(volPercent,
                   [0, 100],
                   [400, 150])

                cv2.rectangle(frame, (60, 150),
                              (100, 400),
                              (0, 255, 0), 3)

                cv2.rectangle(frame, (60, int(volBar)),
                              (100, 400),
                              (0, 255, 0), -1)

                cv2.putText(frame,
                            f'{int(volPercent)} %',
                            (50, 430),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            3)

            mp_draw.draw_landmarks(
                frame,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

    # ===== FPS =====
    cTime = time.time()
    fps = 1 / (cTime - pTime + 0.0001)
    pTime = cTime

    cv2.putText(frame,
                f'FPS: {int(fps)}',
                (450, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2)

    cv2.imshow("Hand Gesture Volume Controller", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('t'):
        gesture_enabled = not gesture_enabled

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
