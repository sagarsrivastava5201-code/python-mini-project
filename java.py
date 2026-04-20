import cv2
import mediapipe as mp
import numpy as np
import math
import screen_brightness_control as sbc
import pyautogui

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands()
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

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            # Thumb tip (4) & Index tip (8)
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # Draw points
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Distance
            length = math.hypot(x2 - x1, y2 - y1)

            # 🔊 Volume control (keyboard based)
            if length < 50:
                pyautogui.press("volumedown")
            elif length > 150:
                pyautogui.press("volumeup")

            # 💡 Brightness control
            brightness = np.interp(length, [20, 200], [0, 100])
            sbc.set_brightness(int(brightness))

            # Display
            cv2.putText(img, f'Brightness: {int(brightness)}%',
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 3)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()