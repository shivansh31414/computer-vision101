import cv2
import mediapipe as mp
import time
import math
import numpy as np

class HandDetector:
    def __init__(self, mode: bool = False, maxHands: int = 2, detectionCon: float = 0.5, trackCon: float = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, img: np.ndarray, draw: bool = True) -> np.ndarray:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img: np.ndarray, handNo: int = 0, draw: bool = True):
        xList, yList = [], []
        bbox = []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            try:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

            except IndexError:
                pass  # Prevents out-of-range errors

        return self.lmList, bbox

    def fingersUp(self) -> list:
        fingers = []
        if len(self.lmList) == 0:
            return fingers  # Prevent errors when no hand is detected

        fingers.append(int(self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]))

        for i in range(1, 5):
            fingers.append(int(self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i] - 2][2]))

        return fingers

    def findDistance(self, p1: int, p2: int, img: np.ndarray, draw: bool = True, r: int = 15, t: int = 3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    cap = cv2.VideoCapture(1)
    detector = HandDetector()
    pTime = time.time()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if lmList:
            print(lmList[4])

        cTime = time.time()
        fps = max(1, int(1 / (cTime - pTime)))
        pTime = cTime

        cv2.putText(img, f'FPS: {fps}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
