import cv2
import time
import os
import handtracking_module as htm

# Define webcam settings
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Load overlay images safely
folderPath = "New folder"
if not os.path.exists(folderPath):
    raise FileNotFoundError(f"Folder '{folderPath}' not found!")

my_List = os.listdir(folderPath)
overlayList = [cv2.imread(f"{folderPath}/{imPath}") for imPath in my_List if
               cv2.imread(f"{folderPath}/{imPath}") is not None]

if not overlayList:
    raise ValueError("No valid images found in the folder!")

pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Failed to read frame!")
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=True)  # Enable drawing for visualization

    if lmList:
        fingers = []
        # Thumb
        fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0)

        # Four fingers
        fingers.extend(
            1 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 0 for id in range(1, 5)
        )

        totalFingers = fingers.count(1)
        print(f"Detected Fingers: {totalFingers}")

        # Safely overlay images with resizing
        if 0 < totalFingers <= len(overlayList):
            overlay_resized = cv2.resize(overlayList[totalFingers - 1], (wCam, hCam))  # Resize overlay
            img[0:hCam, 0:wCam] = overlay_resized

        # Display finger count box
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    # FPS Calculation
    cTime = time.time()
    fps = round(1 / (cTime - pTime), 2) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {fps}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
