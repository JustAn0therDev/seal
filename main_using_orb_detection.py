import cv2
# import numpy as np

capture = cv2.VideoCapture('driving.mp4')

if not capture.isOpened():
    print('Error opening video.')

while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Initiate ORB detector
        orb = cv2.ORB_create(100)
        # find the key points with ORB
        kp = orb.detect(gray, None)
        # compute the descriptors with ORB
        kp, des = orb.compute(gray, kp)

        # draw key points detected
        img = cv2.drawKeypoints(gray, kp, gray, color=(0, 255, 0))

        cv2.imshow('Seal', img)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    continue

capture.release()
cv2.destroyAllWindows()
