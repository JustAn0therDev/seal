"""
    WORKING MATCHER USING ORB + KNN MATCHING DETECTION
"""

import cv2
import numpy as np

capture = cv2.VideoCapture('driving.mp4')
last_frame = None

if not capture.isOpened():
    print('Error opening video.')

while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        if last_frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_from_last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

            # Initiate ORB detector
            orb = cv2.ORB_create(100)

            blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
            blurred_gray_last_frame = cv2.GaussianBlur(gray_from_last_frame, (5, 5), 0)

            # find the key points with ORB
            # kp, des = orb.detectAndCompute(gray, None)
            # kp2, des2 = orb.detectAndCompute(gray_from_last_frame, None)
            kp, des = orb.detectAndCompute(blurred_gray, None)
            kp2, des2 = orb.detectAndCompute(blurred_gray_last_frame, None)

            bf = cv2.BFMatcher()
            if des is not None and des2 is not None and len(des) == len(des2):
                matches = bf.knnMatch(des, des2, k=2)

                for keypoint in kp:
                    for keypoint2 in kp2:
                        if keypoint == keypoint2:
                            print('MATCHED')

                good = []
                for m, n in matches:
                    if m.distance < 0.3 * n.distance:
                        good.append([m])

                draw_params = dict(matchColor=(0, 255, 0),
                                   flags=2)

                img = cv2.drawMatchesKnn(gray, kp, gray_from_last_frame, kp2, good, outImg=None, **draw_params)

                cv2.imshow('Seal', img)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        last_frame = frame

capture.release()
cv2.destroyAllWindows()
