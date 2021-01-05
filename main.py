import cv2

capture = cv2.VideoCapture('driving.mp4')

if not capture.isOpened():
    print('Error opening video.')

while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        # biggest orb found
        biggest: float = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect key feature points
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        # get biggest orb
        for keypoint in kp:
            if keypoint.size > biggest:
                biggest = keypoint.size

        # print biggest orb found if more than 130
        if biggest > 115:
            print("Large orb detected! Size: " + str(biggest))

        # draw key points detected
        img = cv2.drawKeypoints(gray, kp, gray, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('Seal', img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    continue

capture.release()
cv2.destroyAllWindows()
