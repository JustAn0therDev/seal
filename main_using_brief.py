import cv2

# capture = cv2.VideoCapture('day_in_the_life.mp4')
# capture = cv2.VideoCapture('driving.mp4')
capture = cv2.VideoCapture('walking_on_the_street.mp4')

if not capture.isOpened():
    print('Error opening video.')

while capture.isOpened():
    ret, frame = capture.read()

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initiate FAST detector
        star = cv2.xfeatures2d.StarDetector_create(25)
        
        # Initiate BRIEF extractor
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # find the keypoints with STAR
        kp = star.detect(gray, None)

        # compute the descriptors with BRIEF
        kp, des = brief.compute(gray, kp)

        # print(get_biggest_keypoint(kp))

        print(len(kp))

        # draw key points detected
        img = cv2.drawKeypoints(gray, kp, gray,color=(0, 255, 0))

        cv2.imshow('Seal', img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    continue

capture.release()

cv2.destroyAllWindows()
