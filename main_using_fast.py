import cv2

# capture = cv2.VideoCapture('day_in_the_life.mp4')
# capture = cv2.VideoCapture('driving.mp4')
capture = cv2.VideoCapture('walking_on_the_street.mp4')

if not capture.isOpened():
    print('Error while opening video.')


class FastDetector:
    def __init__(self, current_frame, threshold: int):
        self.converted_img_from_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        self.fast_instance = cv2.FastFeatureDetector_create(threshold)
        self.kp = []

    def set_kp_from_converted_img(self):
        self.kp = self.fast_instance.detect(self.converted_img_from_frame, None)

    def get_image_with_keypoints(self, color: tuple):
        return cv2.drawKeypoints(self.converted_img_from_frame, self.kp, None, color=color,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


while capture.isOpened():
    captured, frame = capture.read()
    video_width: float = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height: float = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    middle = (video_height // 2, video_width // 2)
    if captured:
        fd = FastDetector(frame, threshold=100)

        fd.set_kp_from_converted_img()

        img_with_keypoints = fd.get_image_with_keypoints((0, 255, 0))

        for keypoint in fd.kp:
            if abs(keypoint.pt[0] - middle[0]) <= 150 or abs(keypoint.pt[1] - middle[1]) <= 150:
                print('Detected object: %s' % str((keypoint.pt[0], keypoint.pt[1])))

        cv2.imshow('Seal', img_with_keypoints)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()

