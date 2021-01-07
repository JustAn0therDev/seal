import cv2
import numpy as np

# TODO LIST:
# Create a mask image
# Detect the lines to draw
# Get mask image with drawn lines
# Draw lines
# Sum the binary content of both images (masked and grayscale) into one image to display the drawn lanes + keypoints.

# The last item might help identify if an object is closer than it should be to the car.
# TODO FUTURE RUAN: Check if the keypoints are close or inside the "triangle" that composes a lane. That should do it.
# TODO FUTURE RUAN: Check if the keypoints are a bit upper to the lane; They might represent a close object at the end.
# TODO FUTURE RUAN: Computational Lane Detection -> https://www.youtube.com/watch?v=eLTLtUVuuy4&ab_channel=ProgrammingKnowledge

# capture = cv2.VideoCapture('day_in_the_life.mp4')
capture = cv2.VideoCapture('driving.mp4')
# capture = cv2.VideoCapture('walking_on_the_street.mp4')

if not capture.isOpened():
    print('Error while opening video.')

COMMON_THRESHOLD = 100


class FastDetector:
    def __init__(self, current_frame, threshold: int):
        self.converted_img_from_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        self.fast_instance = cv2.FastFeatureDetector_create(threshold)
        self.blurred = cv2.GaussianBlur(self.converted_img_from_frame, (5, 5), 0)
        self.canny = cv2.Canny(self.blurred, 50, 150)
        self.lines = None
        self.kp = []

    def set_kp_from_converted_img(self):
        self.kp = self.fast_instance.detect(self.converted_img_from_frame, None)

    def draw_lines(self):
        # TODO: pass in the cropped_image, not canny
        self.lines = cv2.HoughLinesP(self.canny, 2, np.pi/180,
                                     COMMON_THRESHOLD,
                                     np.array([]),
                                     minLineLength=40,
                                     maxLineGap=5)

    def get_image_with_keypoints(self, color: tuple, run_canny: bool):
        if run_canny:
            return cv2.drawKeypoints(self.canny, self.kp, None, color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return cv2.drawKeypoints(self.blurred, self.kp, None, color=color,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


while capture.isOpened():
    captured, frame = capture.read()
    video_width: float = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height: float = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    middle = (video_height // 2, video_width // 2)
    if captured:
        fd = FastDetector(frame, threshold=COMMON_THRESHOLD)

        fd.set_kp_from_converted_img()

        img_with_keypoints = fd.get_image_with_keypoints((0, 255, 0), run_canny=True)

        for keypoint in fd.kp:
            if abs(keypoint.pt[0] - middle[0]) <= 150 or abs(keypoint.pt[1] - middle[1]) <= 150:
                print('Detected object: %s' % str((keypoint.pt[0], keypoint.pt[1])))

        cv2.imshow('Seal', img_with_keypoints)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()

