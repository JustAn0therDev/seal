import cv2
import numpy as np
from matplotlib import pyplot as plt
from sys import argv

# TODO LIST:
# REFACTOR AND THEN OPTIMIZE. ITS EASIER TO OPTIMIZE WITH MORE READABLE CODE.

# The last item might help identify if an object is closer than it should be to the car.
# TODO FUTURE RUAN: Check if the keypoints are close or inside the "triangle" that composes a lane. That should do it.
# TODO FUTURE RUAN: Check if the keypoints are a bit upper to the lane; They might represent a close object at the end
# (of the lane).

capture = cv2.VideoCapture('driving.mp4')
# capture = cv2.VideoCapture('driving_lane.mp4')
# capture = cv2.VideoCapture('driving_fast.mp4')
# capture = cv2.VideoCapture('fast_bike.mp4')
lane_space_test_img = cv2.imread('lane_space_test.jpg')

if not capture.isOpened():
    print('Error while opening video.')

COMMON_THRESHOLD = 100


class FastDetector:
    def __init__(self, current_frame, threshold: int):
        self.converted_img_from_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        self.fast_instance = cv2.FastFeatureDetector_create(threshold)
        self.blurred = cv2.GaussianBlur(self.converted_img_from_frame, (5, 5), 0)
        self.canny = cv2.Canny(self.blurred, 50, 150)
        self.lines = None
        self.line_image = None
        self.polygons = None
        self.masked_image = None
        self.kp = []

    def set_kp_from_converted_img(self):
        self.kp = self.fast_instance.detect(self.converted_img_from_frame, None)

    def set_polygons(self):
        left = 25
        right = 570
        x_middle = 300
        y_middle = 110
        video_height = self.canny.shape[0]
        self.polygons = np.array([[(left, video_height), (right, video_height), (x_middle, y_middle)]])

    def set_masked_region_of_interest(self):
        mask = np.zeros_like(self.canny)
        cv2.fillPoly(mask, self.polygons, 255)
        masked_image = cv2.bitwise_and(self.canny, mask)
        self.masked_image = masked_image

    def draw_lines(self):
        self.lines = cv2.HoughLinesP(self.masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    def get_image_with_lanes(self):
        self.line_image = np.zeros_like(self.converted_img_from_frame)
        if self.lines is not None:
            for line in self.lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(self.line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

        return cv2.addWeighted(self.converted_img_from_frame, 0.8, self.line_image, 1, 0)

    def get_image_with_keypoints(self, img, color: tuple):
        return cv2.drawKeypoints(img, self.kp, None, color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def run():
    while capture.isOpened():
        captured, frame = capture.read()
        if captured:
            fd = FastDetector(np.copy(frame), threshold=COMMON_THRESHOLD)

            fd.set_polygons()
            fd.set_masked_region_of_interest()
            fd.draw_lines()
            fd.set_kp_from_converted_img()
            lanes_img = fd.get_image_with_lanes()
            img_with_keypoints = fd.get_image_with_keypoints(lanes_img, (0, 255, 0))

            """ 
                TODO: THIS SHOULD BE A METHOD IN THE FASTDETECTOR CLASS
                if (len(fd.kp)) > 0:
                for keypoint in fd.kp:
                    if abs(keypoint.pt[0] - middle[0]) <= 150 or abs(keypoint.pt[1] - middle[1]) <= 150:
                        print('Detected object: %s' % str((keypoint.pt[0], keypoint.pt[1]))) 
            """

            cv2.imshow('Seal', cv2.cvtColor(img_with_keypoints, cv2.COLOR_RGB2GRAY))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()


def test_lane():
    plt.imshow(lane_space_test_img)
    plt.show()


if len(argv) == 0 or argv[1] == 'run':
    run()
elif argv[1] == 'test':
    test_lane()