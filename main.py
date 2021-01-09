import cv2
import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from constants import COMMON_THRESHOLD
from detector import Detector

# currently_open_video_capture = cv2.VideoCapture('driving.mp4')
currently_open_video_capture = cv2.VideoCapture('driving_lane.mp4')
# currently_open_video_capture = cv2.VideoCapture('driving_fast.mp4')
# currently_open_video_capture = cv2.VideoCapture('fast_bike.mp4')

""" 
    TODO: Check if the keypoints are close or inside the "triangle" that composes a lane. Might help
    identifying close objects.
"""

lane_space_test_img = cv2.imread('lane_space_test.jpg')

if not currently_open_video_capture.isOpened():
    print('Error while opening video.')

COMMON_THRESHOLD = 100


def run_program():
    while currently_open_video_capture.isOpened():
        frame_was_captured, frame = currently_open_video_capture.read()
        if frame_was_captured:
            detector_instance = Detector(np.copy(frame), threshold=COMMON_THRESHOLD)

            detector_instance.set_polygons()
            detector_instance.set_masked_region_of_interest()
            detector_instance.draw_lines()
            detector_instance.set_kp_from_converted_img()
            detector_instance.set_image_with_lanes()

            img_with_keypoints = detector_instance.get_image_with_lanes_and_keypoints((255, 255, 255))

            """ 
                TODO: THIS SHOULD BE A METHOD IN THE FAST DETECTOR CLASS
                if (len(fd.kp)) > 0:
                for keypoint in fd.kp:
                    if abs(keypoint.pt[0] - middle[0]) <= 150 or abs(keypoint.pt[1] - middle[1]) <= 150:
                        print('Detected object: %s' % str((keypoint.pt[0], keypoint.pt[1]))) 
            """

            # cv2.imshow('Seal', detector_instance.lane_image)
            cv2.imshow('Seal', img_with_keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            break

    currently_open_video_capture.release()
    cv2.destroyAllWindows()


def test_lane():
    plt.imshow(lane_space_test_img)
    plt.show()


if len(argv) == 0 or argv[1] == 'run':
    run_program()
elif argv[1] == 'test':
    test_lane()
