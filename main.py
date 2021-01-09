import cv2
import numpy as np
from sys import argv
from detector import Detector
from matplotlib import pyplot as plt
from utils.constants import COMMON_THRESHOLD
from utils.file_opener import FileOpener
from enums.video_type import VideoType
from enums.image_type import ImageType
from enums.camera_position import CameraPosition

# TODO:
# Panel is working fine. The only types of cameras supported will be roof and panel, since
# putting a camera in front of the hood is a really bad idea.

opened_video_capture = FileOpener.get_video_capture_by_video_type(VideoType.SLOW)
lane_space_test_img = FileOpener.get_image_by_image_type(ImageType.PANEL_TEST)

if not opened_video_capture.isOpened():
    print('Error while opening video.')

COMMON_THRESHOLD = 100


class Program:
    def __init__(self, video_capture):
        self.video_capture = video_capture
        self.detector = None

    def run_program(self, camera_position: CameraPosition):
        while self.video_capture.isOpened():
            frame_was_captured, frame = self.video_capture.read()
            if frame_was_captured:
                self.detector = Detector(np.copy(frame),
                                         threshold=COMMON_THRESHOLD,
                                         camera_position=camera_position)

                self.detector.set_polygons_representing_lane_region()
                self.detector.set_masked_image()
                self.detector.set_image_with_only_lane_lines()
                self.detector.set_keypoints_from_blurred_image()
                self.detector.set_image_with_lanes()

                # img_with_keypoints = self.detector.get_image_with_lanes_and_keypoints((255, 255, 255))
                img_with_keypoints = self.detector.get_image_with_only_keypoints()

                self.detector.print_object_detection()

                cv2.imshow('Seal', img_with_keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            else:
                break

        self.video_capture.release()
        cv2.destroyAllWindows()


def test_lane():
    plt.imshow(lane_space_test_img)
    plt.show()


if len(argv) == 0 or argv[1] == 'run':
    pos = None
    program = Program(opened_video_capture)

    if argv[2] == 'roof':
        pos = CameraPosition.ROOF
    elif argv[2] == 'panel':
        pos = CameraPosition.PANEL
    elif argv[2] == 'hood':
        pos = CameraPosition.HOOD
    else:
        raise Exception("You cannot run the program without specifying a valid camera position.")

    program.run_program(pos)
elif argv[1] == 'test':
    test_lane()
