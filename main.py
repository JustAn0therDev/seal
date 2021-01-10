import cv2
import numpy as np
from sys import argv
from detector import Detector
from matplotlib import pyplot as plt
from utils.file_opener import FileOpener
from enums.image_type import ImageType
from enums.video_type import VideoType
from utils.constants import COMMON_THRESHOLD
from enums.camera_position import CameraPosition

# TODO: The only types of cameras supported will be roof and panel;
# TODO: Object detection should exist separately for each type of camera position;
# TODO: Object detection should be separated in it's own class; and
# TODO: Object detection should be implemented using strategy pattern (same data, different algorithm).

opened_video_capture = FileOpener.get_video_capture_by_video_type(VideoType.SLOW)
lane_space_test_img = FileOpener.get_image_by_image_type(ImageType.PANEL_TEST)

if not opened_video_capture.isOpened():
    print('Error while opening video.')


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
    program = Program(opened_video_capture)

    if len(argv) <= 2:
        raise Exception(
            "You must provide a camera position argument. "
            "Choose either 'roof' or 'panel', depending on the camera position"
        )

    if argv[2].lower() == 'roof':
        pos = CameraPosition.ROOF
    elif argv[2].lower() == 'panel':
        pos = CameraPosition.PANEL
    else:
        raise Exception("You cannot run the program without specifying a valid camera position.")

    program.run_program(pos)
elif argv[1] == 'test':
    test_lane()
