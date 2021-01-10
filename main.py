import cv2
import numpy as np
from sys import argv
from detector import Detector
from matplotlib import pyplot as plt
from enums.image_type import ImageType
from enums.video_type import VideoType
from factories.image_object_factory import ImageObjectFactory
from factories.video_capture_object_factory import VideoCaptureObjectFactory
from utils.constants import COMMON_THRESHOLD
from enums.camera_position import CameraPosition

opened_video_capture = VideoCaptureObjectFactory().create_video_capture_object(VideoType.FAST)
test_img = ImageObjectFactory().create_image_object(ImageType.LANE_TEST)

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
                self.detector.set_keypoints_from_blurred_image()
                self.detector.set_image_with_lanes()

                img_with_keypoints = self.detector.get_image_with_only_keypoints()

                self.detector.print_object_detection()

                cv2.imshow('Seal', img_with_keypoints)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            else:
                break

        self.video_capture.release()
        cv2.destroyAllWindows()


def test_lane():
    plt.imshow(test_img)
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
    elif argv[2].lower() == 'rearview':
        pos = CameraPosition.REAR_VIEW
    else:
        raise Exception("You cannot run the program without specifying a valid camera position.")

    program.run_program(pos)
elif argv[1] == 'test':
    test_lane()
