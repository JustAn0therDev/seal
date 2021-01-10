import cv2
from enums.video_type import VideoType
from enums.image_type import ImageType


class FileOpener:
    __dictionary_of_types_and_videos = {
        VideoType.SLOW: cv2.VideoCapture('driving.mp4'),
        VideoType.NORMAL: cv2.VideoCapture('driving_lane.mp4'),
        VideoType.FAST: cv2.VideoCapture('driving_fast.mp4'),
        VideoType.PARKING: cv2.VideoCapture('automated_parking.mp4')
    }

    __dictionary_of_types_and_images = {
        ImageType.LANE_TEST: cv2.imread('lane_space_test.jpg'),
        ImageType.PANEL_TEST: cv2.imread('panel_test.jpg'),
        ImageType.ROOF_TEST: cv2.imread('roof_test.jpg'),
        ImageType.PARKING_TEST: cv2.imread('parking_test.jpg')
    }

    @staticmethod
    def get_video_capture_by_video_type(chosen_video_type: VideoType):
        """ Receives a VideoType argument and returns a video capture of that type """
        return FileOpener.__dictionary_of_types_and_videos[chosen_video_type]

    @staticmethod
    def get_image_by_image_type(chosen_image_type: ImageType):
        """ Receives an ImageType argument and returns an image of that type"""
        return FileOpener.__dictionary_of_types_and_images[chosen_image_type]
