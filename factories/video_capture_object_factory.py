import cv2
from enums.video_type import VideoType


class VideoCaptureObjectFactory:
    __dictionary_of_types_and_videos = {
        VideoType.SLOW: cv2.VideoCapture('driving.mp4'),
        VideoType.NORMAL: cv2.VideoCapture('driving_lane.mp4'),
        VideoType.FAST: cv2.VideoCapture('driving_fast.mp4'),
        VideoType.PARKING: cv2.VideoCapture('automated_parking.mp4')
    }

    @staticmethod
    def create_video_capture_object(type: VideoType):
        return VideoCaptureObjectFactory.__dictionary_of_types_and_videos[type]
