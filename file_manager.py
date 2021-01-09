import cv2

"""
    TODO: get_video_capture function should receive an ENUM argument and
    return a video based on the enum:
    SLOW = 0
    NORMAL = 1
    FAST = 2
    BIKE = 3
    OTHER = 4
    
    The same logic applies to image (changing test image).    
"""


def get_video_capture(video_name: str):
    return cv2.videoCapture(video_name)


def get_image(image_name: str):
    return cv2.imread(image_name)
