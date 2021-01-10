import cv2
from enums.image_type import ImageType


class ImageObjectFactory:
    __dictionary_of_types_and_images = {
        ImageType.LANE_TEST: cv2.imread('lane_space_test.jpg'),
        ImageType.PANEL_TEST: cv2.imread('panel_test.jpg'),
        ImageType.ROOF_TEST: cv2.imread('roof_test.jpg'),
        ImageType.PARKING_TEST: cv2.imread('parking_test.jpg')
    }

    @staticmethod
    def create_image_object(type: ImageType):
        return ImageObjectFactory.__dictionary_of_types_and_images[type]
