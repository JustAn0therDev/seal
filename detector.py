import cv2
import numpy as np
from factories.camera_axis_factory import CameraAxisFactory
from enums.camera_position import CameraPosition
from strategies.object_distance_detector_strategy import ObjectDistanceDetectorStrategy


class Detector:
    def __init__(self, current_frame, threshold: int, camera_position: CameraPosition):
        self.threshold = threshold
        self.camera_position = camera_position
        self.converted_img_from_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        self.fast_instance = cv2.FastFeatureDetector_create(self.threshold)
        self.blurred = cv2.GaussianBlur(self.converted_img_from_frame, (5, 5), 0)
        self.canny = cv2.Canny(self.blurred, 50, 150)
        self.camera_axis = None
        self.lines = None
        self.line_image = None
        self.polygons = None
        self.masked_image = None
        self.lane_image = None
        self.keypoints = []

    def set_keypoints_from_blurred_image(self):
        self.keypoints = self.fast_instance.detect(self.converted_img_from_frame, None)

    def set_polygons_representing_lane_region(self):
        self.camera_axis = CameraAxisFactory.create_camera_axis_object(self.camera_position)
        video_height = self.canny.shape[0]
        self.polygons = np.array(
            [
                [(self.camera_axis.min_x_axis, video_height),
                 (self.camera_axis.max_x_axis, video_height),
                 (self.camera_axis.middle_x_axis, self.camera_axis.middle_y_axis)]
            ]
        )

    def set_masked_image(self):
        mask = np.zeros_like(self.canny)
        cv2.fillPoly(mask, self.polygons, 255)
        masked_image = cv2.bitwise_and(self.canny, mask)
        self.masked_image = masked_image

    def set_image_with_only_lane_lines(self):
        self.lines = cv2.HoughLinesP(self.masked_image,
                                     2,
                                     np.pi/180, self.threshold,
                                     np.array([]),
                                     minLineLength=40,
                                     maxLineGap=5)

    def set_image_with_lanes(self):
        self.line_image = np.zeros_like(self.converted_img_from_frame)
        if self.lines is not None:
            for line in self.lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(self.line_image, (x1, y1), (x2, y2), (255, 255, 255), 10)

        self.lane_image = cv2.addWeighted(self.converted_img_from_frame, 0.8, self.line_image, 1, 0)

    def get_image_with_only_keypoints(self):
        return cv2.drawKeypoints(self.converted_img_from_frame,
                                 self.keypoints,
                                 None,
                                 (255, 255, 255),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def get_image_with_lanes_and_keypoints(self, color: tuple):
        return cv2.drawKeypoints(self.lane_image,
                                 self.keypoints,
                                 None,
                                 color,
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def print_object_detection(self):
        ObjectDistanceDetectorStrategy(self.camera_position, self.camera_axis, self.keypoints).execute_strategy()
