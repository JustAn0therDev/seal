from os import system

import cv2
import numpy as np

from utils.camera_axis_factory import CameraAxisFactory
from utils.constants import COMMON_THRESHOLD
from enums.camera_position import CameraPosition


class Detector:
    min_x_axis: int
    max_x_axis: int
    middle_x_axis: int
    middle_y_axis: int

    def __init__(self, current_frame, threshold: int, camera_position: CameraPosition):
        self.camera_position = camera_position
        self.converted_img_from_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        self.fast_instance = cv2.FastFeatureDetector_create(threshold)
        self.blurred = cv2.GaussianBlur(self.converted_img_from_frame, (5, 5), 0)
        self.canny = cv2.Canny(self.blurred, 50, 150)
        self.lines = None
        self.line_image = None
        self.polygons = None
        self.masked_image = None
        self.lane_image = None
        self.keypoints = []

    def set_keypoints_from_blurred_image(self):
        self.keypoints = self.fast_instance.detect(self.converted_img_from_frame, None)

    def set_polygons_representing_lane_region(self):
        positions = CameraAxisFactory.make_camera_axis_object(self.camera_position)
        video_height = self.canny.shape[0]
        self.polygons = np.array(
            [
                [(positions.min_x_axis, video_height),
                 (positions.max_x_axis, video_height),
                 (positions.middle_x_axis, positions.middle_y_axis)]
            ]
        )

        self.min_x_axis = positions.min_x_axis
        self.max_x_axis = positions.max_x_axis
        self.middle_y_axis = positions.middle_y_axis
        self.middle_x_axis = positions.middle_x_axis

    def set_masked_image(self):
        mask = np.zeros_like(self.canny)
        cv2.fillPoly(mask, self.polygons, 255)
        masked_image = cv2.bitwise_and(self.canny, mask)
        self.masked_image = masked_image

    def set_image_with_only_lane_lines(self):
        self.lines = cv2.HoughLinesP(self.masked_image,
                                     2,
                                     np.pi/180, COMMON_THRESHOLD,
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
        if (len(self.keypoints)) > 0:
            detected_count = 0
            for keypoint in self.keypoints:
                # TODO: Refactor this and separate detection for each camera positioning
                x, y = keypoint.pt
                if self.min_x_axis <= x < self.max_x_axis and y >= self.middle_y_axis:
                    # print('Detected object: %s' % str((x, y)))
                    detected_count += 1
            if self.camera_position == CameraPosition.PANEL and detected_count >= 8:
                print('CLOSE OBJECT. Detected: %i' % detected_count)
            elif self.camera_position != CameraPosition.PANEL and detected_count >= 5:
                print('CLOSE OBJECT. Detected: %i' % detected_count)
