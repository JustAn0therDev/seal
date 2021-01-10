from auxiliary_classes.camera_axis import CameraAxis
from strategies.distance_strategy import DistanceStrategy


class RearViewDistanceStrategy(DistanceStrategy):
    def execute(self, camera_axis: CameraAxis, keypoints: tuple):
        if (len(keypoints)) > 0:
            detected_count = 0
            for keypoint in keypoints:
                x, y = keypoint.pt
                if camera_axis.min_x_axis <= x < camera_axis.max_x_axis and y >= camera_axis.middle_y_axis:
                    detected_count += 1
            if detected_count >= 15:
                print('CLOSE OBJECT. Detected: %i' % detected_count)
