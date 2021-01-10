from enums.camera_position import CameraPosition
from auxiliary_classes.camera_axis import CameraAxis


class CameraAxisFactory:
    @staticmethod
    def make_camera_axis_object(camera_position: CameraPosition):
        dictionary_of_camera_positions_and_axis = {
            CameraPosition.ROOF: CameraAxis(min_x_axis=25, max_x_axis=570, middle_x_axis=300, middle_y_axis=110),
            CameraPosition.PANEL: CameraAxis(min_x_axis=25, max_x_axis=570, middle_x_axis=300, middle_y_axis=170),
        }

        return dictionary_of_camera_positions_and_axis[camera_position]
