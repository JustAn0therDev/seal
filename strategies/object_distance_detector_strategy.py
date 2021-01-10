from auxiliary_classes.camera_axis import CameraAxis
from enums.camera_position import CameraPosition
from strategies.distance_strategy import DistanceStrategy
from strategies.panel_distance_strategy import PanelDistanceStrategy
from strategies.roof_distance_strategy import RoofDistanceStrategy


class ObjectDistanceDetectorStrategy:
    __strategy: DistanceStrategy = None

    def initialize_strategy_object(self):
        if self.camera_position == CameraPosition.ROOF:
            self.__strategy = RoofDistanceStrategy()
        elif self.camera_position == CameraPosition.PANEL:
            self.__strategy = PanelDistanceStrategy()
        else:
            raise Exception("Invalid camera position for strategy.")

    def __init__(self, camera_position: CameraPosition, camera_axis: CameraAxis, keypoints: tuple):
        self.camera_position = camera_position
        self.camera_axis = camera_axis
        self.keypoints = keypoints
        self.initialize_strategy_object()

    def execute_strategy(self):
        self.__strategy.execute(self.camera_axis, self.keypoints)
