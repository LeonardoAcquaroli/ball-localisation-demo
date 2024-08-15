from typing import List, Tuple
from constants import PitchDetectionConstants
CLASS_MAPPING = PitchDetectionConstants.CLASS_MAPPING

class SoccerFieldConfiguration:

    width: int = 6800  # [cm]
    length: int = 10500  # [cm]
    penalty_box_width: int = 4032  # [cm]
    penalty_box_length: int = 1650  # [cm]
    goal_box_width: int = 1832  # [cm]
    goal_box_length: int = 550  # [cm]
    centre_circle_radius: int = 915  # [cm]
    penalty_spot_distance: int = 1100  # [cm]

    @property
    def keypoints_xy(self) -> List[Tuple[int, int]]:
        return [
            # Defensive
            (0, 0),  # Piotr: 1 | Ours: 0, D1
            (0, (self.width - self.penalty_box_width) / 2),  # Piotr: 2 | Ours: 4, D2
            (0, (self.width - self.goal_box_width) / 2),  # Piotr: 3 | Ours: 5, D3
            (0, (self.width + self.goal_box_width) / 2),  # Piotr: 4 | Ours: 6, D4
            (0, (self.width + self.penalty_box_width) / 2),  # Piotr: 5 | Ours: 7, D5
            (0, self.width),  # Piotr: 6 | Ours: 8, D6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # Piotr: 7 | Ours: 9, D7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # Piotr: 8 | Ours: 10, D8
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # Piotr: 10 | Ours: 11, D9
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # Piotr: 11 | Ours: 1, D10
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # Piotr: 12 | Ours: 2, D11
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # Piotr: 13 | Ours: 3, D12
            # Midfield
            (self.length / 2, 0),  # Piotr: 14 | Ours: 12, M1
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # Piotr: 15 | Ours: 13, M2
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # Piotr: 16 | Ours: 14, M3
            (self.length / 2, self.width),  # Piotr: 17 | Ours: 15, M4
            # Offensive
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # Piotr: 18 | Ours: 16, O1
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # Piotr: 19 | Ours: 20, O2
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # Piotr: 20 | Ours: 21, O3
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # Piotr: 21 | Ours: 22, O4
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),  # Piotr: 23 | Ours: 23, O5
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),  # Piotr: 24 | Ours: 24, O6
            (self.length, 0),  # Piotr: 25 | Ours: 25, O7
            (self.length, (self.width - self.penalty_box_width) / 2),  # Piotr: 26 | Ours: 26, O8
            (self.length, (self.width - self.goal_box_width) / 2),  # Piotr: 27 | Ours: 27, O9
            (self.length, (self.width + self.goal_box_width) / 2),  # Piotr: 28 | Ours: 17, O10
            (self.length, (self.width + self.penalty_box_width) / 2),  # Piotr: 29 | Ours: 18, O11
            (self.length, self.width),  # Piotr: 30 | Ours: 19, O12
        ]
    
    @property
    def labels(self) -> List[str]:
        return list(CLASS_MAPPING.keys())

    @property
    def ids(self) -> List[int]: 
        return list(CLASS_MAPPING.values())
    
    @property
    def vertices(self) -> dict:
        vertices = {}
        for id, label, xy in zip(self.ids, self.labels, self.keypoints_xy):
            vertices[id] = {"label": label, "xy": xy}
        return vertices