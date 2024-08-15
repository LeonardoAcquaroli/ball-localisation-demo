# Ball detection
from ball_detector import BallDetector
# Pitch detection
from pitch_detector import PitchDetector
# Homography
from homography import HomographyTransformer
# Visualization
import cv2
import supervision as sv
from mplsoccer.pitch import Pitch
import matplotlib.pyplot as plt
# Typing
from typing import Tuple

class BallPositionPipeline:
    def __init__(self, ball_detector_model_path: str, pitch_detector_model_path: str) -> None:
        self.ball_detector = BallDetector(ball_detector_model_path)
        self.pitch_detector = PitchDetector(pitch_detector_model_path)
        self.homography_transformer = HomographyTransformer()

    def predict(self, input_image_path: str) -> Tuple[float, float]:
        self.image_path = input_image_path
        # Detect ball
        ball_pixels_xy = self.ball_detector.predict(self.image_path)
        # Detect pitch
        detected_keypoints, pitch_vertices = self.pitch_detector.predict(self.image_path)
        # Homography
        ball_x, ball_y = self.homography_transformer.transform_points(points=ball_pixels_xy,
                                                                      detected_keypoints=detected_keypoints,
                                                                      pitch_vertices=pitch_vertices)[0]
        if (ball_x, ball_y) == (-1, -1):
            # If (ball_x, ball_y) == (-1, -1) it means that not enough keypoints were detected
            print(f"Not enough keypoints detected.")
            return (ball_x, ball_y)
        elif (ball_x, ball_y) == (-10,-10):
            # If (ball_x, ball_y) == (-10, -10) it means that no ball was detected
            print("No ball detected.")
            return (ball_x, ball_y)
        else:
            return ball_x / 100, ball_y / 100
    
    def plot_annotated_image(self):
        # Load image
        image = cv2.imread(self.image_path)

        # Merge ball and pitch detections
        detections = sv.Detections.merge([self.ball_detector.ball_detections, self.pitch_detector.pitch_detections])

        # Annotate image
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)
        
        # Plot
        # sv.plot_image(annotated_image)
        return annotated_image
    
    def plot_radar(self, ball_x: float, ball_y: float, pitch_length: int = 105, pitch_width: int = 68, line_color: str = 'white', pitch_color: str = 'green', corner_arcs: bool = True):

        # Create a custom pitch object
        pitch = Pitch(
            pitch_type='custom',
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            line_color=line_color,
            line_zorder=0,
            pitch_color=pitch_color,
            corner_arcs=corner_arcs
        )

        # Plot the pitch
        fig, ax = pitch.draw()
        ax.invert_yaxis()
        ax.plot(ball_x, ball_y, 'ro', zorder=1)

        # Add note with ball coordinates
        ax.annotate(f'({round(ball_x, 1)}, {round(ball_y, 1)})', (ball_x, ball_y), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

        return  fig, ax