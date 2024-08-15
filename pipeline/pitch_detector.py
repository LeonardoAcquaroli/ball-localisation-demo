from ultralytics import YOLO
import supervision as sv
import numpy as np
# Typing
from typing import Tuple
# Pitch configuration
from pitch_config import SoccerFieldConfiguration
PITCH_CONFIG = SoccerFieldConfiguration()

class PitchDetector:
    """
    The PitchDetector class is responsible for detecting soccer pitch keypoints from an input image.
    It uses a YOLO-based model to perform detections and processes these detections to extract keypoints
    and vertices of the soccer field.

    Attributes:
        pitch_detector (YOLO): A YOLO model instance for detecting keypoints on the soccer pitch.
        pitch (list): A list containing detection results from the YOLO model. It is a list with only one element because the prediction is made on one frame.
        pitch_detections (sv.Detections): A Supervision Detections object containing the pitch vertices detections.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the PitchDetector with the path to a pre-trained YOLO model.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        if not ('pitch_detector' in globals()):
            self.pitch_detector = YOLO(model_path)
        else:
            print('Pitch detection model already loaded')

    def get_detections(self, image_path: str) -> sv.Detections:
        """
        Runs the YOLO model on the given image and returns the detections.

        Args:
            image_path (str): Path to the image file on which to run the detection.

        Returns:
            sv.Detections: A Supervision Detections object containing the detection results.
        """
        self.pitch = self.pitch_detector.predict(image_path)
        pitch_detections = sv.Detections.from_ultralytics(self.pitch[0])
        return pitch_detections

    def get_detected_keypoints(self, pitch_detections: sv.Detections) -> np.ndarray:
        """
        Extracts and returns the detected keypoints from the pitch detections.

        Args:
            pitch_detections (sv.Detections): The detection results from which to extract keypoints.

        Returns:
            np.ndarray: An array of detected keypoints. If fewer than 4 keypoints are detected, raises a ValueError.
        """
        if len(pitch_detections) < 4:
            print(f"Not enough keypoints detected: detected {len(pitch_detections)}, needed at least 4.")
            # raise ValueError(f"Not enough keypoints detected: detected {len(pitch_detections)}, needed at least 4.")
        else:
            pitch_keypoints = self.pitch[0].keypoints.xy.reshape(len(pitch_detections), 2)
            return np.array(pitch_keypoints)

    def get_pitch_vertices(self) -> np.ndarray:
        """
        Retrieves the vertices of the soccer pitch based on detected keypoints.

        Returns:
            np.ndarray: An array of the vertices' coordinates corresponding to the detected keypoints.
        """
        # Update the keypoints with the class ids from the box detections
        self.pitch[0].keypoints.cls = self.pitch[0].boxes.cls
        keypoints_ids = self.pitch[0].keypoints.cls.tolist()

        # Filter and retrieve the vertex coordinates using the detected keypoints' ids
        keypoints_xy = [PITCH_CONFIG.vertices[id]['xy'] for id in keypoints_ids]

        return np.array(keypoints_xy)

    def predict(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the pitch detection steps, returning both the detected keypoints and the corresponding 2D vertices coordinates.

        Args:
            image_path (str): Path to the image file on which to perform the detection.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - Detected keypoints (np.ndarray)
                - Corresponding 2D pitch vertices (np.ndarray)
        """
        self.pitch_detections = self.get_detections(image_path)
        pitch_keypoints = self.get_detected_keypoints(self.pitch_detections)
        pitch_vertices = self.get_pitch_vertices()

        return pitch_keypoints, pitch_vertices