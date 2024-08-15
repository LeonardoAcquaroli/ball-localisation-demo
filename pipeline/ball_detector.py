from ultralytics import YOLO
import supervision as sv
import numpy as np

class BallDetector:
    """
    The BallDetector class is designed to detect a soccer ball in an image using a YOLO-based model.
    It processes the detections to find the pixel coordinates of the detected ball.

    Attributes:
        ball_detector (YOLO): A YOLO model instance for detecting the soccer ball in images.
        ball_detections: sv.Detections: A Supervision Detections object containing the detection results. 
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the BallDetector with the path to a pre-trained YOLO model.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        if not ('ball_detector' in globals()):
            self.ball_detector = YOLO(model_path)
        else:
            print('Pitch detection model already loaded')

    def get_detections(self, image_path: str) -> sv.Detections:
        """
        Runs the YOLO model on the given image and returns the detections for the soccer ball.

        Args:
            image_path (str): Path to the image file on which to run the detection.

        Returns:
            sv.Detections: A Supervision Detections object containing the detection results.
        """
        ball = self.ball_detector.predict(image_path)
        ball_detections = sv.Detections.from_ultralytics(ball[0])
        return ball_detections
    
    def get_ball_pixels_xy(self, ball_detections: sv.Detections) -> np.ndarray:
        """
        Extracts the pixel coordinates of the ball as the centre of the detection bounding box.

        Args:
            ball_detections (sv.Detections): The detection results from which to extract the ball's coordinates.

        Returns:
            np.ndarray: A 1D array containing the (x, y) pixel coordinates of the ball's center.
                        If no ball is detected, an empty array is returned.
        """
        if len(ball_detections) == 0:
            print("No ball detected")
            # raise ValueError("No ball detected")
        else:
            # Calculate the center of the bounding box for the first detection
            ball_pixels_xy = np.array([
                (ball_detections.xyxy[0][0] + ball_detections.xyxy[0][2]) / 2,  # X-coordinate
                (ball_detections.xyxy[0][1] + ball_detections.xyxy[0][3]) / 2   # Y-coordinate
            ])
            return ball_pixels_xy
    
    def predict(self, image_path: str) -> np.ndarray:
        """
        Runs the ball detection steps, returning the pixel coordinates of the detected soccer ball.

        Args:
            image_path (str): Path to the image file on which to perform the detection.

        Returns:
            np.ndarray: A 1D array containing the (x, y) pixel coordinates of the ball's center.
        """
        self.ball_detections = self.get_detections(image_path)
        ball_pixels_xy = self.get_ball_pixels_xy(self.ball_detections)

        if ball_pixels_xy is not None:
            return ball_pixels_xy
        else:
            return np.array([-10, -10])   
