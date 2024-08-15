from typing import Tuple
import cv2
import numpy as np
import numpy.typing as npt


class HomographyTransformer:
    def __init__(self) -> None:
        pass

    
    def get_homography_matrix(self,
                              detected_keypoints: npt.NDArray[np.float32],
                              pitch_vertices: npt.NDArray[np.float32]) -> np.ndarray:
        """
        Calculate the Homography matrix using detected keypoints and pitch vertices.

        Args:
            detected_keypoints (npt.NDArray[np.float32]): Source points for homography calculation.
            pitch_vertices (npt.NDArray[np.float32]): Target points for homography calculation.

        Returns:
            np.ndarray: Homography matrix.
        Raises:
            ValueError: If detected_keypoints and pitch_vertices do not have the same shape or if they are
                not 2D coordinates.
        """
        if detected_keypoints.shape != pitch_vertices.shape:
            raise ValueError("Source and pitch_vertices must have the same shape.")
        if detected_keypoints.shape[1] != 2:
            raise ValueError("Source and pitch_vertices points must be 2D coordinates.")

        detected_keypoints = detected_keypoints.astype(np.float32)
        pitch_vertices = pitch_vertices.astype(np.float32)
        H, _ = cv2.findHomography(detected_keypoints, pitch_vertices)
        if H is None:
            print("Homography matrix could not be calculated.")
            # raise ValueError("Homography matrix could not be calculated.")
        else:
            return H

    def transform_points(self,
                         points: npt.NDArray[np.float32],
                         detected_keypoints: npt.NDArray[np.float32],
                         pitch_vertices: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Transform the given points using the homography matrix.

        Args:
            points (npt.NDArray[np.float32]): Points to be transformed.

        Returns:
            npt.NDArray[np.float32]: Transformed points.

        Raises:
            ValueError: If points are not 2D coordinates.
        """
        if np.array_equal(points, np.array([-10, -10])):
            print("No ball detected")
            return [(-10, -10)]

        if points.shape[0] != 2:
            raise ValueError("Points must be 2D coordinates.")

        # Homography matrix
        if not (detected_keypoints is None):
            H = self.get_homography_matrix(detected_keypoints, pitch_vertices)
            if H is None:
                return [(-100, -100)]
            reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(reshaped_points, H)
            return transformed_points.reshape(-1, 2).astype(np.float32)
        else:
            # Not enough keypoints (no keypoints at all actually)
            return [(-1, -1)]

    def transform_image(
            self,
            image: npt.NDArray[np.uint8],
            resolution_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        """
        Transform the given image using the homography matrix.

        Args:
            image (npt.NDArray[np.uint8]): Image to be transformed.
            resolution_wh (Tuple[int, int]): Width and height of the output image.

        Returns:
            npt.NDArray[np.uint8]: Transformed image.

        Raises:
            ValueError: If the image is not either grayscale or color.
        """
        if len(image.shape) not in {2, 3}:
            print("Image must be either grayscale or color.")
            # raise ValueError("Image must be either grayscale or color.")
        return cv2.warpPerspective(image, self.H, resolution_wh)