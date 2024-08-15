import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import BallPositionPipeline

st.title("Automatic Ball Position - DEMO")

st.write("Upload a frame of a soccer match")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   # Read the image in BGR format
    image_bgr = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Display the uploaded image in RGB format
    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Get the local file path
    image_path = "temp_image.jpg"

    # Instantiate the pipeline
    pipeline = BallPositionPipeline(
        ball_detector_model_path=r'C:\Users\leoac\vtg-automation\ball_position_estimation\models\ball_detector_yolov10m_ultralytics=8.2.71.pt',
        pitch_detector_model_path=r'C:\Users\leoac\vtg-automation\ball_position_estimation\models\pitch_detector_YOLOv8x-pose.pt'
    )

    col1, col2 = st.columns(2)
    
    with col1:
        annotated_image_button = st.checkbox("Plot annotated image")
    with col2:
        radar_button = st.checkbox("Plot radar")
    predict_button = st.button("Extract ball position")

    if predict_button:
        # Predict using the pipeline
        with st.spinner("Extracting ball position..."): # Spinner while the pipeline is processing
            ball_x, ball_y = pipeline.predict(image_path)

        if (ball_x, ball_y) == (-10,-10):
            st.error("No ball detected.")
        elif (ball_x, ball_y) == (-1,-1):
            st.error("Not enough keypoints detected.")
        elif (ball_x, ball_y) == (-100, -100):
            st.error("Problems obtaining the homography matrix.")
        elif (ball_x > 105) or (ball_x < 0) or (ball_y > 68) or (ball_y < 0):
            st.write(f"Ball position: ({round(ball_x, 1)}, {round(ball_y, 1)})")
            st.error("Ball out of radar.")
        else:
            st.write(f"Ball position: ({round(ball_x, 1)}, {round(ball_y, 1)})")

        if annotated_image_button:
            annotated_image = pipeline.plot_annotated_image()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image, "Annotated image")

        if radar_button:
            fig2, ax2 = pipeline.plot_radar(ball_x, ball_y, pitch_length=105, pitch_width=68)
            ax2.set_xlabel("Radar plot", fontsize=16)
            st.pyplot(fig2)

    # Clean up: remove the temporary file
    import os
    os.remove(image_path)