import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# File paths
image_dir = r'C:/Users/kaviyarasan/Desktop/Jupyter/Human Faces/Data/images'
output_dir = r'C:/Users/kaviyarasan/Desktop/Jupyter/Human Faces/output'

# Load YOLO model
model_path = r'C:/Users/kaviyarasan/Desktop/Jupyter/Human Faces/yolov8n.pt'

@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

model = load_model(model_path)

# Function to load dataset
@st.cache_data
def load_data():
    file_path = r'C:/Users/kaviyarasan/Desktop/Jupyter/Human Faces/Data/faces.csv'
    data = pd.read_csv(file_path)
    return data

# Function to load performance metrics
@st.cache_data
def load_metrics():
    path_to_your_metrics = r'C:/Users/kaviyarasan/Desktop/Jupyter/Human Faces/runs/detect/train/results.csv'
    metrics = pd.read_csv(path_to_your_metrics)
    return metrics

# Function to draw bounding boxes
def draw_bounding_boxes(img, results, conf_threshold=0.5):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Convert tensor to numpy array
        confidences = result.boxes.conf.cpu().numpy()  # Convert tensor to numpy array
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Convert to integer pixel values
            confidence = confidences[i]
            
            if confidence >= conf_threshold:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                label = f'Conf: {confidence:.2f}'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Draw label
                
    return img


# Function to display samples with detections
def display_samples_with_detections(model, images_dir, samples=10):
    sample_images = os.listdir(images_dir)[:samples]
    for img_name in sample_images:
        img_path = os.path.join(images_dir, img_name)

        # Run inference
        results = model(img_path)

        img = cv2.imread(img_path)
        
        # Draw bounding boxes
        img = draw_bounding_boxes(img, results)
        
        # Display the image with bounding boxes
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        st.image(img_rgb, caption="Detected Objects", use_column_width=True)  # Display image

# Sidebar options
st.sidebar.title("Human Faces Object Detection")
menu = st.sidebar.selectbox('Select Function', ['Data', 'EDA - Visual', 'Prediction'])

# 1. Data Section
if menu == 'Data':
    st.title("Data Section")

    st.subheader("Dataset")
    data = load_data()
    st.write(data)

    st.subheader("Model Performance Metrics")
    metrics = load_metrics()
    st.write(metrics)

# 2. EDA - Visual Section
elif menu == 'EDA - Visual':
    st.title("Exploratory Data Analysis (EDA)")

    data = load_data()
    st.write("Dataset Preview", data.head())

    # EDA Plot 1 - Plotly Histogram
    st.subheader("Distribution of a Selected Feature")
    selected_feature = st.selectbox('Select a feature for plotting:', data.columns)
    fig = px.histogram(data, x=selected_feature)
    st.plotly_chart(fig)

    # EDA Plot 2 - Seaborn Boxplot
    st.subheader("Boxplot of a Selected Feature")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[selected_feature])
    st.pyplot(plt)

# 3. Prediction Section
elif menu == 'Prediction':
    st.title("Model Prediction")

    # Upload Image Section
    uploaded_image = st.file_uploader("Upload an image for object detection:", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Convert uploaded image to a file-like object for OpenCV processing
        image = Image.open(uploaded_image)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert RGB (PIL format) to BGR (OpenCV format)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform Detection on the Uploaded Image
        if st.button("Run Detection"):
            # Save the image temporarily for YOLO inference
            temp_image_path = "temp_uploaded_image.jpg"
            cv2.imwrite(temp_image_path, img_bgr)  # Save image to disk for YOLO model

            # Run inference on the uploaded image
            results = model(temp_image_path)

            # Draw bounding boxes
            img = cv2.imread(temp_image_path)
            img = draw_bounding_boxes(img, results)

            # Convert image from BGR to RGB for display in Streamlit
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display the image with bounding boxes
            st.image(img_rgb, caption="Detected Objects", use_column_width=True)

            # Remove temporary image
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

# Display the results in sidebar
st.sidebar.subheader("Sample Detections")
if st.sidebar.button("Show Sample Detections"):
    display_samples_with_detections(model, image_dir)