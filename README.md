# Human Pose Detection using OpenCV DNN

## 📌 Overview

This project implements 2D human pose detection using OpenCV's Deep Neural Network (DNN) module and a pre-trained TensorFlow pose estimation model.

The system detects human body keypoints from an input image and connects them to form a skeletal representation of the detected pose.

## 🎯 Objectives

- Detect human body keypoints from images.
- Identify 18 major body joints.
- Generate a skeletal representation by connecting detected keypoints.
- Use a pre-trained deep learning model for pose estimation.
- Implement the system using Python and OpenCV DNN.

## 🧠 How It Works

The system follows these steps:

1. Load the pre-trained TensorFlow pose model.
2. Read the input image.
3. Pre-process the image and create a neural-network input blob.
4. Pass the image through the pose estimation model.
5. Generate heatmaps for the body keypoints.
6. Identify the most confident location for each keypoint.
7. Apply a confidence threshold.
8. Connect detected keypoints using predefined pose pairs.
9. Display the final skeletal pose.

## 🔑 Body Keypoints

The model detects 18 body keypoints:

- Nose
- Neck
- Right Shoulder
- Right Elbow
- Right Wrist
- Left Shoulder
- Left Elbow
- Left Wrist
- Right Hip
- Right Knee
- Right Ankle
- Left Hip
- Left Knee
- Left Ankle
- Right Eye
- Left Eye
- Right Ear
- Left Ear

A background class is also included by the model.

## 🛠️ Technologies Used

- Python
- OpenCV
- OpenCV DNN
- TensorFlow pre-trained pose model
- NumPy
- Google Colab
- Jupyter Notebook

## ⚙️ Configuration

The current implementation uses:

| Parameter | Value |
|---|---|
| Input Size | 368 × 368 |
| Confidence Threshold | 0.20 |
| Number of Body Keypoints | 18 |
| Model Format | TensorFlow Frozen Graph |

## 🚀 Running the Project

### 1. Open the notebook

Open:

`human_pose_detection.ipynb`

in Google Colab.

### 2. Download the model

Download the required TensorFlow model file:

`graph_opt.pb`

and place it in:

`/content/graph_opt.pb`

### 3. Provide an input image

Place the input image at:

`/content/20180729_180105.jpg`

### 4. Run the notebook

Execute the cells in order.

The resulting image will contain the detected body keypoints and skeletal connections.

## 📂 Project Structure

```text
human-pose-detection-opencv/
│
├── README.md
├── human_pose_detection.ipynb
└── human_pose_detection.py
