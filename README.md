# Waste-Segregation-using-Deep-Learning
This project develops an AI-powered system to classify waste into categories. Using MobileNetV2, it predicts classes with confidence scores and provides recycling advice, integrated with a Flask API and Streamlit dashboard for real-time interaction.

# Objectives
Analyze and classify garbage images using deep learning techniques

Preprocess and augment image data for improved model performance

Build a classification model using MobileNetV2 (transfer learning)

Generate predictions with confidence scores

Provide recycling and disposal recommendations based on category

Deploy the system using a Flask API for real-time predictions

Develop an interactive dashboard using Streamlit for visualization

# Key Concepts Used
Convolutional Neural Networks (CNN)

Transfer Learning (MobileNetV2)

Image Preprocessing & Augmentation

Multi-class Classification

Softmax Probability & Confidence Scores

Model Deployment (Flask API)

Interactive Visualization (Streamlit)

Real-time Prediction Systems

# Tools & Technologies
Language:

Python

Libraries:

TensorFlow / Keras

NumPy

OpenCV

Pillow (PIL)

Matplotlib

Flask

Streamlit

# Project Structure
AI-Garbage-Classification/
│
│
├── src/
│   ├── train_model.py              # Model training pipeline
│   ├── app.py                      # Flask API for prediction
│   └── dashboard.py                # Streamlit dashboard
│
├── reports/
│   └── project_report/             # Academic report (PDF/LaTeX)
│
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
├── LICENSE                        # MIT License
└── .gitignore
