# 🫁 AI-Assisted Pneumonia Screening System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)

An AI-powered diagnostic support tool designed to assist medical professionals in detecting **Pneumonia** from chest X-ray images.
This project utilizes **Transfer Learning** with a **ResNet18** architecture and provides a user-friendly web interface via **Streamlit**.

---

## 🚀 Key Features

- **Automated Detection**  
  Classifies chest X-ray images into two categories: `NORMAL` vs. `PNEUMONIA`.

- **Safety-First Approach**  
  Implements a configurable **confidence threshold** to prioritize high sensitivity (minimizing false negatives).

- **Real-time Analysis**  
  Instant model inference with probability visualization for uploaded X-ray images.

- **Interactive UI**  
  Built with **Streamlit**, enabling easy use by non-technical medical staff.

---

## 📂 Project Structure

AI_Assisted_Pneumonia_Screening/
│
├── train.py              # Model training script (ResNet18, data augmentation, evaluation)
├── app_xray_v2.py        # Streamlit deployment app (preprocessing, inference, visualization)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

---

## 🛠️ Installation & Usage

### 1. Prerequisites

- Python 3.8+
- Git

Clone the repository and install dependencies:

git clone https://github.com/baond12/AI_Assisted_Pneumonia_Screening.git  
cd AI_Assisted_Pneumonia_Screening  
pip install -r requirements.txt  

---

### 2. Model Setup

Note: A trained model file (.pth) is required to run the application.

Option A – Train the model from scratch:

python train.py  

(Requires the Chest X-Ray Images (Pneumonia) dataset.)

Option B – Use a pre-trained model:

- Place an existing .pth file in the project directory.
- Update the model_path variable in app_xray_v2.py to point to the model file.

---

### 3. Run the Application

streamlit run app_xray_v2.py

---

## 📊 Model Overview & Design

- Backbone: ResNet18 (pre-trained on ImageNet, frozen feature extractor)
- Classifier: Custom fully connected layer (2 output classes)
- Optimization: SGD with Momentum
- Design Objective: High recall (sensitivity) for clinical screening scenarios

The confidence threshold can be adjusted to control the trade-off between false positives and missed cases.

---

## ⚠️ Medical Disclaimer

This project is developed for educational and experimental purposes only.
It is NOT a certified medical device.

All predictions must be reviewed by qualified medical professionals.
The system is designed to assist screening workflows, not to provide definitive diagnoses.

---

## 👥 Authors

Group [Your Group Name]  
Data Mining Course – HK252
