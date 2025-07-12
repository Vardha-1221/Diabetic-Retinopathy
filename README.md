# 🧿 Diabetic Retinopathy Detection

Diabetic retinopathy is a complication of diabetes that can lead to vision loss if not detected early. 
Our project uses a deep learning approach—specifically a **ResNet-18 convolutional neural network**—to make early detection easier and more accessible.
With this tool, anyone can upload a retinal image and get an instant prediction of disease severity, powered by state-of-the-art computer vision.

---

## Why Dual Architecture?

By separating the training and deployment parts of the project, we get the best of both worlds:
- **Instant results for users:** The web app is always ready to give quick predictions without any heavy computation or training delays.
- **Flexibility for developers:** You can retrain or improve the model as new data becomes available, without touching the user interface or interrupting the service.
- **Easy updates:** The model can be updated in the background and swapped in for better accuracy, all while users keep using the app as usual.
- **Resource efficiency:** Training (which is resource-intensive) can be done on powerful machines, while the web app runs smoothly even on modest hardware.

This approach makes the system robust, maintainable, and user-friendly for both end-users and developers.

---

## Dual Architecture: How the Project Works

This project is split into two main parts:

### 1. Deployment (Web App)
- **File:** `deploy/app.py`
- **What it does:** Lets you upload a retinal image and instantly get a prediction in your browser.
- **Main libraries and tech used:**
  - `streamlit` (for the web interface)
  - `torch` (PyTorch, for loading and running the trained ResNet-18 model)
  - `albumentations` and `opencv-python (cv2)` (for image preprocessing)
  - `PIL` (Python Imaging Library) and `numpy` (for image handling and array operations)
- **How it works:** Loads a trained ResNet-18 model, preprocesses your image, and shows the result in a clean UI.

### 2. Training (Model Development)
- **File:** `training/train_test.py`
- **What it does:** Lets you train or retrain the deep learning model (ResNet-18) on your own dataset.
- **Main libraries and tech used:**
  - `torch` and `torchvision` (for model, training, and data loading)
  - `numpy` (for data handling)
  - `os`, `random` (for reproducibility and file management)
- **How it works:** Trains a ResNet-18 model (with transfer learning) on your images, saves the model, and prints test accuracy.

---

## What does it do?
- Upload a retinal image (JPG, JPEG, or PNG).
- The app preprocesses the image and runs it through a trained **ResNet-18** model.
- You’ll see a diagnosis: one of five levels of diabetic retinopathy (from No DR to Proliferative DR).

---

## Project Structure
```
Diabetic-Retinopathy/
├── deploy/
│   ├── app.py           # Streamlit web app (uses ResNet-18 for inference)
│   ├── full_model.pth   # Trained ResNet-18 model for inference
│   └── requirements.txt # Dependencies for deployment
├── training/
│   ├── train_test.py    # Model training/testing (ResNet-18)
│   ├── new_model.pth    # Output from training
│   └── data/            # Training/test images
├── images/              # Sample images for testing
└── README.md
```

---

## Disease Classes
| Class | Name              |
|-------|-------------------|
| 0     | No DR             |
| 1     | Mild              |
| 2     | Moderate          |
| 3     | Severe            |
| 4     | Proliferative DR  |

---

## How to Use
- Open the web app (Streamlit).
- Upload your eye image.
- Click the **Analyze Image** button.
- See your result instantly, powered by ResNet-18.

## For Developers
- Retrain the model using the scripts in `training/` if you want to improve or adapt it. The training script uses PyTorch and torchvision to train a ResNet-18 model from scratch or with transfer learning.
- The app will show an error if the model file is missing.

---

## Technologies & Libraries Used
- **Deep Learning Model:** ResNet-18 (PyTorch, torchvision)
- **Web App:** Streamlit
- **Image Preprocessing:** albumentations, OpenCV (cv2), PIL, numpy
- **Training Utilities:** torch, torchvision, numpy, os, random

---

## Notes
- This tool is for educational and research purposes. It’s not a substitute for a doctor’s diagnosis.
- If you want to contribute or have ideas, feel free to open an issue or pull request.

---





