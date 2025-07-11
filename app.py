import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import cv2

# Streamlit page config
st.set_page_config(
    page_title="DR Detector - Group 8",
    page_icon="🧿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load the model
model = torch.load('full_model.pth', map_location=torch.device('cpu'), weights_only=False)
names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Modern UI Styling
st.markdown("""
    <style>
        h1 {color: #2c3e50; text-align: center; font-size: 40px; margin-bottom: 10px;}
        h4 {text-align: center; color: #7b1fa2;}
        .stButton>button {
            background: linear-gradient(to right, #43cea2, #185a9d);
            color: white;
            padding: 10px 25px;
            font-size: 16px;
            border-radius: 10px;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #185a9d, #43cea2);
        }
        .prediction {
            font-size: 26px;
            font-weight: bold;
            color: #d32f2f;
            text-align: center;
            margin-top: 20px;
        }
        .team-card {
            background: #ffffff;
            padding: 15px;
            margin: 10px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
        }
        .team-name {
            font-weight: bold;
            color: #2e7d32;
        }
        .role {
            font-style: italic;
            color: #5e35b1;
        }
        footer {
            font-size: 15px;
            color: #6c757d;
            text-align: center;
            margin-top: 50px;
            border-top: 1px solid #ccc;
            padding-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🧿 Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)
st.markdown("<h4>MINI Project - Group 8</h4>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("##### Upload an eye image to analyze for diabetic retinopathy severity.")

# Image preprocessing
def crop_image_from_gray(img, tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray_img > tol
    if mask.any():
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
    return img

# Transformations
val_transforms = A.Compose([
    A.Resize(height=728, width=728),
    A.Normalize(mean=[0.3199, 0.2240, 0.1609],
                std=[0.3020, 0.2183, 0.1741],
                max_pixel_value=255.0),
    ToTensorV2(),
])

# Prediction function
def make_prediction(model, img):
    x = val_transforms(image=img)["image"].unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        pred[pred < 0.75] = 0
        pred[(pred >= 0.75) & (pred < 1.5)] = 1
        pred[(pred >= 1.5) & (pred < 2.4)] = 2
        pred[(pred >= 2.4) & (pred < 3.4)] = 3
        pred[pred >= 3.4] = 4
        pred = pred.long().squeeze(1)
    return pred.cpu().numpy()[0]

# File uploader
uploaded_image = st.file_uploader("📤 Upload Eye Image (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    img = Image.open(uploaded_image).convert('RGB')
    st.image(img.resize((512, 512)), caption="Uploaded Eye Image")

    processed_img = crop_image_from_gray(np.array(img))

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing..."):
            result = make_prediction(model, processed_img)
            st.markdown("---")
            st.markdown(f'<div class="prediction">🩺 Diagnosis: <span style="color:#1976d2">{names[result]}</span></div>', unsafe_allow_html=True)

# Styled Footer with team members in two columns
st.markdown("---")
st.markdown("""
    <footer>
        <h4>🔹 Project Team Members 🔹</h4>
    </footer>
""", unsafe_allow_html=True)

# Two columns layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="team-card">
            <p class="team-name">Adepu Vaishnavi</p>
            <p class="role">22071A1266</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="team-card">
            <p class="team-name">Harsha vardhan Botlagunta</p>
            <p class="role">22071A1285</p>
            
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="team-card">
            <p class="team-name">Maganti Pranathi</p>
            <p class="role">22071A1296</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="team-card">
            <p class="team-name">Yedla Pranav</p>
            <p class="role">22071A12C9</p>
        </div>
    """, unsafe_allow_html=True)

# Guide info centered
st.markdown("""
    <div style="text-align: center;">
        <div class="team-card" style="max-width: 300px; margin: 20px auto;">
            <p class="team-name">Dr. G.Naga Chandrika</p>
            <p class="role">Mentor</p>
        </div>
    </div>
    <footer>
        <p>© 2025 MINI Project Group 8</p>
    </footer>
""", unsafe_allow_html=True)
