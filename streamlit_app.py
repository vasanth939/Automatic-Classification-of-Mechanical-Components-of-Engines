import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the SVM model correctly
@st.cache_resource
def load_model():
    return joblib.load("svm_resnet50_model.pkl")  # ✅ Corrected model loading

model = load_model()

# Load pre-trained ResNet50 feature extractor
@st.cache_resource
def load_resnet():
    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
    resnet.to(device)
    resnet.eval()
    return resnet

resnet = load_resnet()

# Define image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("Mechanical Component Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        features = resnet(img_tensor)
    features = features.view(features.size(0), -1).cpu().numpy()  # Flatten

    # Predict using the SVM classifier
    prediction = model.predict(features)[0]
    st.success(f"Predicted Class: **{prediction}**")
