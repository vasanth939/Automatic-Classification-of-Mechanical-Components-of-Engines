import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib
import torchvision.models as models
import asyncio
import os

# Ensure correct Scikit-learn version (1.2.2)
try:
    import sklearn
    assert sklearn.__version__ == "1.2.2", "Scikit-learn version mismatch! Install `scikit-learn==1.2.2`"
except AssertionError as e:
    st.error(str(e))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the SVM model
@st.cache_resource
def load_model():
    try:
        return joblib.load("svm_resnet50_model.pkl")
    except EOFError:
        st.error("Error loading the model! The file might be corrupted. Try re-saving the model.")
        return None

model = load_model()

# Load pre-trained ResNet50 feature extractor
@st.cache_resource
def load_resnet():
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
    resnet.to(device)
    resnet.eval()
    return resnet

resnet = load_resnet()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class labels (Modify based on your dataset)
CLASS_NAMES = {
    0: "Bolt",
    1: "LocatingPin",
    2: "Nut",
    3: "Washer",
}

st.title("🔧 Mechanical Component Classifier")

# Example images
st.subheader("Example Images")
example_folder = "sample_dir"  # Make sure this folder exists with example images
if os.path.exists(example_folder):
    example_files = [f for f in os.listdir(example_folder) if f.endswith(("jpg", "png", "jpeg"))]
    cols = st.columns(len(example_files))
    for i, file in enumerate(example_files):
        img_path = os.path.join(example_folder, file)
        cols[i].image(img_path, caption=file.split(".")[0], use_container_width=True)
else:
    st.warning("No example images found. Make sure `example_images/` folder exists.")

# Upload Image
st.subheader("Upload Your Own Image")
uploaded_file = st.file_uploader("Upload an image of a mechanical component", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        features = resnet(img_tensor)
    features = features.view(features.size(0), -1).cpu().numpy()  # Flatten

    # Predict using the SVM classifier
    if model:
        prediction = model.predict(features)[0]
        class_name = CLASS_NAMES.get(prediction, "Unknown Class")  # Convert to class name
        st.success(f"🔍 **Predicted Class:** {class_name}")
    else:
        st.error("Model failed to load. Please check logs.")
