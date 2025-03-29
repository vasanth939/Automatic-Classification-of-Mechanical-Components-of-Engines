import streamlit as st
import torch
import torchvision.transforms as transforms
import joblib  # 🔥 Load .pkl model
from torchvision import models
from PIL import Image
import os

# ----------------------------
# Streamlit page config
st.set_page_config(
    page_title="Mechanical Components Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load the model (.pkl file)
@st.cache_resource
def load_model():
    with open("svm_resnet50_model.pkl", "rb") as f:
        model = joblib.load(f)  # 🔥 Load model
    model.to(device)
    model.eval()
    return model

model = load_model()

# ----------------------------
# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Class names
class_names = ['bolt', 'locatingpin', 'nut', 'washer']

# ----------------------------
# Streamlit UI
st.title("🔧 Mechanical Components Classification")
st.markdown("""
Upload an image or choose a sample to classify mechanical components.

**Supported Classes:**
- 🔩 Bolt
- 📌 Locating Pin
- 🔩 Nut
- 🏵️ Washer
---
""")

# ----------------------------
# Image Selection (Upload or Sample)
option = st.radio("Select image source:", ("Upload", "Sample"), index=1)

if option == "Upload":
    uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)
elif option == "Sample":
    sample_dir = "sample_dir"  # 📁 Adjust folder name as needed
    try:
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    except FileNotFoundError:
        st.error(f"Folder '{sample_dir}' not found.")
        sample_files = []

    if sample_files:
        st.markdown("### Select a Sample Image")
        cols = st.columns(3)
        selected_sample = None
        for idx, file in enumerate(sample_files):
            img_path = os.path.join(sample_dir, file)
            thumb = Image.open(img_path).convert("RGB")
            with cols[idx % 3]:
                if st.button(file, key=file):
                    selected_sample = file
                st.image(thumb, caption=file, width=150)
        if selected_sample is None:
            selected_sample = sample_files[0]
        image = Image.open(os.path.join(sample_dir, selected_sample)).convert("RGB")
        st.image(image, caption=f"Selected Image: {selected_sample}", use_container_width=False)
    else:
        st.write("No sample images found in the sample folder.")

# ----------------------------
# Perform Classification
if 'image' in locals():
    st.markdown("---")
    st.success("### Model Prediction")

    # Preprocess
    input_img = transform(image)
    input_tensor = input_img.unsqueeze(0).to(device)

    # Model inference
    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    pred_class = class_names[pred_idx]

    # Display prediction
    st.success(f"### **Predicted Class: {pred_class}**")
