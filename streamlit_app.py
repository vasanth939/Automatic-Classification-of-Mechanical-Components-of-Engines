import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ----------------------------
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load the model (cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    # Recreate the ResNet50 architecture
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    num_classes = 4  # Change this if you have a different number of classes
    model.fc = nn.Linear(num_features, num_classes)
    # Load the saved weights (ensure this file is in your working directory)
    model.load_state_dict(torch.load("resnet50_gradcam_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ----------------------------
# Define image transformation (must match what was used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Update with your actual class names from training
class_names = ['bolt', 'locatingpin', 'nut', 'washer']

# ----------------------------
# Define Grad-CAM (using register_backward_hook to avoid conflicts)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        handle_backward = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.append(handle_forward)
        self.hook_handles.append(handle_backward)
    
    def __call__(self, input_tensor, class_idx=None):
        # Ensure the input requires gradients
        input_tensor.requires_grad = True
        
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        target = output[0, class_idx]
        target.backward()
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        # Weighted combination of forward activations
        grad_cam_map = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        # Upsample to the input image size
        grad_cam_map = torch.nn.functional.interpolate(grad_cam_map, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        grad_cam_map = grad_cam_map.squeeze().cpu().numpy()
        # Normalize between 0 and 1
        grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
        return grad_cam_map

# Initialize Grad-CAM with the last convolutional layer (ResNet50's layer4 block)
target_layer = model.layer4[-1]
grad_cam = GradCAM(model, target_layer)

import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ----------------------------
# Page configuration for a polished look
st.set_page_config(
    page_title="Mechanical Components Classification Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Title and introduction
st.title("🔧 Mechanical Components Classification Demo")
st.markdown("""
Welcome to the **Mechanical Components Classification Demo**. This interactive app uses a deep learning model to automatically classify mechanical engine components and visualize the model's decision process with Grad‑CAM.

**How to Use the App:**
- **Upload:** Choose your own image file.
- **Sample:** Select from our collection of sample images.
- The model will predict the component type and display a Grad‑CAM heatmap overlay highlighting the regions influencing the prediction.
---
""")

# ----------------------------
# Option for image source: Upload or Sample
option = st.radio("Select image source:", ("Upload", "Sample"))

if option == "Upload":
    uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)
elif option == "Sample":
    sample_dir = "sample_dir"  # Update with your folder name on GitHub
    try:
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    except FileNotFoundError:
        st.error(f"Folder '{sample_dir}' not found. Please ensure the folder exists in your repository.")
        sample_files = []
    
    if sample_files:
        st.markdown("### Select a Sample Image")
        # Create a grid of thumbnails using 3 columns per row
        cols = st.columns(3)
        selected_sample = None
        for idx, file in enumerate(sample_files):
            img_path = os.path.join(sample_dir, file)
            thumb = Image.open(img_path).convert("RGB")
            with cols[idx % 3]:
                if st.button(file, key=file):
                    selected_sample = file
                st.image(thumb, caption=file, width=150)
        # Default to the first image if none is selected
        if selected_sample is None:
            selected_sample = sample_files[0]
        image = Image.open(os.path.join(sample_dir, selected_sample)).convert("RGB")
        st.image(image, caption=f"Selected Image: {selected_sample}", use_container_width=False)
    else:
        st.write("No sample images found in the sample folder.")

# ----------------------------
# If an image is available, run model prediction and Grad‑CAM visualization
if 'image' in locals():
    st.markdown("---")
    st.markdown("### Model Prediction & Grad‑CAM Visualization")
    
    # Preprocess the image and create a batch
    input_img = transform(image)
    input_tensor = input_img.unsqueeze(0).to(device)
    
    # Run the model to get prediction
    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    pred_class = class_names[pred_idx]
    st.markdown(f"**Predicted Class:** {pred_class}")
    
    # Generate Grad‑CAM heatmap for the predicted class
    heatmap = grad_cam(input_tensor, class_idx=pred_idx)
    
    # Plot the original image with Grad‑CAM overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    img_np = np.array(image.resize((224, 224)))
    ax.imshow(img_np)
    ax.imshow(heatmap, cmap='jet', alpha=0.5, extent=(0, 224, 224, 0))
    ax.axis('off')
    st.pyplot(fig)
