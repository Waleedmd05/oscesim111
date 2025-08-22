import os
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
import skimage.io
import torchxrayvision as xrv

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import Resize

# === SETUP ===
st.set_page_config(page_title="Chest X-Ray AI Assistant", layout="wide")
st.title("🫁 Chest X-Ray AI Assistant")
st.markdown("Drop a chest X-ray image to get predictions, heatmaps, and a suggested diagnosis.")

# === SIDEBAR ===
st.sidebar.title("📤 Upload CXR Image")
uploaded_file = st.sidebar.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"])

# === THRESHOLDS ===
def map_findings_to_diagnosis(results, threshold=0.59):
    if results.get("Tuberculosis", 0) > 0.6:
        return "Tuberculosis"
    elif results.get("Mass", 0) > 0.62 or results.get("Nodule", 0) > 0.65:
        return "Lung tumor"
    elif results.get("Emphysema", 0) > 0.51 or results.get("Fibrosis", 0) > 0.6:
        return "COPD"
    elif results.get("Cardiomegaly", 0) > 0.62 and results.get("Edema", 0) > 0.6:
        return "Heart failure"
    elif results.get("Consolidation", 0) > 0.58 or results.get("Edema", 0) > 0.63:
        return "Pneumonia"
    elif results.get("Pleural Effusion", 0) > 0.61:
        return "Pleural effusion"
    elif results.get("Pneumothorax", 0) > 0.6:
        return "Pneumothorax"
    elif results.get("Fracture", 0) > 0.65 or results.get("Rib Fracture", 0) > 0.65:
        return "Fracture"
    elif results.get("Interstitial lung disease", 0) > 0.6 or results.get("Fibrosis", 0) > 0.6:
        return "Interstitial lung disease"
    elif results.get("Edema", 0) > 0.58 and results.get("Consolidation", 0) > 0.58:
        return "ARDS"
    elif all(score < threshold for score in results.values()):
        return "No finding"
    else:
        return "Other diseases"

# === PROCESS IMAGE ===
def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image).astype(np.float32)
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    img = np.clip(img, -3, 3) * 300
    img = img[None, ...]
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)
    img_tensor = torch.from_numpy(img)[None, ...]
    return img_tensor, img

# === DISPLAY REPORT ===
if uploaded_file:
    img_tensor, processed_img = process_image(uploaded_file)
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        probs = output[0].detach().numpy()
        results = dict(zip(model.pathologies, probs))
        filtered = {k: v for k, v in results.items() if v > 0.05}
        top_3 = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:3]
        pred_idx = output[0].argmax().item()
        diagnosis = map_findings_to_diagnosis(results)

    # Grad-CAM
    target_layers = [model.features.denseblock4]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
    grayscale_cam = Resize([224, 224])(torch.from_numpy(grayscale_cam)).numpy()

    # Heatmap overlay
    rgb_img = np.repeat(processed_img, 3, axis=0).transpose(1, 2, 0)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    heatmap_img = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Chest X-Ray")
        st.image(processed_img[0], clamp=True, caption="Original X-Ray", use_column_width=True)

    with col2:
        st.subheader("🧠 What did the AI focus on?")
        st.image(heatmap_img, caption="AI Attention Map", use_column_width=True)

    # Summary
    st.markdown("---")
    st.subheader("📋 Prediction Summary")
    for cond, score in top_3:
        st.write(f"- **{cond}**: {score:.2f}")
    st.success(f"📌 Most Probable Diagnosis: **{diagnosis}**")

else:
    st.info("Upload a chest X-ray on the left to begin.")
