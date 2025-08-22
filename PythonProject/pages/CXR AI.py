# Wmed | CXR AI — LLM Teaching Explanation (No external Grad-CAM)
# ------------------------------------------------------------------
# - Loads .env from this file's dir if the hardcoded path doesn't exist
# - Falls back to Streamlit secrets or inline key
# - Uses an inlined Grad-CAM (no pytorch-grad-cam dependency)
# ------------------------------------------------------------------

import os
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt  # (ok to keep even if unused)
import skimage.io  # (ok to keep even if unused)
import torchxrayvision as xrv
from PIL import Image
from torchvision.transforms import Resize  # (ok to keep even if unused)
import torch.nn.functional as F
import requests
from pathlib import Path
from dotenv import load_dotenv
from matplotlib import cm  # for heatmap coloring

# ================== Secrets / ENV Loading ==================
# Try your hardcoded local path first (if it exists), otherwise use ./.env near this file.
HARD_ENV_PATH = Path("/Users/waleedal-tahafi/PycharmProjects/PythonProject/pages/.env")
FALLBACK_ENV_PATH = Path(__file__).resolve().parent / ".env"
ENV_PATH = HARD_ENV_PATH if HARD_ENV_PATH.exists() else FALLBACK_ENV_PATH
load_dotenv(dotenv_path=str(ENV_PATH), override=True)

OPENAI_API_KEY_INLINE = ""  # leave empty in prod
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY_INLINE

# If Streamlit secrets exist, they can override missing env
try:
    if not OPENAI_API_KEY and "OPENAI_API_KEY" in st.secrets:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"

# Early check
st.set_page_config(page_title="Wmed | CXR AI", layout="wide")
if not OPENAI_API_KEY:
    st.error("No API key found. Add OPENAI_API_KEY to .env (same folder), or to Streamlit Secrets, or paste inline.")
    st.stop()

# ================== UI ==================
st.title("🧠 Wmed AI - Chest X-Ray Analysis")
st.markdown("Level up your diagnostics with **AI-powered precision**. Upload a chest X-ray to begin.")

st.sidebar.title("📤 Upload Chest X-Ray")
uploaded_file = st.sidebar.file_uploader("Accepted formats: PNG, JPG, JPEG", type=["png", "jpg", "jpeg"])

# ================== Diagnosis Heuristics ==================
def map_findings_to_diagnosis(results, threshold=0.59):
    if results.get("Tuberculosis", 0) > 0.6:
        return "Tuberculosis"
    elif results.get("Mass", 0) > 0.51 or results.get("Nodule", 0) > 0.65:
        return "Lung tumor"
    elif results.get("Emphysema", 0) > 0.51 or results.get("Fibrosis", 0) > 0.6:
        return "COPD"
    elif results.get("Cardiomegaly", 0) > 0.52 and results.get("Edema", 0) > 0.55:
        return "Heart failure"
    elif results.get("Consolidation", 0) > 0.58 or results.get("Edema", 0) > 0.63:
        return "Pneumonia"
    elif results.get("Pleural Effusion", 0) > 0.61 or results.get("Effusion", 0) > 0.55:
        return "Pleural effusion"
    elif results.get("Pneumothorax", 0) > 0.54:
        return "Pneumothorax"
    elif results.get("Fracture", 0) > 0.65 or results.get("Rib Fracture", 0) > 0.65:
        return "Fracture"
    elif results.get("Interstitial lung disease", 0) > 0.6 or results.get("Fibrosis", 0) > 0.6:
        return "Interstitial lung disease"
    elif results.get("Edema", 0) > 0.58 and results.get("Consolidation", 0) > 0.58:
        return "ARDS"
    elif all(score < 0.51 for score in results.values()):
        return "No finding"
    else:
        return "Other diseases"

explanations = {
    "Effusion": "Fluid accumulation in the pleural space, often linked to infection or heart failure.",
    "Lung Opacity": "An area of increased density, may indicate pneumonia, tumour, or fibrosis.",
    "Enlarged Cardiomediastinum": "Could suggest cardiomegaly or mediastinal masses.",
    "Mass": "Localized lesion that may suggest malignancy.",
    "Nodule": "Smaller lesion that may be benign or malignant depending on characteristics.",
    "Consolidation": "Lung filled with fluid or pus, typically seen in pneumonia.",
    "Edema": "Fluid accumulation usually due to cardiac causes like heart failure.",
    "Cardiomegaly": "Enlargement of the heart, commonly associated with chronic hypertension or heart failure.",
    "Pneumothorax": "Air trapped in pleural space causing lung collapse.",
    "Fibrosis": "Scarring of lung tissue, often from chronic inflammation or exposure.",
}

# ================== Image Processing ==================
def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    orig_size = image.size
    img = np.array(image).astype(np.float32)
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    img = np.clip(img, -3, 3) * 300
    img = img[None, ...]
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)
    img_tensor = torch.from_numpy(img)[None, ...]
    return img_tensor, img, orig_size, np.array(image)

# ================== Inlined Grad-CAM (no external lib) ==================
@torch.no_grad()
def _safe_forward(model, x):
    return model(x)

def generate_gradcam(model, target_layer, input_tensor, target_index=None):
    """
    Minimal Grad-CAM:
    - forward hook grabs activations
    - register_hook on the activation grabs gradients
    - weights = GAP over gradients
    - cam = ReLU(sum_k(weights_k * activations_k))
    Returns a HxW numpy array normalized to [0,1] at the model's input resolution.
    """
    # We need gradients; temporarily enable grad context for this pass
    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    activations = []
    gradients = []

    def f_hook(module, inp, out):
        activations.append(out)
        # capture gradients w.r.t. the activation tensor
        out.register_hook(lambda g: gradients.append(g))

    handle = target_layer.register_forward_hook(f_hook)

    # forward pass *with* grad
    output = model(input_tensor)
    if target_index is None:
        target_index = output[0].argmax().item()

    score = output[:, target_index]
    # clear old grads, then backward
    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    # pop what we captured
    act = activations[-1]     # [1, C, h, w]
    grad = gradients[-1]      # [1, C, h, w]

    # weights: global-average over spatial dims
    weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, h, w]
    cam = F.relu(cam)

    # upsample to model input spatial size
    cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                        mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    # normalize to [0,1]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # cleanup
    handle.remove()
    if was_training:
        model.train()
    return cam

def overlay_heatmap_on_image(raw_gray_img, cam_resized, alpha=0.35):
    """
    raw_gray_img: HxW (0..255 or arbitrary). cam_resized: HxW in [0,1].
    Returns RGB uint8 image with heatmap overlay.
    """
    rgb_img = np.repeat(raw_gray_img[..., None], 3, axis=2).astype(np.float32)
    # normalize image to [0,1]
    rgb_norm = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    heatmap = cm.get_cmap("jet")(cam_resized)[..., :3]  # RGB in [0,1]
    overlay = heatmap * alpha + rgb_norm * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)

# ================== LLM Teaching Explanation ==================
def llm_teaching_explanation(diagnosis: str, top_findings: dict, notes: str = "") -> str:
    url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    sys = (
        "You are a radiology tutor for medical students. Educational use only. "
        "Do not claim diagnostic certainty. Use plain language. Keep it under 120 words. "
        "Structure your answer with: Findings → Why they suggest the possible diagnosis → What else to consider / next steps."
    )
    user = f"Possible diagnosis: {diagnosis}. Top findings: {top_findings}. Notes: {notes or 'none'}. Write a concise explanation."

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ],
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(LLM explanation unavailable: {e})"

# ================== Main Inference ==================
if uploaded_file:
    img_tensor, processed_img, orig_size, raw_img = process_image(uploaded_file)

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

    # ---- Grad-CAM without external dependency ----
    target_layer = model.features.denseblock4
    grayscale_cam = generate_gradcam(model, target_layer, img_tensor, pred_idx)

    # resize Grad-CAM to the raw uploaded image size
    grayscale_cam_tensor = torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0)
    grayscale_cam_resized = F.interpolate(
        grayscale_cam_tensor,
        size=(raw_img.shape[0], raw_img.shape[1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()

    # overlay heatmap
    heatmap_img = overlay_heatmap_on_image(raw_img, grayscale_cam_resized, alpha=0.35)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original X-Ray")
        st.image(raw_img, clamp=True, caption="Uploaded Chest X-Ray", use_container_width=True)
    with col2:
        st.subheader("🔥 AI Focus Heatmap")
        st.image(heatmap_img, caption="Where the AI looked", use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Prediction Results")
    for cond, score in top_3:
        st.write(f"- **{cond}**: {score:.2f}")
        st.caption(explanations.get(cond, "No explanation available."))

    st.success(f"🏁 **AI Diagnosis**: {diagnosis}")

    st.markdown("### 🎓 Teaching Explanation (LLM)")
    tf_dict = {k: float(v) for k, v in top_3}
    notes = "Heatmap highlights regions driving the model's top class; treat as pedagogical only."
    expl = llm_teaching_explanation(diagnosis, tf_dict, notes)
    st.write(expl)
    st.caption("Education only · Not for diagnosis or patient care")

else:
    st.info("📂 Upload a chest X-ray to begin diagnosis.")
