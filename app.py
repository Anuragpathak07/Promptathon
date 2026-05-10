"""
===============================================================
Step 4 — Gradio Demo: Anomaly Heatmap Visualiser
---------------------------------------------------------------
Interactive web UI that:
  1. Accepts a component image upload
  2. Lets the user select the MVTec category (model to use)
  3. Runs PatchCore inference
  4. Displays the original image, anomaly heatmap overlay,
     and a textual verdict (NORMAL / DEFECTIVE)
===============================================================
Run:
    python app.py
Then open http://localhost:7860
===============================================================
"""

import io
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torchvision.transforms as T
import cv2
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
from PIL import Image

from model import build_patchcore, PatchCore

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

# ── Pre-load all trained models ──────────────────────────────────────
CATEGORIES    = CFG["dataset"]["categories"]
IMAGE_SIZE    = CFG["dataset"]["image_size"]
HEATMAP_ALPHA = CFG["demo"]["heatmap_alpha"]
COLORMAP      = CFG["demo"]["colormap"]

MODELS: dict[str, PatchCore] = {}


def load_all_models() -> None:
    for cat in CATEGORIES:
        try:
            pc = build_patchcore(cat)
            pc.load()
            MODELS[cat] = pc
            log.info(f"  ✓ Loaded model: {cat}")
        except FileNotFoundError:
            log.warning(f"  ✗ No trained model for {cat} — skipping.")


load_all_models()

# Image normalisation (ImageNet stats)
TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Result images dir ────────────────────────────────────────────────
RESULTS_DIR = Path(CFG["evaluation"]["output_dir"])
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================== #
#  Core inference + visualisation                                      #
# ================================================================== #
def make_heatmap_overlay(
    original_pil: Image.Image,
    score_map: np.ndarray,
    threshold: Optional[float] = None,
    alpha: float = HEATMAP_ALPHA,
    cmap_name: str = COLORMAP,
) -> Image.Image:
    """
    Blend a coloured anomaly heatmap over the original image.
    """
    orig_w, orig_h = original_pil.size
    orig_np = np.array(original_pil.convert("RGB"))

    # Normalise score map relative to threshold if available, otherwise min-max
    if threshold is not None:
        s_min = 0.0
        # Scale max to at least the threshold so normal regions stay cold (blue),
        # and anomalies exceeding the threshold display high-contrast hot colors (red).
        s_max = max(threshold, score_map.max())
    else:
        s_min, s_max = score_map.min(), score_map.max()

    if s_max > s_min:
        norm_map = (score_map - s_min) / (s_max - s_min)
    else:
        norm_map = np.zeros_like(score_map)

    # Resize to original image dimensions
    norm_resized = cv2.resize(norm_map, (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    import matplotlib
    cmap    = matplotlib.colormaps.get_cmap(cmap_name)
    colored = (cmap(norm_resized)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    blended = (
        (1 - alpha) * orig_np + alpha * colored
    ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def make_score_bar(
    anomaly_score: float,
    threshold: float,
    width: int = 400,
    height: int = 60,
) -> Image.Image:
    """
    Render a horizontal score bar indicating normal vs anomalous.
    """
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    norm_score = min(anomaly_score / (threshold * 2 + 1e-8), 1.0)
    colour     = "#ff4d6d" if anomaly_score >= threshold else "#06d6a0"

    ax.barh([0], [norm_score], color=colour, height=0.6, alpha=0.85)
    ax.barh([0], [1],          color="#2a2b4c", height=0.6, alpha=0.3)
    ax.axvline(0.5, color="#ffd93d", lw=1.5, ls="--")
    ax.set_xlim(0, 1)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100,
                facecolor="#0f0f1a")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def run_inference(
    image: Optional[Image.Image],
    category: str,
) -> Tuple[Image.Image, Image.Image, str, str]:
    """
    Gradio inference function.

    Returns
    -------
    heatmap_overlay : PIL image
    score_bar_img   : PIL image
    verdict_text    : HTML string
    metrics_text    : plain text
    """
    if image is None:
        dummy = Image.new("RGB", (224, 224), "#1a1a2e")
        return dummy, dummy, "<p>Please upload an image.</p>", ""

    if category not in MODELS:
        dummy = Image.new("RGB", (224, 224), "#1a1a2e")
        return (
            dummy, dummy,
            f"<p style='color:#ff4d6d'>No trained model for '{category}'."
            f" Run train.py first.</p>", ""
        )

    pc = MODELS[category]

    # Preprocess
    img_pil   = image.convert("RGB")
    img_tensor = TRANSFORM(img_pil).unsqueeze(0)  # (1, 3, H, W)

    # Inference
    anomaly_score, score_map = pc.predict_image(img_tensor)
    threshold                = pc.threshold
    is_anomaly               = anomaly_score >= threshold

    # Visualisations
    overlay   = make_heatmap_overlay(img_pil, score_map, threshold=threshold)
    score_bar = make_score_bar(anomaly_score, threshold)

    # Verdict HTML
    if is_anomaly:
        verdict_html = f"""
        <div style="background:#1c0a0a;border-left:4px solid #ff4d6d;
                    padding:16px;border-radius:8px;font-family:Inter,sans-serif;">
          <div style="font-size:1.5rem;font-weight:700;color:#ff4d6d">
            ⚠️ DEFECT DETECTED
          </div>
          <div style="color:#ffb3b3;margin-top:6px">
            Anomaly score <b>{anomaly_score:.4f}</b> exceeds threshold
            <b>{threshold:.4f}</b>
          </div>
          <div style="color:#888;margin-top:4px;font-size:0.85rem">
            Category: <b>{category}</b>
          </div>
        </div>
        """
    else:
        verdict_html = f"""
        <div style="background:#0a1c10;border-left:4px solid #06d6a0;
                    padding:16px;border-radius:8px;font-family:Inter,sans-serif;">
          <div style="font-size:1.5rem;font-weight:700;color:#06d6a0">
            ✅ NORMAL — No Defect
          </div>
          <div style="color:#a3ffd4;margin-top:6px">
            Anomaly score <b>{anomaly_score:.4f}</b> is below threshold
            <b>{threshold:.4f}</b>
          </div>
          <div style="color:#888;margin-top:4px;font-size:0.85rem">
            Category: <b>{category}</b>
          </div>
        </div>
        """

    metrics_text = (
        f"Anomaly Score : {anomaly_score:.6f}\n"
        f"Threshold     : {threshold:.6f}\n"
        f"Verdict       : {'ANOMALY' if is_anomaly else 'NORMAL'}\n"
        f"Category      : {category}\n"
        f"Model trained : ✓"
    )

    return overlay, score_bar, verdict_html, metrics_text


# ================================================================== #
#  Gradio UI                                                           #
# ================================================================== #
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

body, .gradio-container {
    background: linear-gradient(135deg, #0d0d1a 0%, #13132a 100%) !important;
    color: #e0e0f0 !important;
}

.gr-panel, .gr-box, .gr-padded {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(124,106,247,0.2) !important;
    border-radius: 12px !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #7c6af7, #a855f7) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,106,247,0.4) !important;
}

h1, h2, h3 { color: #e0e0f0 !important; }

.label-wrap { color: #a0a0c0 !important; }
"""

HEADER_MD = """
<div style="text-align:center;padding:24px 0 16px;font-family:Inter,sans-serif;">
  <h1 style="font-size:2.2rem;font-weight:700;
             background:linear-gradient(90deg,#7c6af7,#f7a96a);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             margin:0;">
    🔬 Industrial Anomaly Detector
  </h1>
  <p style="color:#8888aa;margin-top:8px;font-size:1rem;">
    PatchCore · ResNet-50 · MVTec-AD &nbsp;|&nbsp;
    Upload a component image to detect manufacturing defects
  </p>
</div>
"""

INSTRUCTIONS_MD = """
### How to Use
1. **Select category** — choose the component type that matches your image.
2. **Upload image** — drag-and-drop or click to browse.
3. **Detect** — click the button to run anomaly detection.
4. **Inspect heatmap** — red/warm regions indicate potential defect locations.

> **Note:** A trained PatchCore model must exist for the selected category.
> Run `python train.py` first.
"""


def build_demo() -> gr.Blocks:
    available = list(MODELS.keys()) or CATEGORIES
    default_cat = available[0] if available else CATEGORIES[0]

    with gr.Blocks(
        title="Industrial Anomaly Detector",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="violet",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
    ) as demo:

        gr.HTML(HEADER_MD)

        with gr.Row():
            # ── Left panel ───────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown(INSTRUCTIONS_MD)

                category_dd = gr.Dropdown(
                    choices=CATEGORIES,
                    value=default_cat,
                    label="Component Category",
                    interactive=True,
                )

                input_img = gr.Image(
                    type="pil",
                    label="Upload Component Image",
                    height=280,
                )

                detect_btn = gr.Button(
                    "🔍 Detect Anomaly",
                    variant="primary",
                    size="lg",
                )

                metrics_box = gr.Textbox(
                    label="Raw Metrics",
                    lines=6,
                    interactive=False,
                    placeholder="Inference results will appear here …",
                )

            # ── Right panel ──────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Detection Results")

                verdict_html = gr.HTML("<p style='color:#666'>Run detection to see verdict.</p>")

                heatmap_out = gr.Image(
                    label="Anomaly Heatmap Overlay",
                    height=280,
                    interactive=False,
                )

                score_bar_out = gr.Image(
                    label="Anomaly Score Bar  (yellow = threshold)",
                    height=70,
                    interactive=False,
                )

        # ── Examples ─────────────────────────────────────────────────
        example_dir = Path("./sample_images")
        if example_dir.exists():
            sample_imgs = list(example_dir.glob("*.png")) + \
                          list(example_dir.glob("*.jpg"))
            if sample_imgs:
                gr.Examples(
                    examples=[[str(p), default_cat] for p in sample_imgs[:6]],
                    inputs=[input_img, category_dd],
                    label="Sample Images",
                )

        # ── Event binding ─────────────────────────────────────────────
        detect_btn.click(
            fn=run_inference,
            inputs=[input_img, category_dd],
            outputs=[heatmap_out, score_bar_out, verdict_html, metrics_box],
        )

        gr.Markdown(
            "<div style='text-align:center;color:#555577;padding-top:16px;"
            "font-size:0.8rem;'>PatchCore | ResNet-50 | MVTec-AD | "
            "Built with Gradio</div>"
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_port=CFG["demo"]["port"],
        share=CFG["demo"]["share"],
        inbrowser=True,
    )
