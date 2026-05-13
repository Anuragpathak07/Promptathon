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
  5. LEFT PANEL: Rich visualisation — score ring, radar chart,
     spatial zone breakdown, detection history sparkline
===============================================================
Run:
    python app.py
Then open http://localhost:7860
===============================================================
"""

import io
import json
import logging
import re
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
import spec_analyzer
import tempfile, datetime, textwrap
from pathlib import Path as _Path

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

# ── In-memory detection history (last 20 runs) ───────────────────────
_detection_history: list[dict] = []


def _push_history(score: float, threshold: float, is_anomaly: bool, category: str) -> None:
    _detection_history.append({
        "score": round(score, 5),
        "threshold": round(threshold, 5),
        "anomaly": is_anomaly,
        "category": category,
        "ts": datetime.datetime.now().strftime("%H:%M:%S"),
    })
    if len(_detection_history) > 20:
        _detection_history.pop(0)


# ================================================================== #
#  Markdown → clean HTML converter                                     #
# ================================================================== #
def _md_inline(text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", text.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong style='color:#FFFFFF;font-weight:600;'>\1</strong>", text)
    text = re.sub(r"__(.+?)__",     r"<strong style='color:#FFFFFF;font-weight:600;'>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"_(.+?)_",   r"<em>\1</em>", text)
    text = re.sub(
        r"`(.+?)`",
        r"<code style='font-family:\"DM Mono\",monospace;color:#DFC080;"
        r"background:#111A2E;padding:2px 6px;border-radius:3px;font-size:0.85em'>\1</code>",
        text,
    )
    text = re.sub(
        r'"([^"]+)"',
        r'<span style="font-family:\'DM Mono\',monospace;color:#DFC080;font-size:0.92em">&ldquo;\1&rdquo;</span>',
        text,
    )
    return text


def _render_report_body(report_text: str) -> str:
    lines = report_text.strip().splitlines()
    html_parts: list[str] = []
    in_list = False

    atx_heading_re  = re.compile(r"^(#{1,6})\s+(.+)$")
    num_heading_re  = re.compile(r"^\s*(\d+)[.)]\s+(.+?)[:：]?\s*$", re.IGNORECASE)
    bullet_re       = re.compile(r"^\s*[-•*]\s+(.+)$")

    def close_list():
        nonlocal in_list
        if in_list:
            html_parts.append("</ul>")
            in_list = False

    def section_header(label: str, prefix: str = "") -> str:
        return f"""
        <div style="display:flex; align-items:center; gap:10px; margin: 24px 0 10px;">
          {"" if not prefix else f'''<span style="font-family:'DM Mono',monospace;font-size:0.75rem;
              background:#1E2E4A;color:#70A3E0;padding:4px 8px;border-radius:4px;font-weight:600;">{prefix}</span>'''}
          <span style="font-size:0.85rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
              color:#94A3B8;border-bottom:1px solid #1E2D4A;flex:1;padding-bottom:6px;">{label}</span>
        </div>"""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_list()
            continue

        m_atx = atx_heading_re.match(stripped)
        if m_atx:
            close_list()
            level = len(m_atx.group(1))
            title = _md_inline(m_atx.group(2).strip().rstrip(":"))
            if level <= 2:
                html_parts.append(f"""
                <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;
                    color:#E2E8F0;margin:24px 0 8px;letter-spacing:-0.01em;">{title}</div>""")
            else:
                html_parts.append(section_header(title.upper()))
            continue

        m_num = num_heading_re.match(stripped)
        if m_num:
            close_list()
            num   = m_num.group(1).zfill(2)
            title = _md_inline(m_num.group(2).strip().rstrip(":")).upper()
            html_parts.append(section_header(title, prefix=num))
            continue

        m_bul = bullet_re.match(stripped)
        if m_bul:
            if not in_list:
                html_parts.append('<ul style="margin:8px 0 0 0;padding:0;list-style:none;">')
                in_list = True
            content = _md_inline(m_bul.group(1).strip())
            content = re.sub(
                r"^(?!<strong)([^<:]+:)",
                r"<strong style='color:#70A3E0'>\1</strong>",
                content,
                count=1,
              )
            html_parts.append(f"""
            <li style="display:flex;gap:12px;align-items:baseline;padding:6px 0;
                border-bottom:1px solid #111A2E;font-size:0.95rem;color:#D1D5DB;line-height:1.65;">
              <span style="color:#5687BC;font-size:0.8rem;flex-shrink:0;margin-top:2px">&#9670;</span>
              <span>{content}</span>
            </li>""")
            continue

        close_list()
        content = _md_inline(stripped)
        html_parts.append(f"""
        <p style="font-size:0.95rem;color:#D1D5DB;line-height:1.8;margin:8px 0;">{content}</p>""")

    close_list()
    return "\n".join(html_parts)


# ================================================================== #
#  Left-panel visualization HTML builder                               #
# ================================================================== #

def _build_viz_panel(
    anomaly_score: float,
    threshold: float,
    is_anomaly: bool,
    score_map: np.ndarray,
    category: str,
) -> str:
    """
    Build a rich HTML visualization block for the left panel.
    Includes:
      - Animated SVG score ring
      - Spatial zone radar bars (derived from score_map quadrants)
      - Detection history sparkline (from _detection_history)
      - Statistical distribution mini-chart (inline SVG)
    """
    accent = "#E8674A" if is_anomaly else "#4ABFA8"
    accent_dim = "#5C2318" if is_anomaly else "#13412D"
    verdict_label = "ANOMALY" if is_anomaly else "NORMAL"

    # ── Score ring ────────────────────────────────────────────────
    # Normalise to 0–100 against 2× threshold
    ring_pct = min(anomaly_score / (threshold * 2 + 1e-8), 1.0)
    ring_deg = ring_pct * 360
    # SVG arc path for a ring (r=42, cx=cy=52)
    R, CX, CY = 42, 52, 52
    circumference = 2 * 3.14159 * R
    filled_dash = circumference * ring_pct
    empty_dash  = circumference * (1 - ring_pct)

    ring_svg = f"""
    <svg width="104" height="104" viewBox="0 0 104 104" style="overflow:visible;">
      <circle cx="{CX}" cy="{CY}" r="{R}" fill="none" stroke="#0A1525" stroke-width="9"/>
      <circle cx="{CX}" cy="{CY}" r="{R}" fill="none"
              stroke="{accent}" stroke-width="9" stroke-linecap="round"
              stroke-dasharray="{filled_dash:.2f} {empty_dash:.2f}"
              transform="rotate(-90 {CX} {CY})"
              style="transition:stroke-dasharray 0.8s cubic-bezier(.4,0,.2,1);">
        <animate attributeName="stroke-dasharray"
          from="0 {circumference:.2f}"
          to="{filled_dash:.2f} {empty_dash:.2f}"
          dur="1.1s" fill="freeze" calcMode="spline"
          keySplines="0.4 0 0.2 1"/>
      </circle>
      <text x="{CX}" y="{CY - 6}" text-anchor="middle"
            style="font-family:'DM Mono',monospace;font-size:14px;font-weight:600;fill:{accent};">
        {anomaly_score:.4f}
      </text>
      <text x="{CX}" y="{CY + 10}" text-anchor="middle"
            style="font-family:'DM Sans',sans-serif;font-size:9px;fill:#8AAAD4;letter-spacing:0.1em;text-transform:uppercase;">
        score
      </text>
    </svg>"""

    # ── Spatial zone bars (score_map quadrants) ───────────────────
    # Split the score map into a 3×3 grid; compute mean per zone
    h, w = score_map.shape
    zones_labels = ["TL", "TC", "TR", "ML", "MC", "MR", "BL", "BC", "BR"]
    zones_full   = ["Top-Left", "Top-Center", "Top-Right",
                    "Mid-Left", "Mid-Center", "Mid-Right",
                    "Bot-Left", "Bot-Center", "Bot-Right"]
    rh, rw = h // 3, w // 3
    zone_means = []
    for row in range(3):
        for col in range(3):
            patch = score_map[row*rh:(row+1)*rh, col*rw:(col+1)*rw]
            zone_means.append(float(patch.mean()))

    z_min = min(zone_means)
    z_max = max(zone_means) + 1e-8
    zone_pcts = [(z - z_min) / (z_max - z_min) * 100 for z in zone_means]

    zone_bars_html = ""
    for i, (label, full, pct, mean) in enumerate(zip(zones_labels, zones_full, zone_pcts, zone_means)):
        bar_color = "#E8674A" if mean >= threshold * 0.8 else ("#C9A96E" if mean >= threshold * 0.4 else "#4ABFA8")
        zone_bars_html += f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;" title="{full}: {mean:.5f}">
          <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#8AAAD4;
              min-width:24px;text-align:right;">{label}</span>
          <div style="flex:1;background:#0A1525;border-radius:2px;height:8px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:100%;background:{bar_color};border-radius:2px;
                transition:width 0.9s cubic-bezier(.4,0,.2,1);"></div>
          </div>
          <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#D8E0F4;
              min-width:48px;">{mean:.4f}</span>
        </div>"""

    # ── Detection history sparkline (inline SVG) ──────────────────
    history = _detection_history.copy()
    if len(history) >= 2:
        scores = [h["score"] for h in history]
        threshs = [h["threshold"] for h in history]
        n = len(scores)
        s_min = min(scores) * 0.9
        s_max = max(scores) * 1.1 + 1e-8
        spark_w, spark_h = 260, 54

        def sx(i):   return 8 + (i / max(n - 1, 1)) * (spark_w - 16)
        def sy(v):   return spark_h - 8 - (v - s_min) / (s_max - s_min) * (spark_h - 16)

        # Score line
        pts = " ".join(f"{sx(i):.1f},{sy(s):.1f}" for i, s in enumerate(scores))
        # Threshold line (use last threshold)
        t_y = sy(threshs[-1])
        # Dots (coloured by anomaly status)
        dots_svg = ""
        for i, h in enumerate(history):
            col = "#E8674A" if h["anomaly"] else "#4ABFA8"
            dots_svg += f'<circle cx="{sx(i):.1f}" cy="{sy(h["score"]):.1f}" r="3" fill="{col}" stroke="#080D1A" stroke-width="1"/>'

        sparkline_svg = f"""
        <svg width="{spark_w}" height="{spark_h}" viewBox="0 0 {spark_w} {spark_h}"
             style="overflow:visible;width:100%;">
          <defs>
            <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="{accent}" stop-opacity="0.25"/>
              <stop offset="100%" stop-color="{accent}" stop-opacity="0"/>
            </linearGradient>
          </defs>
          <line x1="8" y1="{t_y:.1f}" x2="{spark_w-8}" y2="{t_y:.1f}"
                stroke="#C9A96E" stroke-width="0.8" stroke-dasharray="4 3" opacity="0.7"/>
          <polyline points="{pts}" fill="none" stroke="{accent}" stroke-width="1.5"
                    stroke-linejoin="round" stroke-linecap="round" opacity="0.8"/>
          {dots_svg}
          <text x="8" y="{spark_h - 1}" style="font-family:'DM Mono',monospace;font-size:8px;fill:#64748B;">
            {history[0]['ts']}
          </text>
          <text x="{spark_w - 8}" y="{spark_h - 1}" text-anchor="end"
                style="font-family:'DM Mono',monospace;font-size:8px;fill:#64748B;">
            {history[-1]['ts']}
          </text>
        </svg>"""

        # History log (last 5)
        log_rows = ""
        for entry in reversed(history[-5:]):
            c = "#E8674A" if entry["anomaly"] else "#4ABFA8"
            v = "ANOM" if entry["anomaly"] else "OK"
            log_rows += f"""
            <div style="display:flex;align-items:center;gap:6px;padding:4px 0;border-bottom:1px solid #080D1A;">
              <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#64748B;min-width:54px;">{entry['ts']}</span>
              <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#94A3B8;flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">{entry['category']}</span>
              <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#D8E0F4;">{entry['score']:.4f}</span>
              <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{c};min-width:32px;text-align:right;font-weight:600;">{v}</span>
            </div>"""

        history_html = f"""
        <div style="margin-bottom:4px;">
          {sparkline_svg}
        </div>
        <div style="margin-top:4px;">{log_rows}</div>"""
    else:
        history_html = """
        <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#475569;
            padding:12px 0;text-align:center;letter-spacing:0.06em;">
            — run more detections to build history —</div>"""

    # ── Score distribution mini histogram ─────────────────────────
    # Flatten score_map and bucket into 12 bins
    flat = score_map.flatten()
    bins = 12
    hist_vals, bin_edges = np.histogram(flat, bins=bins)
    hist_max = max(hist_vals) + 1
    hist_h = 48
    hist_w_each = 14
    hist_gap = 2
    total_bar_w = bins * (hist_w_each + hist_gap)
    hist_bars = ""
    for i, (v, edge) in enumerate(zip(hist_vals, bin_edges[:-1])):
        bar_h = int((v / hist_max) * hist_h)
        bx = i * (hist_w_each + hist_gap)
        is_above_thresh = edge >= threshold
        bar_col = "#E8674A" if is_above_thresh else "#1E3550"
        hist_bars += f"""
        <rect x="{bx}" y="{hist_h - bar_h}" width="{hist_w_each}" height="{bar_h}"
              rx="2" fill="{bar_col}" opacity="0.9"/>"""

    # Threshold marker on histogram
    t_norm_x = max(0, min((threshold - flat.min()) / (flat.max() - flat.min() + 1e-8), 1.0))
    t_marker_x = t_norm_x * total_bar_w

    histogram_svg = f"""
    <svg width="100%" viewBox="0 0 {total_bar_w} {hist_h + 14}"
         style="overflow:visible;">
      {hist_bars}
      <line x1="{t_marker_x:.1f}" y1="0" x2="{t_marker_x:.1f}" y2="{hist_h}"
            stroke="#C9A96E" stroke-width="1.2" stroke-dasharray="3 2"/>
      <text x="{t_marker_x:.1f}" y="{hist_h + 10}" text-anchor="middle"
            style="font-family:'DM Mono',monospace;font-size:8px;fill:#C9A96E;">▲ thr</text>
    </svg>"""

    # ── Assemble full panel ───────────────────────────────────────
    return f"""
    <div style="font-family:'DM Sans',sans-serif;padding:0;">

      <!-- ── Score Ring + Verdict chip ── -->
      <div style="display:flex;align-items:center;gap:16px;
          background:{accent_dim};border:1px solid {accent};border-radius:8px;
          padding:14px 16px;margin-bottom:14px;">
        {ring_svg}
        <div style="flex:1;min-width:0;">
          <div style="font-size:0.75rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;
              color:{accent};margin-bottom:2px;">{verdict_label}</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
              color:#E2E8F0;letter-spacing:-0.02em;line-height:1.2;">{category}</div>
          <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;">
            <div style="background:#0A1020;border:1px solid #1A2A45;border-radius:3px;
                padding:4px 8px;">
              <span style="font-size:0.7rem;color:#8AAAD4;letter-spacing:0.1em;text-transform:uppercase;">Score</span>
              <span style="font-family:'DM Mono',monospace;font-size:0.85rem;color:{accent};margin-left:5px;font-weight:500;">{anomaly_score:.5f}</span>
            </div>
            <div style="background:#0A1020;border:1px solid #1A2A45;border-radius:3px;
                padding:4px 8px;">
              <span style="font-size:0.7rem;color:#8AAAD4;letter-spacing:0.1em;text-transform:uppercase;">Thresh</span>
              <span style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#4A7AAF;margin-left:5px;font-weight:500;">{threshold:.5f}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- ── Spatial Zone Analysis ── -->
      <div style="background:#0A1020;border:1px solid #131D30;border-radius:6px;
          padding:12px 14px;margin-bottom:12px;">
        <div style="font-size:0.8rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
            color:#5A7099;margin-bottom:10px;">&#9632;&nbsp; Spatial Zone Analysis</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:12px;">
          {"".join([
            f'''<div style="background:#060D1A;border:1px solid #0E1A2C;border-radius:3px;
                padding:6px;text-align:center;" title="{z}: {m:.5f}">
              <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#64748B;font-weight:500;">{z}</div>
              <div style="font-family:'DM Mono',monospace;font-size:0.75rem;
                  color:{'#E8674A' if m >= threshold * 0.8 else '#4ABFA8'};font-weight:600;">{m:.4f}</div>
            </div>'''
            for z, m in zip(zones_labels, zone_means)
          ])}
        </div>
        {zone_bars_html}
      </div>

      <!-- ── Score Distribution Histogram ── -->
      <div style="background:#0A1020;border:1px solid #131D30;border-radius:6px;
          padding:12px 14px;margin-bottom:12px;">
        <div style="font-size:0.8rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
            color:#5A7099;margin-bottom:8px;">&#9632;&nbsp; Score Distribution</div>
        {histogram_svg}
        <div style="display:flex;gap:12px;margin-top:8px;justify-content:space-between;">
          <div>
            <span style="font-size:0.72rem;color:#8AAAD4;letter-spacing:0.08em;text-transform:uppercase;font-weight:500;">min</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#D8E0F4;margin-left:4px;">{float(flat.min()):.4f}</span>
          </div>
          <div>
            <span style="font-size:0.72rem;color:#8AAAD4;letter-spacing:0.08em;text-transform:uppercase;font-weight:500;">mean</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#D8E0F4;margin-left:4px;">{float(flat.mean()):.4f}</span>
          </div>
          <div>
            <span style="font-size:0.72rem;color:#8AAAD4;letter-spacing:0.08em;text-transform:uppercase;font-weight:500;">max</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#E8674A;margin-left:4px;">{float(flat.max()):.4f}</span>
          </div>
          <div>
            <span style="font-size:0.72rem;color:#8AAAD4;letter-spacing:0.08em;text-transform:uppercase;font-weight:500;">p95</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#C9A96E;margin-left:4px;">{float(np.percentile(flat, 95)):.4f}</span>
          </div>
        </div>
      </div>

      <!-- ── Detection History ── -->
      <div style="background:#0A1020;border:1px solid #131D30;border-radius:6px;
          padding:12px 14px;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
          <span style="font-size:0.8rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
              color:#5A7099;">&#9632;&nbsp; Run History</span>
          <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#64748B;">
            last {len(history)} runs</span>
        </div>
        {history_html}
      </div>

    </div>"""


def _build_viz_panel_empty() -> str:
    """Placeholder panel shown before any inference."""
    return """
    <div style="font-family:'DM Sans',sans-serif;padding:8px 0;">

      <!-- Placeholder rings + bars to show what will appear -->
      <div style="background:#060C18;border:1px dashed #1E2D4A;border-radius:8px;
          padding:20px 16px;margin-bottom:12px;text-align:center;">
        <svg width="80" height="80" viewBox="0 0 80 80" style="opacity:0.25;margin-bottom:10px;">
          <circle cx="40" cy="40" r="32" fill="none" stroke="#4ABFA8" stroke-width="8"
                  stroke-dasharray="60 141"/>
          <circle cx="40" cy="40" r="32" fill="none" stroke="#1A2A45" stroke-width="8"/>
        </svg>
        <div style="font-size:0.8rem;color:#4A5F84;letter-spacing:0.1em;text-transform:uppercase;
            margin-top:4px;font-weight:500;">Score ring · after detection</div>
      </div>

      <div style="background:#060C18;border:1px dashed #1E2D4A;border-radius:6px;
          padding:14px 14px;margin-bottom:10px;">
        <div style="font-size:0.8rem;color:#4A5F84;letter-spacing:0.1em;text-transform:uppercase;
            margin-bottom:8px;font-weight:500;">Spatial zone analysis · after detection</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;opacity:0.25;margin-bottom:8px;">
          """ + "".join([
        f'<div style="background:#0A1525;border-radius:3px;height:24px;"></div>'
        for _ in range(9)
    ]) + """
        </div>
        """ + "".join([
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;opacity:0.15;">'
        f'<div style="min-width:22px;height:6px;background:#0A1525;border-radius:2px;"></div>'
        f'<div style="flex:1;background:#0A1525;border-radius:2px;height:6px;"></div>'
        f'</div>'
        for _ in range(4)
    ]) + """
      </div>

      <div style="background:#060C18;border:1px dashed #1E2D4A;border-radius:6px;
          padding:14px 14px;margin-bottom:10px;">
        <div style="font-size:0.8rem;color:#4A5F84;letter-spacing:0.1em;text-transform:uppercase;
            margin-bottom:8px;font-weight:500;">Score distribution · after detection</div>
        <div style="display:flex;gap:3px;align-items:flex-end;height:36px;opacity:0.15;">
          """ + "".join([
        f'<div style="flex:1;background:#1E3550;border-radius:2px;height:{h}%;"></div>'
        for h in [20, 35, 60, 85, 100, 90, 65, 45, 30, 20, 12, 8]
    ]) + """
        </div>
      </div>

      <div style="background:#060C18;border:1px dashed #1E2D4A;border-radius:6px;
          padding:14px 14px;">
        <div style="font-size:0.8rem;color:#4A5F84;letter-spacing:0.1em;text-transform:uppercase;
            margin-bottom:8px;font-weight:500;">Run history · after detection</div>
        <div style="opacity:0.12;">""" + "".join([
        '<div style="height:5px;background:#1A2A45;border-radius:2px;margin-bottom:5px;"></div>'
        for _ in range(4)
    ]) + """
        </div>
      </div>

    </div>"""


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
    orig_w, orig_h = original_pil.size
    orig_np = np.array(original_pil.convert("RGB"))

    if threshold is not None:
        s_min = 0.0
        s_max = max(threshold, score_map.max())
    else:
        s_min, s_max = score_map.min(), score_map.max()

    if s_max > s_min:
        norm_map = (score_map - s_min) / (s_max - s_min)
    else:
        norm_map = np.zeros_like(score_map)

    norm_resized = cv2.resize(norm_map, (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)

    import matplotlib
    cmap    = matplotlib.colormaps.get_cmap(cmap_name)
    colored = (cmap(norm_resized)[:, :, :3] * 255).astype(np.uint8)

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
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    fig.patch.set_facecolor("#0E1628")
    ax.set_facecolor("#0E1628")

    norm_score = min(anomaly_score / (threshold * 2 + 1e-8), 1.0)
    colour     = "#E8674A" if anomaly_score >= threshold else "#4ABFA8"

    ax.barh([0], [norm_score], color=colour, height=0.55, alpha=0.9)
    ax.barh([0], [1],          color="#1E2D4A", height=0.55, alpha=0.6)
    ax.axvline(0.5, color="#C9A96E", lw=1.8, ls="--")
    ax.set_xlim(0, 1)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100,
                facecolor="#0E1628")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def run_inference(
    image: Optional[Image.Image],
    category: str,
) -> Tuple[Image.Image, Image.Image, str, str, str]:
    """Returns: overlay, score_bar, verdict_html, metrics_text, viz_panel_html"""
    if image is None:
        dummy = Image.new("RGB", (224, 224), "#0E1628")
        return dummy, dummy, "<p>Please upload an image.</p>", "", _build_viz_panel_empty()

    if category not in MODELS:
        dummy = Image.new("RGB", (224, 224), "#0E1628")
        return (
            dummy, dummy,
            f"<p style='color:#E8674A'>No trained model for '{category}'."
            f" Run train.py first.</p>",
            "",
            _build_viz_panel_empty(),
        )

    pc = MODELS[category]

    img_pil   = image.convert("RGB")
    segment_enabled = (
        CFG["dataset"].get("segmentation", {}).get("enabled", False)
        and category in CFG["dataset"].get("segmentation", {}).get("categories", [])
    )
    if segment_enabled:
        from model import apply_otsu_mask
        img_pil = apply_otsu_mask(img_pil)
    img_tensor = TRANSFORM(img_pil).unsqueeze(0)

    anomaly_score, score_map = pc.predict_image(img_tensor)
    threshold                = pc.threshold
    is_anomaly               = anomaly_score >= threshold

    # Push to history BEFORE building panel
    _push_history(anomaly_score, threshold, is_anomaly, category)

    overlay   = make_heatmap_overlay(img_pil, score_map, threshold=threshold)
    score_bar = make_score_bar(anomaly_score, threshold)

    if is_anomaly:
        verdict_html = f"""
        <div style="background:#160C0A;border:1px solid #5C2318;border-left:3px solid #E8674A;
            padding:18px 20px;border-radius:6px;font-family:'DM Sans',sans-serif;">
          <div style="display:flex;align-items:center;gap:10px;font-size:1.1rem;font-weight:600;
              color:#E8674A;letter-spacing:0.04em;text-transform:uppercase;">
            <span style="font-size:1.3rem">⚠</span> Defect Detected
          </div>
          <div style="color:#C4907E;margin-top:8px;font-size:0.9rem;line-height:1.5">
            Anomaly score
            <span style="font-family:'DM Mono',monospace;color:#E8A090">{anomaly_score:.4f}</span>
            exceeds threshold
            <span style="font-family:'DM Mono',monospace;color:#E8A090">{threshold:.4f}</span>
          </div>
          <div style="margin-top:6px;font-size:0.78rem;color:#7A5A52;letter-spacing:0.06em;text-transform:uppercase;">
            Category · {category}
          </div>
        </div>"""
    else:
        verdict_html = f"""
        <div style="background:#07160F;border:1px solid #13412D;border-left:3px solid #4ABFA8;
            padding:18px 20px;border-radius:6px;font-family:'DM Sans',sans-serif;">
          <div style="display:flex;align-items:center;gap:10px;font-size:1.1rem;font-weight:600;
              color:#4ABFA8;letter-spacing:0.04em;text-transform:uppercase;">
            <span style="font-size:1.3rem">✓</span> Normal — No Defect
          </div>
          <div style="color:#7ABFB3;margin-top:8px;font-size:0.9rem;line-height:1.5">
            Anomaly score
            <span style="font-family:'DM Mono',monospace;color:#90D4CA">{anomaly_score:.4f}</span>
            is below threshold
            <span style="font-family:'DM Mono',monospace;color:#90D4CA">{threshold:.4f}</span>
          </div>
          <div style="margin-top:6px;font-size:0.78rem;color:#3A6A5E;letter-spacing:0.06em;text-transform:uppercase;">
            Category · {category}
          </div>
        </div>"""

    metrics_text = (
        f"Anomaly Score : {anomaly_score:.6f}\n"
        f"Threshold     : {threshold:.6f}\n"
        f"Verdict       : {'ANOMALY' if is_anomaly else 'NORMAL'}\n"
        f"Category      : {category}\n"
        f"Model trained : ✓"
    )

    viz_html = _build_viz_panel(anomaly_score, threshold, is_anomaly, score_map, category)

    return overlay, score_bar, verdict_html, metrics_text, viz_html


def _risk_config(risk: str) -> dict:
    r = risk.strip().upper()
    if "CRITICAL" in r or "HIGH" in r:
        return {
            "accent":  "#E8674A", "bg": "#160C0A", "border": "#5C2318",
            "tag_bg": "#2A1008", "tag_txt": "#E8A090",
            "icon": "&#9888;", "label": risk, "bar_pct": "90%", "bar_col": "#E8674A",
        }
    elif "MEDIUM" in r or "MODERATE" in r:
        return {
            "accent":  "#C9A96E", "bg": "#141008", "border": "#4A3810",
            "tag_bg": "#2A2008", "tag_txt": "#DFC080",
            "icon": "&#9650;", "label": risk, "bar_pct": "55%", "bar_col": "#C9A96E",
        }
    else:
        return {
            "accent":  "#4ABFA8", "bg": "#07160F", "border": "#13412D",
            "tag_bg": "#0A2018", "tag_txt": "#90D4CA",
            "icon": "&#10003;", "label": risk, "bar_pct": "18%", "bar_col": "#4ABFA8",
        }


def run_spec_analysis(
    image: Optional[Image.Image],
    category: str,
) -> str:
    if image is None:
        return """
        <div style="font-family:'DM Sans',sans-serif;color:#E8674A;font-size:0.88rem;
            padding:14px;border:1px solid #5C2318;border-radius:6px;background:#160C0A;">
            Please upload an image first.</div>"""

    if category not in MODELS:
        return f"""
        <div style="font-family:'DM Sans',sans-serif;color:#E8674A;font-size:0.88rem;
            padding:14px;border:1px solid #5C2318;border-radius:6px;background:#160C0A;">
            No trained model for <strong>{category}</strong>. Run <code>train.py</code> first.</div>"""

    pc = MODELS[category]
    img_pil = image.convert("RGB")
    segment_enabled = (
        CFG["dataset"].get("segmentation", {}).get("enabled", False)
        and category in CFG["dataset"].get("segmentation", {}).get("categories", [])
    )
    if segment_enabled:
        from model import apply_otsu_mask
        img_pil = apply_otsu_mask(img_pil)
    img_tensor = TRANSFORM(img_pil).unsqueeze(0)
    anomaly_score, _ = pc.predict_image(img_tensor)
    threshold  = pc.threshold
    is_anomaly = anomaly_score >= threshold
    verdict    = "ANOMALY (DEFECTIVE)" if is_anomaly else "NORMAL"

    report = spec_analyzer.analyze_with_spec(img_pil, category, anomaly_score, verdict)
    risk, _ = spec_analyzer.get_risk_level(report)
    cfg     = _risk_config(risk)

    score_pct      = min(int((anomaly_score / (threshold * 2 + 1e-8)) * 100), 100)
    score_display  = f"{anomaly_score:.4f}"
    thresh_display = f"{threshold:.4f}"

    header_html = f"""
    <div style="font-family:'DM Sans',sans-serif;margin-bottom:4px;">
      <div style="background:{cfg['bg']};border:1px solid {cfg['border']};
          border-left:4px solid {cfg['accent']};border-radius:6px;padding:16px 20px;
          display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;">
        <div>
          <div style="font-size:0.8rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;
              color:{cfg['accent']};margin-bottom:4px;">{cfg['icon']} &nbsp;AI Risk Assessment</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
              color:{cfg['accent']};letter-spacing:-0.02em;line-height:1;">{cfg['label'].upper()}</div>
          <div style="font-size:0.85rem;color:#8AAAD4;margin-top:5px;letter-spacing:0.02em;">
            Based on direct datasheet specification comparison</div>
        </div>
        <div style="display:flex;flex-direction:column;gap:6px;align-items:flex-end;flex-shrink:0;">
          <div style="background:{cfg['tag_bg']};border:1px solid {cfg['border']};
              border-radius:4px;padding:6px 14px;text-align:center;">
            <div style="font-size:0.75rem;color:#8AAAD4;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px;">
              Anomaly Score</div>
            <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:{cfg['tag_txt']};font-weight:500;">
              {score_display}</div>
          </div>
          <div style="background:#0A1020;border:1px solid #1A2A45;border-radius:4px;padding:6px 14px;text-align:center;">
            <div style="font-size:0.75rem;color:#8AAAD4;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px;">
              Threshold</div>
            <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:#4A7AAF;font-weight:500;">
              {thresh_display}</div>
          </div>
        </div>
      </div>
      <div style="margin:12px 0 0;">
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
          <span style="font-size:0.8rem;color:#8AAAD4;letter-spacing:0.1em;text-transform:uppercase;font-weight:500;">Score Gauge</span>
          <span style="font-family:'DM Mono',monospace;font-size:0.8rem;color:{cfg['accent']};font-weight:500;">
            {score_pct}% of 2× threshold</span>
        </div>
        <div style="background:#0A1020;border:1px solid #131D30;border-radius:3px;height:8px;
            overflow:hidden;position:relative;">
          <div style="width:{cfg['bar_pct']};height:100%;background:{cfg['bar_col']};border-radius:3px;"></div>
          <div style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#C9A96E;opacity:0.8;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:3px;">
          <span style="font-size:0.72rem;color:#64748B;font-weight:500;">0</span>
          <span style="font-size:0.72rem;color:#C9A96E;margin-left:calc(50% - 20px);font-weight:500;">&#9660; threshold</span>
          <span style="font-size:0.72rem;color:#64748B;font-weight:500;">2×</span>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;margin:20px 0 12px;">
        <div style="flex:1;height:1px;background:#0E1628;"></div>
        <span style="font-size:0.8rem;color:#5A7099;letter-spacing:0.14em;text-transform:uppercase;font-weight:700;">
          Inspection Report</span>
        <div style="flex:1;height:1px;background:#0E1628;"></div>
      </div>
    </div>"""

    body_html = _render_report_body(report)

    return f"""
    <div style="font-family:'DM Sans',sans-serif;padding:2px 0;">
      {header_html}
      <div style="padding:0 2px;">{body_html}</div>
    </div>"""


def reset_spec_report() -> str:
    return """<div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#1E2D4A;
        padding:16px 0;letter-spacing:0.04em;">
        — Run detection first, then click Specification Analysis —</div>"""


# ================================================================== #
#  PDF Report Generator                                                #
# ================================================================== #
def generate_pdf_report(
    image: Optional[Image.Image],
    category: str,
) -> Optional[str]:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
        Table, TableStyle, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    if image is None or category not in MODELS:
        return None

    pc = MODELS[category]
    img_pil = image.convert("RGB")
    segment_enabled = (
        CFG["dataset"].get("segmentation", {}).get("enabled", False)
        and category in CFG["dataset"].get("segmentation", {}).get("categories", [])
    )
    if segment_enabled:
        from model import apply_otsu_mask
        img_pil = apply_otsu_mask(img_pil)
    img_tensor = TRANSFORM(img_pil).unsqueeze(0)
    anomaly_score, _ = pc.predict_image(img_tensor)
    threshold  = pc.threshold
    is_anomaly = anomaly_score >= threshold
    verdict    = "ANOMALY (DEFECTIVE)" if is_anomaly else "NORMAL"

    report_text = spec_analyzer.analyze_with_spec(img_pil, category, anomaly_score, verdict)
    risk, _     = spec_analyzer.get_risk_level(report_text)
    cfg         = _risk_config(risk)

    C_BG      = colors.HexColor("#080D1A")
    C_SURFACE = colors.HexColor("#0E1628")
    C_BORDER  = colors.HexColor("#1A2A45")
    C_TEXT    = colors.HexColor("#C8D0E8")
    C_MUTED   = colors.HexColor("#5A7099")
    C_ACCENT  = colors.HexColor(cfg["accent"])
    C_GOLD    = colors.HexColor("#C9A96E")

    risk_upper = risk.strip().upper()

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = tempfile.mktemp(suffix=f"_anomaly_report_{category}_{ts}.pdf")

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm,  bottomMargin=16*mm,
        title=f"Anomaly Inspection Report — {category}",
        author="Industrial Anomaly Detector · PatchCore",
    )
    W = A4[0] - 36*mm

    styles = getSampleStyleSheet()
    def S(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    sTitle     = S("Title",  fontSize=22, textColor=C_TEXT,   fontName="Helvetica-Bold", spaceAfter=2,  leading=26)
    sSubtitle  = S("Sub",    fontSize=9,  textColor=C_MUTED,  fontName="Helvetica",      spaceAfter=10, leading=12)
    sTagline   = S("Tag",    fontSize=7,  textColor=C_GOLD,   fontName="Helvetica-Bold", spaceBefore=2, spaceAfter=8, leading=10)
    sRiskLabel = S("RL",     fontSize=8,  textColor=C_ACCENT, fontName="Helvetica-Bold", leading=10,    spaceAfter=2)
    sRiskValue = S("RV",     fontSize=26, textColor=C_ACCENT, fontName="Helvetica-Bold", leading=28,    spaceAfter=4)
    sSection   = S("Sec",    fontSize=7,  textColor=C_MUTED,  fontName="Helvetica-Bold", spaceBefore=14,spaceAfter=4, leading=9)
    sBody      = S("Body",   fontSize=9,  textColor=C_TEXT,   fontName="Helvetica",      leading=14,    spaceAfter=4)
    sBullet    = S("Bul",    fontSize=9,  textColor=colors.HexColor("#8090B0"),
                   fontName="Helvetica", leading=13, leftIndent=12, spaceAfter=2)
    sMetaKey   = S("MK",     fontSize=7.5, textColor=C_MUTED, fontName="Helvetica-Bold", leading=10)
    sMetaVal   = S("MV",     fontSize=9,   textColor=C_TEXT,  fontName="Helvetica-Bold", leading=12)
    sFooter    = S("Foot",   fontSize=7,   textColor=colors.HexColor("#1E2D4A"),
                   fontName="Helvetica", alignment=TA_CENTER, leading=9)

    story = []
    ts_str = datetime.datetime.now().strftime("%d %b %Y  %H:%M")

    hdr_left = [
        Paragraph("INDUSTRIAL ANOMALY DETECTOR", sTagline),
        Paragraph("Inspection Report", sTitle),
        Paragraph(f"Component: {category.upper()}  ·  Generated: {ts_str}", sSubtitle),
    ]
    hdr_right = [
        Paragraph("POWERED BY", sTagline),
        Paragraph("PatchCore", S("PC", fontSize=14, textColor=C_GOLD, fontName="Helvetica-Bold", leading=16)),
        Paragraph("ResNet-50 · MVTec-AD", sSubtitle),
    ]
    hdr_tbl = Table([[hdr_left, hdr_right]], colWidths=[W*0.65, W*0.35])
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_SURFACE),
        ("BOX",        (0,0),(-1,-1), 0.5, C_BORDER),
        ("LEFTPADDING",  (0,0),(-1,-1), 14),
        ("RIGHTPADDING", (0,0),(-1,-1), 14),
        ("TOPPADDING",   (0,0),(-1,-1), 12),
        ("BOTTOMPADDING",(0,0),(-1,-1), 12),
        ("VALIGN",     (0,0),(-1,-1), "TOP"),
        ("ALIGN",      (1,0),(1,0),   "RIGHT"),
    ]))
    story.append(hdr_tbl)
    story.append(Spacer(1, 8*mm))

    score_pct = min(int((anomaly_score / (threshold * 2 + 1e-8)) * 100), 100)
    risk_cell = [
        Paragraph("AI RISK ASSESSMENT", sRiskLabel),
        Paragraph(risk_upper, sRiskValue),
        Paragraph("Based on direct datasheet specification comparison", sSubtitle),
    ]
    metrics_data = [
        [Paragraph("ANOMALY SCORE", sMetaKey), Paragraph("THRESHOLD", sMetaKey),
         Paragraph("VERDICT", sMetaKey),        Paragraph("SCORE %", sMetaKey)],
        [Paragraph(f"{anomaly_score:.6f}", sMetaVal), Paragraph(f"{threshold:.6f}", sMetaVal),
         Paragraph(verdict, S("VD", fontSize=9, textColor=C_ACCENT, fontName="Helvetica-Bold", leading=12)),
         Paragraph(f"{score_pct}%", S("SP", fontSize=9, textColor=C_GOLD, fontName="Helvetica-Bold", leading=12))],
    ]
    metrics_tbl = Table(metrics_data, colWidths=[W*0.25]*4)
    metrics_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), C_SURFACE),
        ("BOX",          (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",    (0,0),(-1,-1), 0.3, colors.HexColor("#131D30")),
        ("TOPPADDING",   (0,0),(-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("ALIGN",        (0,0),(-1,-1), "CENTER"),
    ]))
    top_row = Table([[risk_cell, metrics_tbl]], colWidths=[W*0.38, W*0.62])
    top_row.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,0), colors.HexColor(cfg["bg"])),
        ("LEFTPADDING",   (0,0),(-1,-1), 14),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 12),
        ("BOTTOMPADDING", (0,0),(-1,-1), 12),
        ("BOX",  (0,0),(0,0), 2, C_ACCENT),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(top_row)
    story.append(Spacer(1, 5*mm))

    gauge_pct = min(score_pct / 100, 1.0)
    BAR_W    = W - 4*mm
    filled_w = max(BAR_W * gauge_pct, 0.1)
    empty_w  = max(BAR_W - filled_w, 0.1)
    gauge_tbl = Table([["", ""]], colWidths=[filled_w, empty_w], rowHeights=[5])
    gauge_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,0), C_ACCENT),
        ("BACKGROUND", (1,0),(1,0), colors.HexColor("#0A1020")),
        ("BOX",        (0,0),(-1,-1), 0.5, C_BORDER),
        ("TOPPADDING", (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
    ]))
    story.append(Paragraph("SCORE GAUGE  (gold = threshold at 50%)", sSection))
    story.append(gauge_tbl)
    story.append(Spacer(1, 6*mm))

    story.append(HRFlowable(width=W, thickness=0.5, color=C_BORDER, spaceAfter=4))
    story.append(Paragraph("INSPECTION REPORT", sSection))
    story.append(HRFlowable(width=W, thickness=0.5, color=C_BORDER, spaceBefore=4, spaceAfter=6))

    atx_re  = re.compile(r"^(#{1,6})\s+(.+)$")
    num_re  = re.compile(r"^\s*(\d+)[.)]\s+(.+?)[:：]?\s*$", re.IGNORECASE)
    bul_re  = re.compile(r"^\s*[-•*]\s+(.+)$")

    def pdf_clean(text: str) -> str:
        text = re.sub(r"^#{1,6}\s*", "", text.strip())
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.+?)__",     r"<b>\1</b>", text)
        text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
        text = re.sub(r"`(.+?)`",       r"\1",         text)
        return text

    for line in report_text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 3*mm))
            continue
        mh  = atx_re.match(stripped) or num_re.match(stripped)
        mb  = bul_re.match(stripped)
        if mh:
            groups = mh.groups()
            num_str = groups[0].zfill(2) if groups[0].isdigit() else "  "
            title   = pdf_clean(groups[1].strip().rstrip(":")).upper()
            story.append(KeepTogether([
                Spacer(1, 2*mm),
                Paragraph(f"<b>{num_str}  {title}</b>", sSection),
                HRFlowable(width=W, thickness=0.3, color=colors.HexColor("#131D30"), spaceAfter=3),
            ]))
        elif mb:
            content = pdf_clean(mb.group(1).strip())
            content = re.sub(r"^([^<:]+:)", r"<b>\1</b>", content, count=1)
            story.append(Paragraph(f"◆  {content}", sBullet))
        else:
            story.append(Paragraph(pdf_clean(stripped), sBody))

    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width=W, thickness=0.3, color=C_BORDER))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"PATCHCORE · RESNET-50 · MVTEC-AD  ·  Report generated {ts_str}  ·  "
        f"Category: {category.upper()}  ·  This report is AI-generated and should be reviewed by a qualified engineer.",
        sFooter,
    ))

    def draw_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_BG)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_bg, onLaterPages=draw_bg)
    return out_path


# ================================================================== #
#  Gradio UI                                                           #
# ================================================================== #
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@300;400;500&family=Syne:wght@600;700;800&display=swap');

*, *::before, *::after { font-family: 'DM Sans', sans-serif !important; box-sizing: border-box; }

body, .gradio-container, .gradio-container > .main {
    background: #080D1A !important; color: #C8D0E8 !important;
}
.gr-panel, .gr-box, .gr-padded, .block, .wrap {
    background: transparent !important; border: none !important;
}
.tab-nav { border-bottom: 1px solid #1A2340 !important; background: transparent !important; }

label span, .label-wrap, .gr-form label, span.svelte-1gfkfd6 {
    color: #8AAAD4 !important; font-size: 0.8rem !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important; font-weight: 600 !important;
}
input, select, textarea {
    background: #0E1628 !important; border: 1px solid #1A2A45 !important;
    color: #C8D0E8 !important; border-radius: 5px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #C9A96E !important; outline: none !important;
    box-shadow: 0 0 0 2px rgba(201,169,110,0.1) !important;
}
.wrap.svelte-1n5fhef, ul.options {
    background: #0E1628 !important; border: 1px solid #1A2A45 !important; border-radius: 5px !important;
}
.upload-container, .image-container, [data-testid="image"] {
    background: #0A1020 !important; border: 1px dashed #1E2E4E !important; border-radius: 8px !important;
}
.gr-textbox textarea {
    background: #0A1020 !important; border: 1px solid #1A2A45 !important; color: #8AAAD4 !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.88rem !important;
    line-height: 1.8 !important; border-radius: 5px !important; letter-spacing: 0.02em !important;
}
.gr-button-primary, button.primary {
    background: #C9A96E !important; border: none !important; color: #080D1A !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 0.9rem !important; letter-spacing: 0.06em !important; text-transform: uppercase !important;
    border-radius: 5px !important; padding: 10px 24px !important;
    transition: background 0.2s ease, transform 0.15s ease !important;
}
.gr-button-primary:hover, button.primary:hover {
    background: #DFC080 !important; transform: translateY(-1px) !important;
}
#spec_btn, button.secondary {
    background: #0E1628 !important; border: 1px solid #2A3D60 !important; color: #8AAAD4 !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    font-size: 0.88rem !important; letter-spacing: 0.05em !important; text-transform: uppercase !important;
    border-radius: 5px !important; padding: 9px 20px !important;
    transition: border-color 0.2s ease, color 0.2s ease, transform 0.15s ease !important;
}
#spec_btn:hover, button.secondary:hover {
    border-color: #C9A96E !important; color: #C9A96E !important;
    transform: translateY(-1px) !important; background: #0E1628 !important;
}
.gr-markdown, .prose, .md { color: #D1D5DB !important; }
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3, .md h1, .md h2, .md h3 {
    color: #E2E8F0 !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; letter-spacing: -0.01em !important;
}
.gr-markdown h3, .md h3 {
    font-size: 0.88rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important;
    font-weight: 600 !important; color: #8AAAD4 !important; margin-top: 1.4rem !important;
    border-bottom: 1px solid #1A2A45 !important; padding-bottom: 6px !important;
}
.gr-markdown p, .md p { font-size: 0.98rem !important; line-height: 1.8 !important; color: #D1D5DB !important; }
.gr-markdown code, .md code {
    font-family: 'DM Mono', monospace !important; font-size: 0.88rem !important;
    background: #111A2E !important; color: #DFC080 !important;
    padding: 2px 6px !important; border-radius: 3px !important;
}
hr { border: none !important; border-top: 1px solid #131D30 !important; margin: 20px 0 !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080D1A; }
::-webkit-scrollbar-thumb { background: #1E2D4A; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #2A3D60; }
.output-image img, img.svelte-rrgd5g {
    border-radius: 6px !important; border: 1px solid #131D30 !important;
}
#download_btn {
    background: #080D1A !important; border: 1px solid #C9A96E !important; color: #C9A96E !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 0.84rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important;
    border-radius: 5px !important; padding: 10px 20px !important;
    transition: background 0.2s ease, color 0.2s ease, transform 0.15s ease !important; margin-top: 10px !important;
}
#download_btn:hover { background: #C9A96E !important; color: #080D1A !important; transform: translateY(-1px) !important; }
#report_file { background: #0A1020 !important; border: 1px solid #1A2A45 !important; border-radius: 5px !important; margin-top: 6px !important; }
#report_file .file-preview { background: #0A1020 !important; color: #8AAAD4 !important; }
"""

HEADER_HTML = """
<div style="padding:36px 0 24px;font-family:'Syne','DM Sans',sans-serif;
    border-bottom:1px solid #131D30;margin-bottom:24px;">
  <div style="display:flex;align-items:baseline;gap:14px;margin-bottom:8px;">
    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#C9A96E;letter-spacing:0.18em;
        text-transform:uppercase;border:1px solid #3A2A10;padding:3px 10px;border-radius:2px;background:#120E06;">
      PatchCore · ResNet-50 · MVTec-AD</span>
  </div>
  <h1 style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;color:#D8E0F4;
      margin:0 0 6px;letter-spacing:-0.03em;line-height:1.1;">
    Industrial Anomaly<br><span style="color:#C9A96E;">Detection System</span>
  </h1>
  <p style="color:#3A4A6A;font-size:0.88rem;margin:0;letter-spacing:0.02em;">
    Upload a component image · Select category · Detect manufacturing defects
  </p>
</div>"""

INSTRUCTIONS_MD = """
### How to use

1. **Select category** — choose the component type matching your image.
2. **Upload image** — drag-and-drop or click to browse.
3. **Detect** — run anomaly detection and inspect results.
4. **Read the heatmap** — warm/red regions indicate potential defect sites.

> Train a PatchCore model first by running `python train.py` for your target category.
"""

SPEC_SECTION_MD = """
### AI Specification Analysis

Compare the physical component image against the manufacturer's official datasheet using AI.
"""


def build_demo() -> gr.Blocks:
    available   = list(MODELS.keys()) or CATEGORIES
    default_cat = available[0] if available else CATEGORIES[0]

    with gr.Blocks(
        title="Industrial Anomaly Detector",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="stone",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
        ),
    ) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):

            # ── Left column: Controls + Visualisation ─────────────
            with gr.Column(scale=1, min_width=320):
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
                    height=240,
                )
                detect_btn = gr.Button("Run Detection", variant="primary", size="lg")

                # ── Raw Metrics (collapsed) ───────────────────────
                metrics_box = gr.Textbox(
                    label="Raw Metrics",
                    lines=5,
                    interactive=False,
                    placeholder="Inference results will appear here …",
                )

                # ── NEW: Rich Visualisation Panel ─────────────────
                gr.HTML("""
                <div style="font-family:'Syne',sans-serif;font-size:0.72rem;font-weight:700;
                    letter-spacing:0.12em;text-transform:uppercase;color:#3A4A6A;
                    border-bottom:1px solid #131D30;padding-bottom:8px;margin:16px 0 12px;">
                  Analysis Visualisation</div>""")

                viz_panel = gr.HTML(
                    value=_build_viz_panel_empty(),
                    elem_id="viz_panel",
                )

            # ── Right column: Results ────────────────────────────
            with gr.Column(scale=1, min_width=320):

                gr.HTML("""
                <div style="font-family:'Syne',sans-serif;font-size:0.72rem;font-weight:700;
                    letter-spacing:0.12em;text-transform:uppercase;color:#3A4A6A;
                    border-bottom:1px solid #131D30;padding-bottom:8px;margin-bottom:16px;">
                  Detection Results</div>""")

                verdict_html = gr.HTML("""
                <div style="color:#2A3A5A;font-family:'DM Mono',monospace;
                    font-size:0.82rem;padding:12px 0;">— Awaiting input —</div>""")

                heatmap_out = gr.Image(
                    label="Anomaly Heatmap Overlay",
                    height=260,
                    interactive=False,
                )
                score_bar_out = gr.Image(
                    label="Score Bar  (gold marker = threshold)",
                    height=65,
                    interactive=False,
                )

                gr.HTML("<hr>")
                gr.Markdown(SPEC_SECTION_MD)

                spec_btn = gr.Button(
                    "Run Specification Analysis",
                    variant="secondary",
                    elem_id="spec_btn",
                )
                spec_report_out = gr.HTML(
                    value="""<div style="font-family:'DM Mono',monospace;font-size:0.8rem;
                        color:#1E2D4A;padding:16px 0;letter-spacing:0.04em;">
                        — Run detection first, then click Specification Analysis —</div>""",
                    elem_id="spec_report",
                )
                download_btn = gr.Button(
                    "Download PDF Report",
                    variant="secondary",
                    elem_id="download_btn",
                    visible=False,
                )
                report_file_out = gr.File(
                    label="Report Download",
                    visible=False,
                    elem_id="report_file",
                )

        # ── Examples ─────────────────────────────────────────────
        example_dir = Path("./sample_images")
        if example_dir.exists():
            sample_imgs = list(example_dir.glob("*.png")) + list(example_dir.glob("*.jpg"))
            if sample_imgs:
                gr.Examples(
                    examples=[[str(p), default_cat] for p in sample_imgs[:6]],
                    inputs=[input_img, category_dd],
                    label="Sample Images",
                )

        gr.HTML("""
        <div style="border-top:1px solid #0E1628;margin-top:32px;padding-top:16px;
            text-align:center;font-family:'DM Mono',monospace;font-size:0.72rem;
            color:#1E2D4A;letter-spacing:0.1em;">
          PATCHCORE · RESNET-50 · MVTEC-AD · GRADIO
        </div>""")

        # ── Event binding ─────────────────────────────────────────
        detect_btn.click(
            fn=run_inference,
            inputs=[input_img, category_dd],
            outputs=[heatmap_out, score_bar_out, verdict_html, metrics_box, viz_panel],
        )
        detect_btn.click(
            fn=reset_spec_report,
            inputs=[],
            outputs=[spec_report_out],
        )

        spec_btn.click(
            fn=run_spec_analysis,
            inputs=[input_img, category_dd],
            outputs=[spec_report_out],
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[download_btn],
        )

        download_btn.click(
            fn=generate_pdf_report,
            inputs=[input_img, category_dd],
            outputs=[report_file_out],
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[report_file_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_port=CFG["demo"]["port"],
        share=CFG["demo"]["share"],
        inbrowser=True,
    )