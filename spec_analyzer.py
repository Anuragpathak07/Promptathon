"""
spec_analyzer.py
----------------
Calls the AI (Groq / OpenAI-compatible) to produce an inspection report
comparing the uploaded component image against its datasheet specification.

IMPORTANT — The system prompt explicitly forbids markdown syntax so that
the HTML renderer in app.py receives clean plain text with no ## / ** / ` artifacts.
"""

import base64
import io
import logging
import os
import re
from typing import Tuple
from dotenv import load_dotenv
from PIL import Image
# Load variables from .env
load_dotenv()
log = logging.getLogger(__name__)

# ── Client setup ─────────────────────────────────────────────────────────────
# Supports Groq (default) or any OpenAI-compatible endpoint.
# Set GROQ_API_KEY (or OPENAI_API_KEY) in your environment.

def _get_client():
    """Lazy-import and return an OpenAI-compatible client."""
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            return Groq(api_key=api_key), "groq"
    except ImportError:
        pass

    try:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            return OpenAI(api_key=api_key), "openai"
    except ImportError:
        pass

    raise RuntimeError(
        "No AI client available. Install 'groq' or 'openai' and set the API key."
    )


def _pil_to_b64(image: Image.Image, max_dim: int = 1024) -> str:
    """Resize and base64-encode a PIL image as JPEG."""
    img = image.copy()
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── System prompt — NO markdown ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert industrial quality-control engineer with deep knowledge of
electronic components, PCB assemblies, and manufacturer datasheets.

Your task is to inspect a component image and produce a structured inspection report.

STRICT FORMATTING RULES — follow these exactly or the report will be unreadable:
1. Do NOT use any markdown syntax whatsoever.
   - No hash characters (#) for headings.
   - No asterisks (*) for bold or italic.
   - No backticks (`) for code.
   - No underscores (_) for formatting.
   - No triple-dashes (---) for horizontal rules.
2. Structure your report using ONLY plain-text numbered sections, like:
     1. CURRENT STATUS:
     2. SPEC COMPLIANCE:
     3. RISK LEVEL:
     4. RECOMMENDATIONS:
3. Use plain dashes for bullet points, like:
     - Item one description here
     - Item two description here
4. For emphasis, write in ALL CAPS instead of using asterisks.
5. Numbers, measurements, and specs should be written plainly, e.g.: 45mm x 20mm x 15mm.

The report should be factual, concise, and based on what is visually observable in the image
combined with the known datasheet specifications for the identified component category."""


def analyze_with_spec(
    image: Image.Image,
    category: str,
    anomaly_score: float,
    verdict: str,
) -> str:
    """
    Send the component image + PatchCore verdict to the AI and return a
    plain-text inspection report (no markdown syntax).

    Parameters
    ----------
    image         : PIL image of the component under inspection
    category      : MVTec-AD category string (e.g. 'transistor', 'bottle')
    anomaly_score : Float score from PatchCore
    verdict       : 'NORMAL' or 'ANOMALY (DEFECTIVE)'

    Returns
    -------
    Plain-text report string.
    """
    client, backend = _get_client()
    b64_image = _pil_to_b64(image)

    user_prompt = (
        f"Component category: {category}\n"
        f"PatchCore anomaly score: {anomaly_score:.6f}\n"
        f"PatchCore verdict: {verdict}\n\n"
        "Please inspect the attached component image and produce a full inspection report "
        "following the formatting rules in your system instructions. "
        "Include:\n"
        "1. CURRENT STATUS — describe what you see in the image (markings, physical condition, "
        "visible features, any defects).\n"
        "2. SPEC COMPLIANCE — list key datasheet specifications and whether the component "
        "appears to comply with each one based on visual inspection.\n"
        "3. RISK LEVEL — state LOW, MEDIUM, HIGH, or CRITICAL with a one-sentence justification.\n"
        "4. RECOMMENDATIONS — list any actions the engineer should take.\n\n"
        "Remember: plain text only, no markdown symbols."
    )

    try:
        if backend == "groq":
            # Groq vision model
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                max_tokens=1024,
                temperature=0.2,
            )
        else:
            # OpenAI GPT-4o
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                max_tokens=1024,
                temperature=0.2,
            )

        report = response.choices[0].message.content.strip()

        # ── Safety net: strip any residual markdown that the model ignored ──
        report = _strip_residual_markdown(report)
        return report

    except Exception as exc:
        log.error(f"AI spec analysis failed: {exc}")
        return (
            f"1. CURRENT STATUS:\n"
            f"- AI analysis unavailable: {exc}\n\n"
            f"2. SPEC COMPLIANCE:\n"
            f"- Unable to perform automated datasheet comparison.\n\n"
            f"3. RISK LEVEL:\n"
            f"UNKNOWN — manual inspection required.\n\n"
            f"4. RECOMMENDATIONS:\n"
            f"- Check API key and network connectivity.\n"
            f"- Perform manual visual inspection against the component datasheet."
        )


def _strip_residual_markdown(text: str) -> str:
    """
    Post-processing safety net.
    Removes any markdown syntax the model included despite instructions.
    Converts common patterns to plain-text equivalents.
    """
    lines = text.splitlines()
    clean = []
    for line in lines:
        # ## Heading → plain uppercase text
        line = re.sub(r"^#{1,6}\s+", "", line)
        # **bold** or __bold__ → UPPERCASE the content
        line = re.sub(r"\*\*(.+?)\*\*", lambda m: m.group(1).upper(), line)
        line = re.sub(r"__(.+?)__",     lambda m: m.group(1).upper(), line)
        # *italic* or _italic_ → plain
        line = re.sub(r"\*(.+?)\*", r"\1", line)
        line = re.sub(r"_(.+?)_",   r"\1", line)
        # `code` → plain
        line = re.sub(r"`(.+?)`", r"\1", line)
        # Horizontal rules
        line = re.sub(r"^-{3,}$", "", line)
        line = re.sub(r"^\*{3,}$", "", line)
        clean.append(line)
    return "\n".join(clean)


def get_risk_level(report: str) -> Tuple[str, str]:
    """
    Parse the RISK LEVEL section from the plain-text report.
    Returns (risk_label, justification).
    """
    # Look for "3. RISK LEVEL:" section or "RISK LEVEL:" anywhere
    section_re = re.compile(
        r"(?:3\.\s*)?RISK\s*LEVEL\s*[:\-]?\s*\n?(.*?)(?:\n\n|\n\d+\.|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    m = section_re.search(report)
    if not m:
        return "LOW", ""

    content = m.group(1).strip()
    first_line = content.splitlines()[0].strip() if content else ""

    # Extract the risk word
    for level in ("CRITICAL", "HIGH", "MEDIUM", "MODERATE", "LOW"):
        if level in first_line.upper():
            rest = re.sub(level, "", first_line, flags=re.IGNORECASE).strip(" -:—")
            return level, rest

    # Fall back: return first non-empty word
    word = re.sub(r"[^A-Za-z]", "", first_line.split()[0]) if first_line.split() else "LOW"
    return word.upper() or "LOW", first_line