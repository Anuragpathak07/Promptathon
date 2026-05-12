# spec_analyzer.py

import os
import base64
import httpx
import io
import pypdf
from PIL import Image
from groq import Groq

# ---- Direct PDF URLs for all 4 PCBs ----
DATASHEET_URLS = {
    "pcb1": "https://cdn.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf",
    "pcb2": "https://www.handsontec.com/dataspecs/HC-SR04-Ultrasonic.pdf",
    "pcb3": "https://www.vishay.com/docs/83760/tcrt5000.pdf",
    "pcb4": "https://dlnmh9ip6v2uc.cloudfront.net/datasheets/Prototyping/TP4056.pdf"
}

def extract_pdf_text_from_url(url):
    """Download PDF from URL and extract text using pypdf"""
    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"--- Page {i+1} ---\n{page_text}\n"
        return text
    except Exception as e:
        return f"Error downloading or extracting PDF datasheet from {url}: {str(e)}"

def encode_image(image):
    """Convert PIL image to base64 for Groq API"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_with_spec(image, category, anomaly_score, verdict):
    """
    Sends PCB image + extracted datasheet text to Groq
    Groq reads the actual spec and compares against the image
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "Error: GROQ_API_KEY environment variable is not set. "
            "Please configure your Groq API key in your environment to use this feature."
        )

    # Fetch the actual datasheet URL
    pdf_url = DATASHEET_URLS.get(category)
    if not pdf_url:
        return f"No official datasheet mapping found for category: '{category}'. Only pcb1, pcb2, pcb3, and pcb4 are supported."

    # Extract the text content from the PDF datasheet
    pdf_text = extract_pdf_text_from_url(pdf_url)
    if "Error downloading" in pdf_text:
        return pdf_text

    # Encode the PCB image
    try:
        image_data = encode_image(image)
    except Exception as e:
        return f"Error encoding PCB image: {str(e)}"

    # Initialize Groq client
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"Error initializing Groq client: {str(e)}"

    # Prompt
    prompt = f"""
    You are an expert PCB quality control engineer.

    I am giving you two things:
    1. The official manufacturing datasheet text for this component
    2. An actual photo of this component from our production line

    The automated vision system reported:
    - Anomaly Score: {anomaly_score:.4f}
    - Verdict: {verdict}
    - Category: {category}

    Please read the datasheet carefully and then inspect the image.
    Give me:

    1. CURRENT STATUS:
       What do you physically see in the image?
       Any visible issues even minor ones?

    2. SPEC COMPLIANCE:
       Based on the datasheet you just read,
       does this component meet its specifications?
       Call out any specific measurements, tolerances,
       or visual requirements from the datasheet
       that this component may be violating or borderline on.

    3. RISK LEVEL: Pick one — LOW / MEDIUM / HIGH
       LOW = Perfectly fine, ship it
       MEDIUM = Looks okay now but has warning signs
       HIGH = Will likely fail, do not ship

    4. PREDICTED FAILURE:
       If shipped today, what will fail, why, and roughly when?
       Base this on the datasheet specs.

    5. RECOMMENDATION:
       What should the QC engineer do right now?

    Be specific. Reference actual values from the datasheet.
    Do not be vague.
    """

    # We append the PDF text directly to the system/user text message context
    full_text = f"""
Official manufacturing specification document for {category} component:
========================================================================
{pdf_text}
========================================================================

{prompt}
"""

    try:
        # Send text and image to Groq's multimodal Llama 4 Vision model
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=1500,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error executing Groq API call: {str(e)}"


def get_risk_level(analysis_text):
    """Extract risk level from Groq response"""
    text_upper = analysis_text.upper()
    if "HIGH" in text_upper:
        return "HIGH", "#ff4444"
    elif "MEDIUM" in text_upper:
        return "MEDIUM", "#ffaa00"
    else:
        return "LOW", "#44ff44"
