
# app.py — Streamlit AI Text-to-Image Web Application (FIXED)

import os
import io
import time
import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Text-to-Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished UI
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# HEADER SECTION
# ------------------------------------------------------------------------------
st.title("🎨 AI Text-to-Image Generator")
st.caption("Generate high-quality artwork from text prompts using Hugging Face AI Models.")

# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Generation Settings")

# Model Selection - Using models fully supported on HF Serverless API
model_choice = st.sidebar.selectbox(
    "Select AI Model",
    options=[
        "black-forest-labs/FLUX.1-schnell",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "prompthero/openjourney-v4"
    ],
    index=0,
    help="FLUX.1-schnell provides fast, ultra-realistic generation. SD v1.5 is the classic stable diffusion model."
)

# API Token Handling (Supports Streamlit Secrets & Environment Variables)
default_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
hf_token = st.sidebar.text_input(
    "Hugging Face Token",
    value=default_token,
    type="password",
    help="Enter your free Hugging Face API token (hf_...) from huggingface.co/settings/tokens"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Parameters")

steps = st.sidebar.slider("Inference Steps", min_value=1, max_value=50, value=25)
guidance_scale = st.sidebar.slider("Guidance Scale (CFG)", min_value=1.0, max_value=20.0, value=7.5, step=0.5)

width = st.sidebar.selectbox("Width", options=[512, 768, 1024], index=0)
height = st.sidebar.selectbox("Height", options=[512, 768, 1024], index=0)

# ------------------------------------------------------------------------------
# MAIN APP LAYOUT
# ------------------------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Text Prompt")
    
    prompt = st.text_area(
        "Enter your prompt:",
        value="A futuristic city at night with flying cars, neon lights, cinematic lighting, ultra realistic",
        height=140,
        help="Describe what you want to see in the image."
    )

    negative_prompt = st.text_input(
        "Negative Prompt (Optional):",
        value="blurry, low quality, distorted, deformed",
        help="Specify elements you want the AI to avoid."
    )

    generate_button = st.button("🚀 Generate Image", type="primary", use_container_width=True)

with col2:
    st.subheader("🖼️ Generated Output")

    if generate_button:
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt before clicking generate.")
        else:
            with st.spinner("✨ Generating your image... Please wait."):
                try:
                    start_time = time.time()

                    token_to_use = hf_token.strip() if hf_token and hf_token.strip() else None

                    # Initialize HF Inference Client
                    client = InferenceClient(
                        token=token_to_use
                    )

                    # Build API parameters dynamically
                    api_params = {
                        "prompt": prompt,
                        "model": model_choice,
                        "height": height,
                        "width": width,
                    }

                    if negative_prompt.strip():
                        api_params["negative_prompt"] = negative_prompt.strip()

                    # Pass steps and guidance_scale where supported
                    if "FLUX" not in model_choice:
                        api_params["num_inference_steps"] = steps
                        api_params["guidance_scale"] = guidance_scale

                    # Call Serverless Text-to-Image API
                    image = client.text_to_image(**api_params)

                    elapsed = time.time() - start_time

                    # Display Generated Image (Fixed parameter here)
                    st.image(
                        image,
                        caption=f"Generated in {elapsed:.2f}s | Model: {model_choice.split('/')[-1]}",
                        use_container_width=True
                    )

                    # Prepare Image for Download
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="📥 Download High-Res PNG",
                        data=byte_im,
                        file_name="generated_image.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    st.success("✅ Image generated successfully!")

                except Exception as e:
                    st.error("❌ Generation Failed")
                    error_msg = str(e) if str(e) else repr(e)

                    st.error(f"**Error Details:** {error_msg}")

                    if "401" in error_msg or "token" in error_msg.lower() or "authorization" in error_msg.lower():
                        st.info("💡 **Authentication Required:** Please enter a valid free Hugging Face API Token (starting with `hf_...`) in the sidebar or Streamlit Secrets.")
                    elif "503" in error_msg or "loading" in error_msg.lower():
                        st.info("💡 **Model Loading:** The Hugging Face server is warming up this model. Wait 15–20 seconds and click Generate again.")
