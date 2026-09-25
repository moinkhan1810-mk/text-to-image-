# app.py — Streamlit AI Text-to-Image Web Application

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

# Model Selection
model_choice = st.sidebar.selectbox(
    "Select AI Model",
    options=[
        "SimianLuo/LCM_Dreamshaper_v7",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "prompthero/openjourney-v4"
    ],
    index=0,
    help="LCM Dreamshaper generates images in ~1-2 seconds (4 steps). SD v1.5 provides traditional diffusion detail."
)

# API Token Handling (Supports Streamlit Secrets & Environment Variables)
default_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
hf_token = st.sidebar.text_input(
    "Hugging Face Token (Optional)",
    value=default_token,
    type="password",
    help="Optional for public models. Enter your token if you hit rate limits."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Parameters")

# Dynamic step defaults based on model selection
default_steps = 4 if "LCM" in model_choice else 25
steps = st.sidebar.slider("Inference Steps", min_value=1, max_value=50, value=default_steps)
guidance_scale = st.sidebar.slider("Guidance Scale (CFG)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)

width = st.sidebar.selectbox("Width", options=[512, 768], index=0)
height = st.sidebar.selectbox("Height", options=[512, 768], index=0)

use_random_seed = st.sidebar.checkbox("Randomize Seed", value=True)
if not use_random_seed:
    seed = st.sidebar.number_input("Seed Value", min_value=0, max_value=2147483647, value=42)
else:
    seed = None

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

                    # Initialize HF Inference Client
                    client = InferenceClient(
                        token=hf_token if hf_token.strip() else None
                    )

                    # Execute Serverless API Request
                    image = client.text_to_image(
                        prompt=prompt,
                        negative_prompt=negative_prompt if negative_prompt.strip() else None,
                        model=model_choice,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        seed=seed
                    )

                    elapsed = time.time() - start_time

                    # Display Generated Image
                    st.image(
                        image,
                        caption=f"Generated in {elapsed:.2f}s | Model: {model_choice.split('/')[-1]}",
                        use_column_width=True
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
                    error_str = str(e)

                    if "503" in error_str or "loading" in error_str.lower():
                        st.info("💡 **Model Booting:** The model is warming up on Hugging Face servers. Please wait 15–20 seconds and click Generate again.")
                    elif "401" in error_str or "token" in error_str.lower():
                        st.info("💡 **Token Error:** Please enter a valid free Hugging Face API token in the sidebar.")
                    else:
                        st.error(f"Error details: {error_str}")
