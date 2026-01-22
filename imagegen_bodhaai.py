import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
from diffusers import DiffusionPipeline
import torch

st.set_page_config("Bodha AI by Skandan", layout="wide")
st.title("Bodha AI")

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

col1, col2 = st.columns(2)

st.image("bodhaai.jpeg", width=150, caption="Bodha AI")

with col1:
    inputprompt = st.text_input("Enter the Description")

    if st.button("Generate Content"):
        with st.spinner("Generating image..."):
            image = pipe(inputprompt=inputprompt).image[0]
            st.session_state.image = image

with col2:
    if "image" in st.session_state:
        st.image(st.session_state.image, width=400)

        img_buffer = io.BytesIO()
        st.session_state.image.save(img_buffer, format="PNG")

        st.download_button(
            label="⬇️ Download Image",
            data=img_buffer.getvalue(),
            file_name="generated.png",
            mime="image/png"
        )
    else:
        st.info("Generate content first")
