import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io


st.set_page_config("Bodha AI by Skandan", layout="wide")
st.title("Bodha AI")

client = InferenceClient(
    model="black-forest-labs/FLUX.1-schnell",
    token=st.secrets["HF_TOKEN"]
)
col1, col2 = st.columns(2)

st.image("bodhaai.jpeg", width=150, caption="Bodha AI")

with col1:
    prompt = st.text_input("Enter the Description")
    
if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("Prompt cannot be empty")
    else:
        with st.spinner("Generating image..."):
            image = client.text_to_image(prompt)

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
