import streamlit as st
from PIL import Image
import tempfile
from transformers import pipeline

@st.cache_resource
def load_detector():
    return pipeline("image-classification", model="Ateeqq/ai-vs-human-image-detector")

detector = load_detector()

st.set_page_config(page_title="FakeArt - AI vs Human Image Detector", layout="centered")

st.title("🎨 FakeArt")
st.subheader("Detect whether an image is AI-generated or captured by a human")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            image.save(tmp_file.name)
            result = detector(tmp_file.name)

    top_result = sorted(result, key=lambda x: x["score"], reverse=True)[0]
    label = top_result["label"]
    confidence = top_result["score"] * 100

    st.markdown("---")
    st.subheader("🧠 Result")

    if "ai" in label.lower():
        st.error(f"🖼️ The image is likely AI-generated.\n\nConfidence: {confidence:.2f}%")
    else:
        st.success(f"📷 The image is likely real (human-made).\n\nConfidence: {confidence:.2f}%")

    st.markdown("---")
    
