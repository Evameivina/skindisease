import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gdown
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="DermaScan",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CONFIG
# =========================================================
CLASSES = ["Eczema", "Herpes Zoster", "Normal", "Ringworm"]

MODEL_URL = "https://drive.google.com/uc?id=1s2BhRSSuUTRpuANjXkzTYSEAi_wKTuLR"
MODEL_PATH = "convnext_skin_state_dict.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# DOWNLOAD MODEL
# =========================================================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model dari Google Drive..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>

body {
    background-color: #f5f7fa;
}

.main-title{
    font-size: 2.2rem;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 10px;
}

.sub-text{
    color: #555;
    margin-bottom: 25px;
}

.result-box{
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e0e0e0;
    margin-top: 10px;
}

.result-title{
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 10px;
}

.confidence{
    font-size: 22px;
    font-weight: bold;
    color: #1f77ff;
}

.sidebar-title{
    font-size: 24px;
    font-weight: bold;
    color: #1f77ff;
    margin-bottom: 5px;
}

.sidebar-sub{
    color: #666;
    font-size: 14px;
    margin-bottom: 20px;
}

.info-box{
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e0e0e0;
    margin-bottom: 20px;
}

.stButton > button{
    width: 100%;
    background-color: #1f77ff;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: bold;
}

.stButton > button:hover{
    background-color: #005fe0;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():

    model = models.convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features

    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(CLASSES))
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict)

    model.eval()

    return model.to(device)

# =========================================================
# TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================================================
# PREDICT
# =========================================================
def predict(image, model):

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        probs = probs.cpu().numpy()[0]

    pred_idx = np.argmax(probs)

    return CLASSES[pred_idx], probs[pred_idx]

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🔬 DermaScan</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-sub">Skin Disease Detection</div>',
        unsafe_allow_html=True
    )

    menu = st.radio(
        "MENU",
        ["🩺 Deteksi Penyakit Kulit", "📚 Informasi Penyakit"]
    )

# =========================================================
# MENU DETEKSI
# =========================================================
if menu == "🩺 Deteksi Penyakit Kulit":

    st.markdown(
        '<div class="main-title">Deteksi Penyakit Kulit</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-text">Upload gambar kulit untuk mendapatkan hasil prediksi AI</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(
                image,
                caption="Gambar yang diupload",
                use_container_width=True
            )

        with col2:

            if st.button("Deteksi Sekarang"):

                try:
                    model = load_model()

                    prediction, confidence = predict(image, model)

                    color_map = {
                        "Eczema": "#3498db",
                        "Herpes Zoster": "#e74c3c",
                        "Normal": "#27ae60",
                        "Ringworm": "#f39c12"
                    }

                    color = color_map[prediction]

                    st.markdown(f"""
                    <div class="result-box">

                        <div style="font-size:14px; color:gray;">
                        Hasil Klasifikasi
                        </div>

                        <div class="result-title" style="color:{color};">
                        {prediction}
                        </div>

                        <div style="margin-top:20px; font-size:14px; color:gray;">
                        Confidence Score
                        </div>

                        <div class="confidence">
                        {confidence*100:.2f}%
                        </div>

                    </div>
                    """, unsafe_allow_html=True)

                    st.warning(
                        "Hasil ini hanya untuk screening awal dan bukan diagnosis medis."
                    )

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

# =========================================================
# MENU INFORMASI
# =========================================================
elif menu == "📚 Informasi Penyakit":

    st.markdown(
        '<div class="main-title">Informasi Penyakit Kulit</div>',
        unsafe_allow_html=True
    )

    pilihan = st.selectbox(
        "Pilih penyakit",
        CLASSES
    )

    info = {
        "Eczema": """
        Eczema adalah kondisi kulit yang menyebabkan kulit merah,
        gatal, kering, dan iritasi.
        """,

        "Herpes Zoster": """
        Herpes Zoster adalah infeksi virus yang menyebabkan ruam
        dan rasa nyeri pada kulit.
        """,

        "Normal": """
        Kulit berada dalam kondisi normal dan tidak ditemukan
        indikasi penyakit kulit.
        """,

        "Ringworm": """
        Ringworm adalah infeksi jamur pada kulit yang biasanya
        berbentuk lingkaran dan terasa gatal.
        """
    }

    st.markdown(f"""
    <div class="info-box">

        <h3>{pilihan}</h3>

        <p style="line-height:1.8;">
        {info[pilihan]}
        </p>

    </div>
    """, unsafe_allow_html=True)
