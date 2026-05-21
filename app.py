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
    page_title="Skin Disease Detection",
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
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp {
    background-color: #f5f7fa;
}

.main-title{
    font-size: 38px;
    font-weight: bold;
    color: #1f2937;
    margin-bottom: 5px;
}

.sub-text{
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 25px;
}

.sidebar-title{
    font-size: 24px;
    font-weight: bold;
    color: #2563eb;
    margin-bottom: 5px;
}

.sidebar-sub{
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 20px;
}

[data-testid="stSidebar"]{
    background-color: white;
}

.result-card{
    background-color: white;
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-top: 20px;
}

.info-box{
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    line-height: 1.8;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.disclaimer{
    margin-top: 20px;
    background-color: #fff4e5;
    border: 1px solid #fcd34d;
    color: #b45309;
    padding: 15px;
    border-radius: 12px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
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
        '<div class="sidebar-title">🔬 Skin Disease Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-sub">AI Based Skin Disease Classification</div>',
        unsafe_allow_html=True
    )

    menu = st.radio(
        "MENU",
        [
            "🩺 Deteksi Penyakit Kulit",
            "📚 Informasi Penyakit"
        ]
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

            try:
                model = load_model()

                prediction, confidence = predict(image, model)

                emoji_map = {
                    "Eczema": "🔵",
                    "Herpes Zoster": "🔴",
                    "Normal": "🟢",
                    "Ringworm": "🟠"
                }

                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                st.write("### Hasil Klasifikasi")

                st.markdown(
                    f"## {emoji_map[prediction]} {prediction}"
                )

                st.metric(
                    label="Confidence Score",
                    value=f"{confidence*100:.2f}%"
                )

                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="disclaimer">
                ⚠️ Hasil ini hanya untuk screening awal dan bukan diagnosis medis.
                </div>
                """, unsafe_allow_html=True)

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
        "Pilih Penyakit",
        CLASSES
    )

    info = {

        "Eczema": """
        Eczema adalah kondisi kulit yang menyebabkan kulit merah,
        gatal, kering, dan iritasi. Penyakit ini dapat dipicu oleh
        alergi, iritasi, maupun faktor lingkungan.
        """,

        "Herpes Zoster": """
        Herpes Zoster adalah infeksi virus yang menyebabkan ruam
        dan rasa nyeri pada kulit. Penyakit ini muncul akibat
        reaktivasi virus cacar air.
        """,

        "Normal": """
        Kulit berada dalam kondisi normal dan sehat tanpa adanya
        indikasi penyakit kulit pada gambar yang diunggah.
        """,

        "Ringworm": """
        Ringworm adalah infeksi jamur pada kulit yang biasanya
        berbentuk lingkaran dan terasa gatal.
        """
    }

    st.markdown(
        f"""
        <div class="info-box">

        <h2>{pilihan}</h2>

        <p>
        {info[pilihan]}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )
