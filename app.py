import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gdown
import os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermaScan - Skin Disease Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
IMG_SIZE = 224
MODEL_PATH = "convnext_skin_state_dict.pth"

# Google Drive FILE ID
FILE_ID = "1s2BhRSSuUTRpuANjXkzTYSEAi_wKTuLR"

CLASSES = ["Eczema", "Herpes Zoster", "Normal", "Ringworm"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model dari Google Drive..."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

download_model()

# ─────────────────────────────────────────────────────────────
# DISEASE INFORMATION
# ─────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Eczema": {
        "icon": "💧",
        "color": "#4A90D9",
        "description": "Eczema adalah kondisi kulit yang menyebabkan peradangan, kemerahan, dan rasa gatal.",
        "gejala": [
            "Kulit gatal",
            "Kulit kering",
            "Ruam merah",
            "Kulit bersisik"
        ],
        "penanganan": "Gunakan pelembab dan hindari pemicu alergi."
    },

    "Herpes Zoster": {
        "icon": "⚡",
        "color": "#E74C3C",
        "description": "Herpes Zoster disebabkan oleh reaktivasi virus varicella-zoster.",
        "gejala": [
            "Nyeri atau sensasi terbakar",
            "Ruam merah",
            "Lepuhan cairan",
            "Kesemutan"
        ],
        "penanganan": "Segera konsultasi ke dokter untuk antivirus."
    },

    "Normal": {
        "icon": "✅",
        "color": "#27AE60",
        "description": "Kulit berada dalam kondisi normal dan sehat.",
        "gejala": [
            "Tidak ada iritasi",
            "Warna kulit normal",
            "Tidak ada ruam"
        ],
        "penanganan": "Pertahankan kebersihan dan kesehatan kulit."
    },

    "Ringworm": {
        "icon": "🔄",
        "color": "#F39C12",
        "description": "Ringworm adalah infeksi jamur pada kulit.",
        "gejala": [
            "Ruam berbentuk lingkaran",
            "Kulit bersisik",
            "Rasa gatal",
            "Tepi ruam lebih merah"
        ],
        "penanganan": "Gunakan obat antijamur dan jaga kebersihan kulit."
    }
}

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Arial', sans-serif;
}

.stApp {
    background-color: #0D1117;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main Title */
.main-title {
    font-size: 2.4rem;
    font-weight: bold;
    color: #58A6FF;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #8B949E;
    margin-bottom: 2rem;
}

/* Result Box */
.result-box {
    background: #161B22;
    border-radius: 15px;
    padding: 1.5rem;
    border: 1px solid #30363D;
}

/* Confidence Bar */
.bar-bg {
    width: 100%;
    height: 10px;
    background: #30363D;
    border-radius: 100px;
    overflow: hidden;
    margin-top: 5px;
}

.bar-fill {
    height: 10px;
    border-radius: 100px;
}

/* Card */
.info-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
}

/* Hide Streamlit */
#MainMenu, footer, header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
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

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model.to(device)

# ─────────────────────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ─────────────────────────────────────────────────────────────
# PREDICT FUNCTION
# ─────────────────────────────────────────────────────────────
def predict(image, model):

    image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    pred_idx = np.argmax(probs)

    return CLASSES[pred_idx], probs

# ─────────────────────────────────────────────────────────────
# SIDEBAR MENU (SEBELAH KIRI)
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("## 🔬 DermaScan")
    st.markdown("Skin Disease Detection")
    st.markdown("---")

    menu = st.radio(
        "MENU",
        [
            "🩺 Deteksi Penyakit Kulit",
            "📚 Informasi Penyakit"
        ]
    )

    st.markdown("---")

    st.markdown("""
    ### Kelas yang Didukung
    - Eczema
    - Herpes Zoster
    - Normal
    - Ringworm
    """)

# ─────────────────────────────────────────────────────────────
# PAGE : DETEKSI
# ─────────────────────────────────────────────────────────────
if menu == "🩺 Deteksi Penyakit Kulit":

    st.markdown(
        '<div class="main-title">Deteksi Penyakit Kulit</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Upload gambar kulit untuk mendapatkan hasil prediksi AI</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    # ───────────── LEFT ─────────────
    with col1:

        uploaded = st.file_uploader(
            "Upload gambar",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:

            image = Image.open(uploaded)

            st.image(
                image,
                caption="Gambar yang diupload",
                use_container_width=True
            )

            analyze = st.button(
                "🔍 Analisis Sekarang",
                use_container_width=True
            )

        else:
            analyze = False

    # ───────────── RIGHT ─────────────
    with col2:

        st.subheader("📊 Hasil Analisis")

        if uploaded and analyze:

            try:
                model = load_model()

                pred_class, probs = predict(image, model)

                info = DISEASE_INFO[pred_class]

                confidence = probs.max() * 100

                color = info["color"]

                # RESULT BOX
                st.markdown(f"""
                <div class="result-box">

                    <h2 style="color:{color};">
                        {info['icon']} {pred_class}
                    </h2>

                    <h4>Confidence Score</h4>

                    <div style="font-size:2rem; color:{color}; font-weight:bold;">
                        {confidence:.2f}%
                    </div>

                    <div class="bar-bg">
                        <div class="bar-fill"
                            style="
                                width:{confidence}%;
                                background:{color};
                            ">
                        </div>
                    </div>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Semua Probabilitas")

                # SEMUA SCORE + DEBAR
                for i, cls in enumerate(CLASSES):

                    pct = probs[i] * 100

                    c = DISEASE_INFO[cls]["color"]

                    st.markdown(f"""
                    <div style="margin-bottom:15px;">

                        <div style="
                            display:flex;
                            justify-content:space-between;
                            margin-bottom:4px;
                            font-size:14px;
                        ">
                            <span>{cls}</span>
                            <span>{pct:.2f}%</span>
                        </div>

                        <div class="bar-bg">
                            <div class="bar-fill"
                                style="
                                    width:{pct}%;
                                    background:{c};
                                ">
                            </div>
                        </div>

                    </div>
                    """, unsafe_allow_html=True)

                # INFO CARD
                st.markdown(f"""
                <div class="info-card">
                    <h3 style="color:{color};">
                        Informasi Penyakit
                    </h3>

                    <p>{info['description']}</p>

                    <h4>Gejala:</h4>
                    <ul>
                        {''.join([f"<li>{g}</li>" for g in info['gejala']])}
                    </ul>

                    <h4>Penanganan:</h4>
                    <p>{info['penanganan']}</p>

                </div>
                """, unsafe_allow_html=True)

                st.warning(
                    "⚠️ Hasil ini hanya untuk screening awal dan bukan diagnosis medis."
                )

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

# ─────────────────────────────────────────────────────────────
# PAGE : INFORMASI PENYAKIT
# ─────────────────────────────────────────────────────────────
elif menu == "📚 Informasi Penyakit":

    st.markdown(
        '<div class="main-title">Informasi Penyakit Kulit</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Pilih penyakit untuk melihat detail informasi</div>',
        unsafe_allow_html=True
    )

    selected = st.selectbox(
        "Pilih Penyakit",
        CLASSES
    )

    info = DISEASE_INFO[selected]

    color = info["color"]

    st.markdown(f"""
    <div class="info-card">

        <h2 style="color:{color};">
            {info['icon']} {selected}
        </h2>

        <p>{info['description']}</p>

        <h4>Gejala Umum</h4>

        <ul>
            {''.join([f"<li>{g}</li>" for g in info['gejala']])}
        </ul>

        <h4>Penanganan</h4>

        <p>{info['penanganan']}</p>

    </div>
    """, unsafe_allow_html=True)
