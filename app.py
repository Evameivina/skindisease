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
IMG_SIZE = 224
MODEL_PATH = "convnext_skin_state_dict.pth"

# Google Drive File ID
FILE_ID = "1s2BhRSSuUTRpuANjXkzTYSEAi_wKTuLR"

CLASSES = [
    "Eczema",
    "Herpes Zoster",
    "Normal",
    "Ringworm"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# DOWNLOAD MODEL
# =========================================================
@st.cache_resource
def download_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Mengunduh model..."):

            url = f"https://drive.google.com/uc?id={FILE_ID}"

            gdown.download(
                url,
                MODEL_PATH,
                quiet=False
            )

download_model()

# =========================================================
# DISEASE INFO
# =========================================================
DISEASE_INFO = {

    "Eczema": {
        "icon": "💧",
        "color": "#4A90D9",
        "description": "Eczema adalah kondisi kulit yang menyebabkan peradangan, gatal, dan kemerahan."
    },

    "Herpes Zoster": {
        "icon": "⚡",
        "color": "#E74C3C",
        "description": "Herpes Zoster merupakan infeksi virus yang menyebabkan ruam dan nyeri."
    },

    "Normal": {
        "icon": "✅",
        "color": "#27AE60",
        "description": "Kulit berada dalam kondisi normal dan sehat."
    },

    "Ringworm": {
        "icon": "🔄",
        "color": "#F39C12",
        "description": "Ringworm adalah infeksi jamur pada kulit yang berbentuk melingkar."
    }

}

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp{
    background:#0D1117;
    color:white;
}

/* SIDEBAR */
section[data-testid="stSidebar"]{
    background:#161B22;
    border-right:1px solid #30363D;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* TITLE */
.main-title{
    font-size:42px;
    font-weight:700;
    color:#58A6FF;
    margin-bottom:0;
}

.sub-title{
    color:#8B949E;
    margin-top:0;
    margin-bottom:30px;
}

/* CARD */
.custom-card{
    background:#161B22;
    padding:30px;
    border-radius:20px;
    border:1px solid #30363D;
    margin-bottom:20px;
}

/* RESULT */
.result-title{
    font-size:36px;
    font-weight:700;
    margin-bottom:10px;
}

.result-score{
    font-size:50px;
    font-weight:700;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"]{
    background:#161B22;
    border:2px dashed #30363D;
    border-radius:20px;
    padding:15px;
}

/* BUTTON */
.stButton > button{
    background:#238636;
    color:white;
    border:none;
    border-radius:12px;
    padding:12px;
    font-weight:600;
    width:100%;
}

.stButton > button:hover{
    background:#2EA043;
}

/* HIDE STREAMLIT */
#MainMenu, footer, header{
    visibility:hidden;
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

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model.to(device)

# =========================================================
# TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

    image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    pred_idx = np.argmax(probs)

    return CLASSES[pred_idx], probs

# =========================================================
# SIDEBAR
# =========================================================
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

# =========================================================
# MENU DETEKSI
# =========================================================
if menu == "🩺 Deteksi Penyakit Kulit":

    st.markdown(
        '<div class="main-title">Deteksi Penyakit Kulit</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Upload gambar kulit untuk mendapatkan hasil klasifikasi AI</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1,1])

    # =====================================================
    # LEFT
    # =====================================================
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

            st.markdown("""
            <div class="custom-card" style="text-align:center;">

            <div style="font-size:70px;">
            🖼️
            </div>

            <h3>Upload Gambar Kulit</h3>

            <p style="color:#8B949E;">
            JPG, JPEG, PNG
            </p>

            </div>
            """, unsafe_allow_html=True)

    # =====================================================
    # RIGHT
    # =====================================================
    with col2:

        st.markdown("## 📊 Hasil Analisis")

        if uploaded and analyze:

            try:

                model = load_model()

                pred_class, probs = predict(image, model)

                info = DISEASE_INFO[pred_class]

                confidence = probs.max() * 100

                color = info["color"]

                st.markdown(f"""
                <div class="custom-card">

                    <div style="
                        font-size:14px;
                        color:#8B949E;
                        margin-bottom:10px;
                    ">
                        HASIL KLASIFIKASI
                    </div>

                    <div class="result-title"
                        style="color:{color};">
                        {info['icon']} {pred_class}
                    </div>

                    <div style="
                        margin-top:20px;
                        font-size:14px;
                        color:#8B949E;
                    ">
                        CONFIDENCE SCORE
                    </div>

                    <div class="result-score"
                        style="color:{color};">
                        {confidence:.2f}%
                    </div>

                </div>
                """, unsafe_allow_html=True)

                st.warning(
                    "⚠️ Hasil ini hanya untuk screening awal dan bukan diagnosis medis."
                )

            except Exception as e:

                st.error(f"Terjadi kesalahan: {e}")

        else:

            st.markdown("""
            <div class="custom-card"
                style="
                    text-align:center;
                    padding:60px;
                ">

                <div style="font-size:70px;">
                🔬
                </div>

                <h3>
                Belum Ada Analisis
                </h3>

                <p style="color:#8B949E;">
                Upload gambar lalu klik tombol analisis
                </p>

            </div>
            """, unsafe_allow_html=True)

# =========================================================
# MENU INFORMASI
# =========================================================
elif menu == "📚 Informasi Penyakit":

    st.markdown(
        '<div class="main-title">Informasi Penyakit</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Pilih penyakit untuk melihat informasi</div>',
        unsafe_allow_html=True
    )

    selected = st.selectbox(
        "Pilih Penyakit",
        CLASSES
    )

    info = DISEASE_INFO[selected]

    st.markdown(f"""
    <div class="custom-card">

        <div class="result-title"
            style="color:{info['color']};">

            {info['icon']} {selected}

        </div>

        <p style="
            color:#C9D1D9;
            font-size:17px;
            line-height:1.8;
        ">
            {info['description']}
        </p>

    </div>
    """, unsafe_allow_html=True)
