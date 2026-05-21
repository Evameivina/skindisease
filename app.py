import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
IMG_SIZE = 224
MODEL_PATH = "convnext_skin_state_dict.pth"

CLASSES = [
    "Eczema",
    "Herpes Zoster",
    "Normal",
    "Ringworm"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# INFORMASI PENYAKIT
# ─────────────────────────────────────────────────────────────
DISEASE_INFO = {

    "Eczema": {
        "icon": "🔴",
        "color": "#E05C5C",
        "bg": "#FFF0F0",

        "deskripsi":
        "Eczema atau dermatitis atopik adalah kondisi kulit "
        "yang menyebabkan kulit kering, gatal, dan meradang.",

        "gejala": [
            "Kulit kering",
            "Gatal berlebihan",
            "Kemerahan",
            "Kulit mengelupas"
        ],

        "penanganan":
        "Gunakan pelembap secara rutin dan hindari pemicu alergi."
    },

    "Herpes Zoster": {
        "icon": "🟠",
        "color": "#E07A2F",
        "bg": "#FFF5EC",

        "deskripsi":
        "Herpes Zoster disebabkan oleh reaktivasi virus "
        "varicella-zoster yang menimbulkan ruam nyeri.",

        "gejala": [
            "Ruam merah",
            "Nyeri pada kulit",
            "Lepuhan berisi cairan",
            "Sensasi panas"
        ],

        "penanganan":
        "Konsultasikan ke dokter untuk terapi antivirus."
    },

    "Normal": {
        "icon": "🟢",
        "color": "#2E9E6B",
        "bg": "#EDFAF3",

        "deskripsi":
        "Kulit berada dalam kondisi normal "
        "tanpa indikasi penyakit kulit.",

        "gejala": [
            "Tidak ada tanda penyakit kulit"
        ],

        "penanganan":
        "Jaga kebersihan dan kesehatan kulit."
    },

    "Ringworm": {
        "icon": "🟣",
        "color": "#7B61D4",
        "bg": "#F4F1FF",

        "deskripsi":
        "Ringworm adalah infeksi jamur pada kulit "
        "yang membentuk pola melingkar.",

        "gejala": [
            "Ruam melingkar",
            "Kulit bersisik",
            "Gatal",
            "Kemerahan"
        ],

        "penanganan":
        "Gunakan krim antijamur dan jaga kebersihan kulit."
    }
}

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinScan",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* HIDE STREAMLIT */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* SIDEBAR */

[data-testid="collapsedControl"] {
    display: none;
}

section[data-testid="stSidebar"] {
    background: #0F1117;
    border-right: 1px solid #1E2130;
    min-width: 300px !important;
    max-width: 300px !important;
}

/* SIDEBAR TEXT */

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* BRAND */

.sidebar-brand {
    text-align: center;
    padding-top: 1rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #1E2130;
    margin-bottom: 2rem;
}

.brand-icon {
    font-size: 3rem;
}

.brand-title {
    font-size: 2rem;
    font-family: 'DM Serif Display', serif;
}

.brand-subtitle {
    color: #9CA3AF !important;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* MENU */

div[role="radiogroup"] > label {
    background: #1E2130;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 10px;
    border: 1px solid transparent;
    transition: 0.2s;
}

div[role="radiogroup"] > label:hover {
    border: 1px solid #6366F1;
    background: #272C3F;
}

div[role="radiogroup"] label p {
    font-size: 0.95rem !important;
    font-weight: 500;
}

/* PAGE TITLE */

.page-title h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.page-title p {
    color: #6B7280;
    margin-bottom: 2rem;
}

/* UPLOAD */

.upload-box {
    border: 2px dashed #D1D5DB;
    border-radius: 20px;
    padding: 4rem 2rem;
    text-align: center;
    background: #FAFAFA;
}

.upload-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

/* RESULT */

.result-card {
    border-radius: 18px;
    padding: 1.5rem;
    border: 1px solid;
    margin-bottom: 1rem;
}

.result-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    color: #6B7280;
}

.result-name {
    font-size: 2rem;
    font-family: 'DM Serif Display', serif;
}

.result-confidence {
    margin-top: 0.5rem;
    font-size: 1rem;
}

/* PROBABILITY */

.prob-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.3rem;
}

.prob-bg {
    height: 8px;
    background: #E5E7EB;
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 0.8rem;
}

.prob-fill {
    height: 8px;
    border-radius: 999px;
}

/* INFO CARD */

.info-card {
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid;
    margin-top: 1rem;
}

.info-title {
    font-size: 1.8rem;
    font-family: 'DM Serif Display', serif;
    margin-bottom: 1rem;
}

.section-title {
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    text-transform: uppercase;
    color: #6B7280;
    font-weight: 700;
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

    state = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(state)

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
# PREDICT
# ─────────────────────────────────────────────────────────────
def predict(image):

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = load_model()(tensor)

        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    return CLASSES[probs.argmax()], probs

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">

        <div class="brand-icon">
            🔬
        </div>

        <div class="brand-title">
            SkinScan
        </div>

        <div class="brand-subtitle">
            Skin Disease Detection
        </div>

    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "",
        [
            "🩺 Deteksi Penyakit",
            "📖 Informasi Penyakit"
        ],
        label_visibility="collapsed"
    )

# ─────────────────────────────────────────────────────────────
# MENU DETEKSI
# ─────────────────────────────────────────────────────────────
if "Deteksi" in menu:

    st.markdown("""
    <div class="page-title">
        <h1>Deteksi Penyakit Kulit</h1>
        <p>
            Upload gambar kulit untuk mendapatkan
            hasil prediksi dari model AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded is None:

        st.markdown("""
        <div class="upload-box">

            <div class="upload-icon">
                🖼️
            </div>

            <h3>Seret & Lepas Gambar di Sini</h3>

            <p>
                atau klik Browse Files • JPG, JPEG, PNG
            </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(image, use_container_width=True)

        with col2:

            with st.spinner("Menganalisis gambar..."):

                label, probs = predict(image)

                info = DISEASE_INFO[label]

                confidence = float(probs.max()) * 100

                st.markdown(f"""
                <div class="result-card"
                     style="
                     background:{info['bg']};
                     border-color:{info['color']};
                     ">

                    <div class="result-title">
                        Hasil Prediksi
                    </div>

                    <div class="result-name"
                         style="color:{info['color']}">

                        {info['icon']} {label}

                    </div>

                    <div class="result-confidence">

                        Confidence:
                        <strong>{confidence:.2f}%</strong>

                    </div>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Probabilitas")

                for i, cls in enumerate(CLASSES):

                    percentage = float(probs[i]) * 100

                    color = DISEASE_INFO[cls]["color"]

                    st.markdown(f"""
                    <div class="prob-label">

                        <span>
                            {DISEASE_INFO[cls]['icon']} {cls}
                        </span>

                        <span>
                            {percentage:.1f}%
                        </span>

                    </div>

                    <div class="prob-bg">

                        <div class="prob-fill"
                             style="
                             width:{percentage}%;
                             background:{color};
                             ">

                        </div>

                    </div>
                    """, unsafe_allow_html=True)

                if label != "Normal":

                    st.warning(
                        "Hasil ini bukan diagnosis medis. "
                        "Silakan konsultasikan dengan dokter."
                    )

# ─────────────────────────────────────────────────────────────
# MENU INFORMASI
# ─────────────────────────────────────────────────────────────
elif "Informasi" in menu:

    st.markdown("""
    <div class="page-title">

        <h1>Informasi Penyakit Kulit</h1>

        <p>
            Pilih penyakit untuk melihat
            informasi lengkap.
        </p>

    </div>
    """, unsafe_allow_html=True)

    selected = st.selectbox(
        "Pilih Penyakit",
        CLASSES
    )

    info = DISEASE_INFO[selected]

    gejala_html = "".join([
        f"<li>{g}</li>"
        for g in info["gejala"]
    ])

    st.markdown(f"""
    <div class="info-card"
         style="
         background:{info['bg']};
         border-color:{info['color']}40;
         ">

        <div class="info-title"
             style="color:{info['color']}">

            {info['icon']} {selected}

        </div>

        <p>
            {info['deskripsi']}
        </p>

        <div class="section-title">
            Gejala Umum
        </div>

        <ul>
            {gejala_html}
        </ul>

        <div class="section-title">
            Penanganan
        </div>

        <p>
            {info['penanganan']}
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Informasi ini bersifat edukatif dan "
        "tidak menggantikan diagnosis medis profesional."
    )
