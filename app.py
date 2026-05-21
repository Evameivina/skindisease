import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 224
MODEL_PATH = "convnext_skin_state_dict.pth"

CLASSES = [
    "Eczema",
    "Herpes Zoster",
    "Normal",
    "Ringworm"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# INFORMASI PENYAKIT
# ──────────────────────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Eczema": {
        "icon": "🔴",
        "color": "#E05C5C",
        "bg": "#FFF0F0",
        "deskripsi": (
            "Kondisi kulit kronis yang menyebabkan "
            "peradangan, kemerahan, dan rasa gatal."
        ),
        "gejala": [
            "Kulit kering dan gatal",
            "Kemerahan",
            "Kulit bersisik",
            "Peradangan kulit"
        ],
        "penanganan": (
            "Gunakan pelembap secara rutin dan "
            "hindari pemicu alergi."
        ),
    },

    "Herpes Zoster": {
        "icon": "🟠",
        "color": "#E07A2F",
        "bg": "#FFF5EC",
        "deskripsi": (
            "Infeksi virus akibat reaktivasi "
            "virus varicella-zoster."
        ),
        "gejala": [
            "Ruam merah",
            "Nyeri atau panas",
            "Lepuhan berisi cairan",
            "Kesemutan"
        ],
        "penanganan": (
            "Segera konsultasikan ke dokter "
            "untuk pengobatan antivirus."
        ),
    },

    "Normal": {
        "icon": "🟢",
        "color": "#2E9E6B",
        "bg": "#EDFAF3",
        "deskripsi": (
            "Kulit berada dalam kondisi normal "
            "tanpa indikasi penyakit kulit."
        ),
        "gejala": [
            "Tidak ditemukan tanda penyakit kulit"
        ],
        "penanganan": (
            "Tetap jaga kebersihan dan kesehatan kulit."
        ),
    },

    "Ringworm": {
        "icon": "🟣",
        "color": "#7B61D4",
        "bg": "#F4F1FF",
        "deskripsi": (
            "Infeksi jamur pada kulit dengan "
            "pola melingkar."
        ),
        "gejala": [
            "Ruam melingkar",
            "Gatal",
            "Kulit bersisik",
            "Tepi ruam jelas"
        ],
        "penanganan": (
            "Gunakan krim antijamur dan "
            "jaga kebersihan kulit."
        ),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinScan",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #0F1117;
    border-right: 1px solid #1E2130;
    min-width: 300px !important;
    max-width: 300px !important;
}

section[data-testid="stSidebar"] * {
    color: #F3F4F6 !important;
}

/* BRAND */

.sidebar-brand {
    text-align: center;
    padding: 1rem 0 2rem 0;
    border-bottom: 1px solid #1E2130;
    margin-bottom: 2rem;
}

.sidebar-brand .icon {
    font-size: 3rem;
}

.sidebar-brand .title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-top: 0.3rem;
}

.sidebar-brand .subtitle {
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
    margin-bottom: 12px;
    border: 1px solid transparent;
    transition: 0.2s;
    cursor: pointer;
}

div[role="radiogroup"] > label:hover {
    border: 1px solid #6366F1;
    background: #272C3F;
}

div[role="radiogroup"] label p {
    font-size: 0.95rem !important;
    font-weight: 500;
}

/* TITLE */

.page-title h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    color: #111827;
}

.page-title p {
    color: #6B7280;
    margin-bottom: 2rem;
    font-size: 1rem;
}

/* UPLOAD */

.upload-box {
    border: 2px dashed #D1D5DB;
    border-radius: 20px;
    padding: 4rem 2rem;
    text-align: center;
    background: #FAFAFA;
}

.upload-box .icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

/* RESULT CARD */

.result-card {
    border-radius: 18px;
    padding: 1.5rem;
    border: 1.5px solid;
    margin-bottom: 1.5rem;
}

.result-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    color: #6B7280;
    letter-spacing: 0.08em;
}

.result-name {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-top: 0.3rem;
}

.result-confidence {
    margin-top: 0.5rem;
    font-size: 1rem;
}

/* PROBABILITY */

.prob-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.25rem;
    font-size: 0.9rem;
}

.prob-bg {
    height: 8px;
    border-radius: 999px;
    background: #E5E7EB;
    overflow: hidden;
    margin-bottom: 0.8rem;
}

.prob-fill {
    height: 8px;
    border-radius: 999px;
}

/* INFO CARD */

.info-card {
    border-radius: 18px;
    padding: 1.5rem;
    border: 1px solid;
    margin-bottom: 1rem;
}

.info-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    margin-bottom: 0.8rem;
}

.info-desc {
    color: #4B5563;
    line-height: 1.7;
}

.section-title {
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #6B7280;
    letter-spacing: 0.08em;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORM
# ──────────────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ──────────────────────────────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────────────────────────────
def predict(image):

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = load_model()(tensor)

        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    return CLASSES[probs.argmax()], probs

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <div class="icon">🔬</div>
        <div class="title">SkinScan</div>
        <div class="subtitle">Skin Disease Detection</div>
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

# ──────────────────────────────────────────────────────────────────────────────
# MENU DETEKSI
# ──────────────────────────────────────────────────────────────────────────────
if "Deteksi" in menu:

    st.markdown("""
    <div class="page-title">
        <h1>Deteksi Penyakit Kulit</h1>
        <p>
            Unggah gambar kulit untuk mendapatkan hasil prediksi
            menggunakan model AI ConvNeXt.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded is None:

        st.markdown("""
        <div class="upload-box">
            <div class="icon">🖼️</div>
            <h3>Seret & Lepas Gambar di Sini</h3>
            <p>atau klik Browse Files • JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, use_container_width=True)

        with col2:

            with st.spinner("Menganalisis gambar..."):

                try:

                    label, probs = predict(image)

                    info = DISEASE_INFO[label]

                    confidence = float(probs.max()) * 100

                    st.markdown(f"""
                    <div class="result-card"
                         style="background:{info['bg']};
                                border-color:{info['color']}">

                        <div class="result-label">
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
                            "Silakan konsultasikan dengan dokter kulit."
                        )

                except Exception as e:

                    st.error(f"Error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# MENU INFORMASI
# ──────────────────────────────────────────────────────────────────────────────
elif "Informasi" in menu:

    st.markdown("""
    <div class="page-title">
        <h1>Informasi Penyakit Kulit</h1>
        <p>
            Penjelasan singkat mengenai penyakit kulit
            yang dapat dideteksi model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    cols = [col1, col2]

    for idx, (cls, info) in enumerate(DISEASE_INFO.items()):

        with cols[idx % 2]:

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

                    {info['icon']} {cls}

                </div>

                <div class="info-desc">
                    {info['deskripsi']}
                </div>

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
