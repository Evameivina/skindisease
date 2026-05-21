import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
MODEL_PATH = "convnext_skin_state_dict.pth"

CLASSES = ["Eczema", "Herpes Zoster", "Normal", "Ringworm"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASE_INFO = {
    "Eczema": {
        "icon": "🔴",
        "color": "#E05C5C",
        "bg": "#FFF0F0",
        "deskripsi": "Kondisi kulit kronis yang menyebabkan peradangan, kemerahan, dan rasa gatal.",
        "gejala": [
            "Kulit kering dan gatal",
            "Kemerahan dan peradangan",
            "Kulit bersisik",
            "Bentol kecil berisi cairan"
        ],
        "penanganan": "Gunakan pelembap secara rutin dan konsultasikan ke dokter kulit."
    },

    "Herpes Zoster": {
        "icon": "🟠",
        "color": "#E07A2F",
        "bg": "#FFF5EC",
        "deskripsi": "Infeksi virus akibat reaktivasi virus varisela-zoster.",
        "gejala": [
            "Nyeri atau kesemutan",
            "Ruam merah",
            "Lepuhan berisi cairan",
            "Sensitif terhadap sentuhan"
        ],
        "penanganan": "Segera konsultasi ke dokter untuk mendapatkan antivirus."
    },

    "Normal": {
        "icon": "🟢",
        "color": "#2E9E6B",
        "bg": "#EDFAF3",
        "deskripsi": "Kulit terdeteksi dalam kondisi normal.",
        "gejala": [
            "Tidak ditemukan tanda penyakit kulit"
        ],
        "penanganan": "Jaga kesehatan kulit dan gunakan sunscreen."
    },

    "Ringworm": {
        "icon": "🟣",
        "color": "#7B61D4",
        "bg": "#F4F1FF",
        "deskripsi": "Infeksi jamur pada kulit berbentuk melingkar.",
        "gejala": [
            "Ruam melingkar",
            "Kulit bersisik",
            "Gatal",
            "Tepi ruam lebih jelas"
        ],
        "penanganan": "Gunakan krim antijamur dan jaga kebersihan kulit."
    },
}

# ── Model ─────────────────────────────────────────────────────────────────────
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

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict(image):
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = load_model()(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    return CLASSES[probs.argmax()], probs

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinScan",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header {
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
}

section[data-testid="stSidebar"] * {
    color: #E5E7EB !important;
}

.sidebar-brand {
    text-align: center;
    padding: 1.5rem 0 2rem;
    border-bottom: 1px solid #1E2130;
    margin-bottom: 2rem;
}

.sidebar-brand .icon {
    font-size: 2.8rem;
}

.sidebar-brand .title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    margin-top: 0.4rem;
}

.sidebar-brand .subtitle {
    font-size: 0.75rem;
    color: #9CA3AF !important;
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
    background: #262B3D;
    border: 1px solid #4F46E5;
}

div[role="radiogroup"] label p {
    font-size: 0.95rem !important;
    font-weight: 500;
}

/* PAGE */

.page-title h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
}

.page-title p {
    color: #6B7280;
    margin-bottom: 2rem;
}

/* UPLOAD */

.upload-box {
    border: 2px dashed #D1D5DB;
    border-radius: 18px;
    padding: 3rem;
    text-align: center;
    background: #FAFAFA;
}

.upload-box .icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}

/* RESULT */

.result-card {
    border-radius: 18px;
    padding: 1.5rem;
    border: 1.5px solid;
    margin-bottom: 1.5rem;
}

.result-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6B7280;
}

.result-name {
    font-size: 2rem;
    font-family: 'DM Serif Display', serif;
}

.confidence {
    font-size: 1rem;
}

/* PROBABILITY */

.prob-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.2rem;
    font-size: 0.9rem;
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
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid;
}

.info-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    margin-bottom: 0.6rem;
}

.info-desc {
    color: #4B5563;
    margin-bottom: 1rem;
}

.info-section {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #6B7280;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

# ── MENU 1 : DETEKSI ──────────────────────────────────────────────────────────
if "Deteksi" in menu:

    st.markdown("""
    <div class="page-title">
        <h1>Deteksi Penyakit Kulit</h1>
        <p>Unggah foto kulit untuk mendapatkan hasil prediksi dari model AI.</p>
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
            <div class="icon">🖼️</div>
            <h3>Seret & lepas gambar di sini</h3>
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
                         style="border-color:{info['color']};
                                background:{info['bg']}">

                        <div class="result-title">
                            Hasil Prediksi
                        </div>

                        <div class="result-name"
                             style="color:{info['color']}">
                             {info['icon']} {label}
                        </div>

                        <div class="confidence">
                            Confidence:
                            <b>{confidence:.2f}%</b>
                        </div>

                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### Probabilitas")

                    for i, cls in enumerate(CLASSES):

                        pct = float(probs[i]) * 100

                        color = DISEASE_INFO[cls]["color"]

                        st.markdown(f"""
                        <div class="prob-label">
                            <span>{DISEASE_INFO[cls]['icon']} {cls}</span>
                            <span>{pct:.1f}%</span>
                        </div>

                        <div class="prob-bg">
                            <div class="prob-fill"
                                 style="width:{pct}%;
                                        background:{color}">
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

# ── MENU 2 : INFORMASI ────────────────────────────────────────────────────────
elif "Informasi" in menu:

    st.markdown("""
    <div class="page-title">
        <h1>Informasi Penyakit Kulit</h1>
        <p>Penjelasan singkat mengenai penyakit yang dapat dideteksi model.</p>
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
                 style="background:{info['bg']};
                        border-color:{info['color']}40">

                <div class="info-name"
                     style="color:{info['color']}">
                     {info['icon']} {cls}
                </div>

                <div class="info-desc">
                    {info['deskripsi']}
                </div>

                <div class="info-section">
                    Gejala Umum
                </div>

                <ul>
                    {gejala_html}
                </ul>

                <div class="info-section">
                    Penanganan
                </div>

                <p>{info['penanganan']}</p>

            </div>
            """, unsafe_allow_html=True)

    st.info(
        "Informasi ini bersifat edukatif dan tidak menggantikan diagnosis medis profesional."
    )
