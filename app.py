import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

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
CLASSES = ["Eczema", "Herpes Zoster", "Normal", "Ringworm"]

DISEASE_INFO = {
    "Eczema": {
        "icon": "💧",
        "color": "#4A90D9",
        "description": "Eczema (dermatitis atopik) adalah kondisi kulit yang ditandai dengan peradangan dan rasa gatal yang intens. Umumnya muncul sebagai ruam kemerahan, kulit kering dan bersisik.",
        "gejala": ["Rasa gatal yang intens, terutama malam hari", "Kulit kemerahan dan meradang", "Kulit kering, sensitif, dan bersisik", "Ruam yang mungkin bernanah dan mengeras"],
        "penanganan": "Konsultasikan dengan dokter kulit. Penggunaan pelembab secara rutin dan menghindari pemicu (iritan, alergen) sangat dianjurkan.",
        "tingkat_risiko": "Sedang"
    },
    "Herpes Zoster": {
        "icon": "⚡",
        "color": "#E74C3C",
        "description": "Herpes Zoster (cacar api) disebabkan oleh reaktivasi virus varicella-zoster. Ditandai dengan ruam melepuh yang menyakitkan pada satu sisi tubuh mengikuti jalur saraf.",
        "gejala": ["Nyeri, rasa terbakar, atau kesemutan", "Sensitivitas berlebihan terhadap sentuhan", "Ruam kemerahan yang muncul beberapa hari setelah nyeri", "Lepuhan berisi cairan yang pecah dan mengeras"],
        "penanganan": "Segera konsultasikan ke dokter. Antivirus paling efektif jika diberikan dalam 72 jam pertama setelah munculnya ruam.",
        "tingkat_risiko": "Tinggi"
    },
    "Normal": {
        "icon": "✅",
        "color": "#27AE60",
        "description": "Kulit dalam kondisi normal dan sehat. Tidak ditemukan tanda-tanda kondisi kulit yang memerlukan perhatian medis khusus.",
        "gejala": ["Tidak ada keluhan kulit yang berarti", "Warna kulit merata dan sehat", "Tekstur kulit halus dan terhidrasi", "Tidak ada lesi atau iritasi"],
        "penanganan": "Pertahankan rutinitas perawatan kulit yang baik: pembersihan rutin, pelembab, dan perlindungan matahari.",
        "tingkat_risiko": "Tidak Ada"
    },
    "Ringworm": {
        "icon": "🔄",
        "color": "#F39C12",
        "description": "Ringworm (tinea corporis) adalah infeksi jamur yang ditandai dengan ruam melingkar berbatas jelas. Meskipun namanya mengandung 'worm', ini adalah infeksi jamur, bukan cacing.",
        "gejala": ["Ruam berbentuk cincin atau bulat dengan tepi jelas", "Tepi ruam tampak lebih menonjol dan bersisik", "Rasa gatal pada area yang terinfeksi", "Rambut di area infeksi dapat rontok (jika di kulit kepala)"],
        "penanganan": "Antijamur topikal biasanya efektif. Jaga kebersihan dan hindari berbagi handuk atau pakaian untuk mencegah penularan.",
        "tingkat_risiko": "Rendah"
    }
}

RISK_COLORS = {
    "Tidak Ada": "#27AE60",
    "Rendah": "#F39C12",
    "Sedang": "#E67E22",
    "Tinggi": "#E74C3C"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    .stApp {
        background: #0D1117;
        color: #E6EDF3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #161B22;
        border-right: 1px solid #30363D;
    }

    [data-testid="stSidebar"] * {
        color: #E6EDF3 !important;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Main title */
    .main-title {
        font-family: 'Sora', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #58A6FF 0%, #79C0FF 50%, #A5F3FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #8B949E;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.02em;
        margin-bottom: 2rem;
    }

    /* Cards */
    .scan-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }

    .scan-card:hover {
        border-color: #58A6FF;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #161B22 0%, #1C2128 100%);
        border: 1px solid #30363D;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .result-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #58A6FF, #79C0FF, #A5F3FC);
    }

    .result-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #8B949E;
        margin-bottom: 0.3rem;
        font-family: 'DM Mono', monospace;
    }

    .result-class {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .confidence-bar-container {
        background: #21262D;
        border-radius: 100px;
        height: 8px;
        margin: 0.5rem 0;
        overflow: hidden;
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #58A6FF, #A5F3FC);
        transition: width 0.8s ease;
    }

    .info-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
    }

    .info-card h4 {
        color: #58A6FF;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'DM Mono', monospace;
    }

    .symptom-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.3rem 0;
        font-size: 0.9rem;
        color: #C9D1D9;
    }

    .symptom-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #58A6FF;
        margin-top: 0.4rem;
        flex-shrink: 0;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'DM Mono', monospace;
    }

    .disclaimer {
        background: #1C2128;
        border: 1px solid #F0883E40;
        border-left: 3px solid #F0883E;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-top: 1.5rem;
        font-size: 0.82rem;
        color: #C9D1D9;
    }

    .disclaimer strong {
        color: #F0883E;
    }

    .all-scores-row {
        display: flex;
        gap: 0.75rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }

    .score-chip {
        background: #21262D;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 0.5rem 0.85rem;
        font-size: 0.82rem;
        font-family: 'DM Mono', monospace;
        color: #8B949E;
        flex: 1;
        min-width: 120px;
        text-align: center;
    }

    .score-chip.top {
        border-color: #58A6FF;
        color: #58A6FF;
    }

    .score-chip .score-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        display: block;
        margin-bottom: 0.2rem;
        color: #6E7681;
    }

    .score-chip.top .score-label {
        color: #79C0FF;
    }

    .score-chip .score-val {
        font-size: 1rem;
        font-weight: 500;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #161B22;
        border: 2px dashed #30363D;
        border-radius: 12px;
        transition: border-color 0.2s;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #58A6FF;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1F6FEB, #58A6FF) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em !important;
        transition: opacity 0.2s !important;
        width: 100%;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* Radio */
    .stRadio [role="radiogroup"] label {
        color: #C9D1D9 !important;
        font-size: 0.9rem;
    }

    /* Divider */
    hr {
        border-color: #30363D !important;
        margin: 1.5rem 0 !important;
    }

    /* Metric */
    [data-testid="stMetric"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #8B949E !important;
        font-size: 0.8rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #E6EDF3 !important;
        font-family: 'DM Mono', monospace !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {display: none !important;}

    .block-container {
        padding-top: 2rem !important;
        max-width: 1100px !important;
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
    state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)

# ─────────────────────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
def predict(image: Image.Image, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return CLASSES[pred_idx], probs

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 DermaScan")
    st.markdown("<p style='color:#8B949E; font-size:0.82rem; margin-top:-0.5rem;'>v1.0 · ConvNeXt-Tiny</p>", unsafe_allow_html=True)
    st.markdown("---")

    menu = st.radio(
        "Navigasi",
        ["🩺 Deteksi Penyakit Kulit", "📚 Informasi Penyakit"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#6E7681; font-size:0.78rem; line-height:1.6;'>
    <strong style='color:#8B949E;'>Kelas yang Didukung</strong><br>
    • Eczema<br>
    • Herpes Zoster<br>
    • Normal<br>
    • Ringworm
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#6E7681; font-size:0.75rem;'>
    ⚠️ Hanya untuk screening awal.<br>Bukan pengganti diagnosis medis.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: DETEKSI
# ─────────────────────────────────────────────────────────────
if "🩺 Deteksi" in menu:

    st.markdown('<p class="main-title">Deteksi Penyakit Kulit</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Unggah foto kulit untuk mendapatkan hasil klasifikasi menggunakan model ConvNeXt-Tiny</p>', unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### 📁 Unggah Gambar")
        uploaded = st.file_uploader(
            "Pilih gambar kulit (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Lebar", f"{image.width} px")
            with col_info2:
                st.metric("Tinggi", f"{image.height} px")

            run_btn = st.button("🔍 Analisis Sekarang", use_container_width=True)
        else:
            st.markdown("""
            <div style='text-align:center; padding:3rem 1rem; color:#6E7681; font-size:0.9rem;'>
                <div style='font-size:3rem; margin-bottom:0.75rem;'>🖼️</div>
                Seret & lepas gambar di sini<br>atau klik untuk memilih file
            </div>
            """, unsafe_allow_html=True)
            run_btn = False

    with col_result:
        st.markdown("#### 📊 Hasil Analisis")

        if uploaded and run_btn:
            with st.spinner("Menganalisis gambar..."):
                try:
                    model = load_model()
                    pred_class, probs = predict(image, model)
                    info = DISEASE_INFO[pred_class]
                    top_conf = float(probs[CLASSES.index(pred_class)]) * 100

                    # ── Result box ──
                    color = info["color"]
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Hasil Klasifikasi</div>
                        <div class="result-class" style="color:{color};">{info['icon']} {pred_class}</div>
                        <div class="result-label" style="margin-top:0.75rem;">Confidence Score</div>
                        <div style="font-size:1.8rem; font-family:'DM Mono',monospace; font-weight:600; color:{color};">{top_conf:.1f}%</div>
                        <div class="confidence-bar-container">
                            <div class="confidence-bar-fill" style="width:{top_conf:.1f}%; background:linear-gradient(90deg,{color}88,{color});"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── All class scores ──
                    chips_html = '<div class="all-scores-row">'
                    for i, cls in enumerate(CLASSES):
                        pct = float(probs[i]) * 100
                        is_top = cls == pred_class
                        chips_html += f"""
                        <div class="score-chip {'top' if is_top else ''}">
                            <span class="score-label">{cls}</span>
                            <span class="score-val">{pct:.1f}%</span>
                        </div>"""
                    chips_html += '</div>'
                    st.markdown(chips_html, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Disease detail ──
                    risk_color = RISK_COLORS[info["tingkat_risiko"]]
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>Deskripsi</h4>
                        <p style="color:#C9D1D9; font-size:0.88rem; line-height:1.6; margin:0;">{info['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    gejala_items = "".join([
                        f'<div class="symptom-item"><div class="symptom-dot"></div><span>{g}</span></div>'
                        for g in info["gejala"]
                    ])
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>Gejala Umum</h4>
                        {gejala_items}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="info-card">
                        <h4>Tingkat Risiko</h4>
                        <span class="risk-badge" style="background:{risk_color}22; color:{risk_color}; border:1px solid {risk_color}44;">
                            {info['tingkat_risiko']}
                        </span>
                        <p style="color:#C9D1D9; font-size:0.85rem; line-height:1.6; margin:0.75rem 0 0 0;">{info['penanganan']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="disclaimer">
                        <strong>⚠️ Disclaimer:</strong> Hasil ini merupakan output screening awal berbasis AI dan 
                        <strong>tidak menggantikan diagnosis medis</strong> oleh tenaga kesehatan profesional. 
                        Segera konsultasikan ke dokter untuk penanganan lebih lanjut.
                    </div>
                    """, unsafe_allow_html=True)

                except FileNotFoundError:
                    st.error(f"❌ File model tidak ditemukan: `{MODEL_PATH}`\n\nPastikan file model berada di direktori yang sama dengan `app.py`.")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {e}")

        elif not uploaded:
            st.markdown("""
            <div style='text-align:center; padding:4rem 1rem; color:#6E7681;'>
                <div style='font-size:2.5rem; margin-bottom:0.75rem;'>🔬</div>
                <div style='font-size:0.9rem;'>Unggah gambar terlebih dahulu<br>untuk melihat hasil analisis</div>
            </div>
            """, unsafe_allow_html=True)

        elif not run_btn:
            st.markdown("""
            <div style='text-align:center; padding:4rem 1rem; color:#6E7681;'>
                <div style='font-size:2.5rem; margin-bottom:0.75rem;'>👆</div>
                <div style='font-size:0.9rem;'>Klik tombol <strong style='color:#58A6FF;'>Analisis Sekarang</strong><br>untuk memulai klasifikasi</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: INFORMASI PENYAKIT
# ─────────────────────────────────────────────────────────────
elif "📚 Informasi" in menu:

    st.markdown('<p class="main-title">Informasi Penyakit Kulit</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Panduan singkat mengenai empat kondisi kulit yang dapat dideteksi oleh aplikasi ini</p>', unsafe_allow_html=True)

    for cls, info in DISEASE_INFO.items():
        color = info["color"]
        risk_color = RISK_COLORS[info["tingkat_risiko"]]

        with st.expander(f"{info['icon']}  {cls}", expanded=(cls == "Eczema")):
            col1, col2 = st.columns([3, 2], gap="large")

            with col1:
                st.markdown(f"""
                <div style='color:#C9D1D9; font-size:0.92rem; line-height:1.7; margin-bottom:1rem;'>
                    {info['description']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"<p style='color:{color}; font-weight:600; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.1em; font-family:DM Mono,monospace;'>Gejala Umum</p>", unsafe_allow_html=True)
                for g in info["gejala"]:
                    st.markdown(f"""
                    <div class="symptom-item" style='margin-bottom:0.1rem;'>
                        <div class="symptom-dot" style='background:{color};'></div>
                        <span style='font-size:0.88rem;'>{g}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style='background:#21262D; border:1px solid #30363D; border-radius:10px; padding:1rem;'>
                    <p style='color:#8B949E; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; font-family:DM Mono,monospace; margin-bottom:0.3rem;'>Tingkat Risiko</p>
                    <span class="risk-badge" style="background:{risk_color}22; color:{risk_color}; border:1px solid {risk_color}44; font-size:0.85rem;">
                        {info['tingkat_risiko']}
                    </span>
                    <p style='color:#8B949E; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; font-family:DM Mono,monospace; margin-top:1rem; margin-bottom:0.5rem;'>Penanganan</p>
                    <p style='color:#C9D1D9; font-size:0.85rem; line-height:1.6; margin:0;'>{info['penanganan']}</p>
                </div>
                """, unsafe_allow_html=True)
