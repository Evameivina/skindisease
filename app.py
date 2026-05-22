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
    page_title="SkinScan — Deteksi Penyakit Kulit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CONFIG
# =========================================================
CLASSES    = ["Eczema", "Herpes Zoster", "Normal", "Ringworm"]
MODEL_URL  = "https://drive.google.com/uc?id=1s2BhRSSuUTRpuANjXkzTYSEAi_wKTuLR"
MODEL_PATH = "convnext_skin_state_dict.pth"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASE_INFO = {
    "Eczema": {
        "color"     : "#E05C5C",
        "bg"        : "#FFF0F0",
        "border"    : "#F5C6C6",
        "icon"      : "🔴",
        "deskripsi" : (
            "Atopic dermatitis (AD) adalah penyakit kulit inflamasi kronis yang ditandai dengan pruritus intens, "
            "kulit kering (xerosis), dan papul kemerahan yang bersifat residif. "
            "Prevalensinya sekitar 20% pada anak-anak dan 1–3% pada dewasa, "
            "dan menempati peringkat pertama di antara semua penyakit kulit berdasarkan disability-adjusted life-years (DALYs) global."
        ),
        "gejala"    : [
            "Pruritus (gatal intens), terutama memburuk pada malam hari",
            "Kulit kering, menebal, dan bersisik (xerosis/iktiosis)",
            "Papul kemerahan yang dapat mengeluarkan cairan jika digaruk",
            "Lesi pada area fleksural (siku, lutut) atau wajah pada bayi",
            "Bersifat kronis dengan periode remisi dan eksaserbasi (flare)",
        ],
        "jurnal"    : "https://pmc.ncbi.nlm.nih.gov/articles/PMC10944924/",
        "sumber"    : "Afshari et al., Front. Immunol. 2024",
    },
    "Herpes Zoster": {
        "color"     : "#D4721A",
        "bg"        : "#FFF5EC",
        "border"    : "#F5D6B0",
        "icon"      : "🟡",
        "deskripsi" : (
            "Herpes zoster adalah reaktivasi virus Varicella-Zoster (VZV) yang sebelumnya bersifat laten di ganglia sensoris "
            "setelah infeksi cacar air. Reaktivasi umumnya dipicu oleh penurunan imunitas akibat usia lanjut, stres, atau imunosupresi, "
            "dan menimbulkan ruam vesikel yang nyeri mengikuti satu jalur dermaton."
        ),
        "gejala"    : [
            "Fase awal: nyeri, rasa terbakar, atau kesemutan pada satu sisi tubuh",
            "Vesikel multipel yang terasa nyeri mengikuti jalur dermaton",
            "Gejala prodromal: sakit kepala, malaise umum, dan fotofobia",
            "Nyeri yang dapat berlanjut setelah ruam sembuh (postherpetic neuralgia)",
        ],
        "jurnal"    : "https://pmc.ncbi.nlm.nih.gov/articles/PMC8876683/",
        "sumber"    : "Patil et al., Viruses 2022",
    },
    "Normal": {
        "color"     : "#1E8A5E",
        "bg"        : "#EDFAF3",
        "border"    : "#A8DFC7",
        "icon"      : "🟢",
        "deskripsi" : (
            "Kulit normal merupakan kondisi kulit sehat dengan fungsi barrier yang optimal. "
            "Kulit tersusun dari tiga lapisan utama — epidermis, dermis, dan hipodermis — "
            "yang bersama-sama melindungi tubuh dari patogen, radiasi UV, bahan kimia, dan cedera mekanis, "
            "sekaligus berperan dalam regulasi suhu tubuh."
        ),
        "gejala"    : [
            "Tidak ditemukan tanda-tanda kelainan atau penyakit kulit",
            "Fungsi barrier kulit optimal — mencegah kehilangan air dan invasi patogen",
            "Warna dan tekstur kulit merata, terhidrasi dengan baik",
            "Tidak ada lesi, kemerahan, bersisik, atau iritasi",
        ],
        "jurnal"    : "https://pmc.ncbi.nlm.nih.gov/articles/PMC11597055/",
        "sumber"    : "Brito et al., Pharmaceutics 2024",
    },
    "Ringworm": {
        "color"     : "#6B4FBF",
        "bg"        : "#F4F1FF",
        "border"    : "#C9BFF5",
        "icon"      : "🟠",
        "deskripsi" : (
            "Tinea corporis (ringworm) adalah infeksi jamur superfisial pada kulit yang disebabkan oleh dermatofita, "
            "paling umum Trichophyton rubrum. Penyakit ini menampilkan lesi anular (berbentuk cincin) berbatas jelas "
            "dengan hipopigmentasi sentral, dan merupakan kondisi kulit paling prevalen di dunia "
            "dengan estimasi risiko seumur hidup sebesar 10–20%."
        ),
        "gejala"    : [
            "Papul dan plak anular (cincin) berbatas jelas dengan tepi lebih menonjol",
            "Hipopigmentasi atau clearing pada bagian tengah lesi",
            "Skuama (sisik) pada tepi lesi yang aktif",
            "Rasa gatal (pruritus) pada area yang terinfeksi",
            "Pada kulit gelap, lesi dapat tampak violaseus atau hiperpigmentasi",
        ],
        "jurnal"    : "https://pmc.ncbi.nlm.nih.gov/articles/PMC12971098/",
        "sumber"    : "Van Alfen et al., HCA Healthcare J Med 2026",
    },
}

# =========================================================
# DOWNLOAD MODEL
# =========================================================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model, harap tunggu..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp { background-color: #F7F8FC; }
.block-container { padding-top: 2.5rem !important; padding-bottom: 2.5rem !important; max-width: 1200px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1A1D2E !important;
    border-right: 1px solid #2A2D3E;
}
[data-testid="stSidebar"] * { color: #C8CCDF !important; }
[data-testid="stSidebar"] hr { border-color: #2A2D3E !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.88rem !important; }

[data-testid="stSidebar"] .stRadio > label:first-child {
    font-size: 0.68rem !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #555870 !important;
}

[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.9rem !important;
    font-weight: 500;
}

/* Headings */
h1 { font-weight: 700 !important; letter-spacing: -0.03em !important; color: #0F1117 !important; }
h2 { font-weight: 600 !important; letter-spacing: -0.02em !important; }
h3 { font-weight: 600 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 14px;
    border: 1.5px dashed #CBD5E1;
    padding: 0.5rem;
}

/* Metric */
[data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    border: 1.5px solid #E5E9F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #8896AB !important; }
[data-testid="stMetricValue"] { font-size: 1.9rem !important; font-weight: 700 !important; }

/* Progress bar */
[data-testid="stProgress"] > div > div { border-radius: 99px !important; }
[data-testid="stProgress"] { border-radius: 99px !important; }

/* Source tag */
.source-tag {
    display: inline-block;
    margin-top: 0.5rem;
    font-size: 0.72rem;
    color: #8896AB;
    background: #F1F3F8;
    border: 1px solid #E2E6EF;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    text-decoration: none;
    font-style: italic;
}
.source-tag:hover {
    color: #4F6AF0;
    border-color: #4F6AF0;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    m = models.convnext_tiny(weights=None)
    in_features = m.classifier[2].in_features
    m.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(CLASSES))
    )
    state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    m.load_state_dict(state)
    m.eval()
    return m.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return CLASSES[np.argmax(probs)], probs

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align:center;padding:1.75rem 0 1.5rem;">
            <div style="font-size:1.5rem;font-weight:700;color:#FFFFFF!important;letter-spacing:-0.02em;margin-top:0.3rem">SkinScan</div>
            <div style="font-size:0.7rem;color:#555870!important;letter-spacing:0.07em;text-transform:uppercase;margin-top:0.2rem">Skin Disease Detection</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2A2D3E;margin:0 0 1rem 0'>", unsafe_allow_html=True)

    menu = st.radio(
        "MENU",
        ["Deteksi", "Informasi"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2A2D3E;margin:1rem 0'>", unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size:0.72rem;color:#555870;line-height:1.7;padding:0 0.25rem">
            <b style="color:#8896AB">Disclaimer</b><br>
            Aplikasi ini hanya untuk tujuan edukatif. Bukan pengganti diagnosis medis profesional.
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# MENU: DETEKSI
# =========================================================
if "Deteksi" in menu:

    st.markdown("## Deteksi Penyakit Kulit")
    st.caption("Unggah foto kulit untuk mendapatkan hasil klasifikasi dan confidence score dari model.")

    with st.expander("Cara Penggunaan & Kelas yang Didukung", expanded=False):
        col_t1, col_t2 = st.columns(2, gap="medium")
        with col_t1:
            st.markdown("**Kelas yang Dapat Dideteksi**")
            st.write("- Eczema — eksim / dermatitis")
            st.write("- Herpes Zoster — cacar api / shingles")
            st.write("- Normal — kulit sehat")
            st.write("- Ringworm — kurap / tinea corporis")
            st.warning("Gambar di luar 4 kategori ini tetap akan diprediksi ke salah satu kelas, namun hasilnya tidak dapat diandalkan.")
        with col_t2:
            st.markdown("**Tutorial Cara Pakai**")
            st.write("1. Siapkan foto kulit yang jelas dan cukup cahaya")
            st.write("2. Pastikan area yang bermasalah terlihat jelas di foto")
            st.write("3. Klik **Browse files** atau seret gambar ke area upload")
            st.write("4. Tunggu hingga model selesai menganalisis")
            st.write("5. Baca hasil prediksi dan confidence score")
            st.write("6. Jika terdeteksi penyakit, konsultasikan ke dokter kulit")
            st.info("Tips: Gunakan foto resolusi tinggi dengan latar belakang polos untuk hasil terbaik.")

    uploaded_file = st.file_uploader(
        "Pilih gambar kulit (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.info("Unggah foto kulit (JPG/PNG) menggunakan tombol di atas.")

    else:
        image = Image.open(uploaded_file)

        with st.spinner("Menganalisis gambar..."):
            model  = load_model()
            label, probs = predict(image, model)

        info = DISEASE_INFO[label]
        conf = float(probs[CLASSES.index(label)]) * 100

        _, center, _ = st.columns([1, 2, 1])

        with center:
            st.image(image, use_container_width=True)
            st.divider()

            st.subheader(f"Hasil: {label}")
            st.metric(label="Confidence Score", value=f"{conf:.2f}%")
            st.divider()

            st.markdown("**Tentang Kondisi Ini**")
            st.write(info["deskripsi"])
            st.markdown(
                f'<a href="{info["jurnal"]}" target="_blank" class="source-tag">📄 {info["sumber"]} ↗</a>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Gejala Umum**")
            for g in info["gejala"]:
                st.write(f"— {g}")

            if label != "Normal":
                st.warning("Hasil ini **bukan diagnosis medis**. Segera konsultasikan ke dokter kulit untuk pemeriksaan lebih lanjut.")

# =========================================================
# MENU: INFORMASI
# =========================================================
elif "Informasi" in menu:

    st.markdown("## Informasi Penyakit Kulit")
    st.caption("Pilih kondisi kulit di bawah untuk melihat penjelasan lengkapnya.")

    btn_cols = st.columns(4, gap="small")
    for idx, cls in enumerate(DISEASE_INFO.keys()):
        with btn_cols[idx]:
            if st.button(cls, key=f"btn_{cls}", use_container_width=True):
                st.session_state["selected_info"] = cls
                st.rerun()

    st.divider()

    selected = st.session_state.get("selected_info", None)

    if selected is None:
        st.info("Pilih salah satu kondisi di atas untuk melihat informasi lengkapnya.")

    else:
        info = DISEASE_INFO[selected]

        _, center_info, _ = st.columns([1, 3, 1])
        with center_info:
            st.subheader(selected)
            st.divider()

            st.markdown("**Deskripsi**")
            st.write(info["deskripsi"])
            st.markdown(
                f'<a href="{info["jurnal"]}" target="_blank" class="source-tag">📄 {info["sumber"]} ↗</a>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Gejala Umum**")
            for g in info["gejala"]:
                st.write(f"— {g}")

    st.divider()
    st.warning("Informasi di atas bersifat **edukatif**. Bukan pengganti diagnosis medis profesional. Selalu konsultasikan kondisi kulit Anda kepada tenaga medis profesional.")
