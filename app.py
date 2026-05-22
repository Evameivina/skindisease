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
        "deskripsi" : "Kondisi kulit kronis yang menyebabkan peradangan, kemerahan, dan rasa gatal. Sering kambuh dan dipicu oleh alergen, stres, atau perubahan cuaca.",
        "gejala"    : ["Kulit kering dan gatal", "Kemerahan dan peradangan", "Kulit bersisik atau mengelupas", "Bentol-bentol kecil berisi cairan"],
        "penanganan": "Gunakan pelembap secara rutin, hindari pemicu alergi, dan konsultasikan ke dokter untuk mendapatkan krim kortikosteroid atau antihistamin jika diperlukan.",
        "jurnal"    : "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9688004/",
    },
    "Herpes Zoster": {
        "color"     : "#D4721A",
        "bg"        : "#FFF5EC",
        "border"    : "#F5D6B0",
        "icon"      : "🟠",
        "deskripsi" : "Infeksi virus akibat reaktivasi virus varisela-zoster (penyebab cacar air). Ditandai dengan ruam melepuh yang terasa nyeri pada satu sisi tubuh.",
        "gejala"    : ["Nyeri, terbakar, atau kesemutan", "Sensitif terhadap sentuhan", "Ruam merah beberapa hari setelah nyeri", "Lepuhan berisi cairan yang pecah dan mengering"],
        "penanganan": "Segera konsultasi ke dokter untuk mendapatkan obat antivirus. Penanganan dini dapat mengurangi keparahan dan durasi penyakit.",
        "jurnal"    : "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8876683/",
    },
    "Normal": {
        "color"     : "#1E8A5E",
        "bg"        : "#EDFAF3",
        "border"    : "#A8DFC7",
        "icon"      : "🟢",
        "deskripsi" : "Kulit terdeteksi dalam kondisi normal tanpa tanda-tanda penyakit kulit yang signifikan.",
        "gejala"    : ["Tidak ditemukan tanda-tanda penyakit kulit"],
        "penanganan": "Jaga kesehatan kulit dengan rutin membersihkan, melembapkan, dan melindungi dari paparan sinar matahari berlebih.",
        "jurnal"    : None,
    },
    "Ringworm": {
        "color"     : "#6B4FBF",
        "bg"        : "#F4F1FF",
        "border"    : "#C9BFF5",
        "icon"      : "🟣",
        "deskripsi" : "Infeksi jamur pada kulit yang membentuk pola melingkar berwarna merah. Disebabkan oleh jamur, bukan cacing, meski namanya mengandung kata 'worm'.",
        "gejala"    : ["Ruam melingkar berwarna merah", "Tepi ruam lebih menonjol", "Gatal pada area yang terinfeksi", "Kulit bersisik di dalam lingkaran"],
        "penanganan": "Gunakan krim antijamur yang tersedia di apotek. Jaga kebersihan dan keringkan kulit dengan baik. Hindari berbagi handuk atau pakaian.",
        "jurnal"    : "https://pmc.ncbi.nlm.nih.gov/articles/PMC7375854/",
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

/* Hide default nav label */
[data-testid="stSidebar"] .stRadio > label:first-child {
    font-size: 0.68rem !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #555870 !important;
}

/* Radio items styled as nav */
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

            st.markdown("**Distribusi Confidence**")
            for i, cls in enumerate(CLASSES):
                cls_conf = float(probs[i]) * 100
                st.write(f"{cls}: {cls_conf:.1f}%")
                st.progress(cls_conf / 100)

            st.divider()
            st.markdown("**Tentang Kondisi Ini**")
            st.write(info["deskripsi"])

            st.markdown("**Gejala Umum**")
            for g in info["gejala"]:
                st.write(f"— {g}")

            st.markdown("**Penanganan**")
            st.info(info["penanganan"])

            if info.get("jurnal"):
                st.markdown(f"[Lihat Jurnal Referensi]({info['jurnal']})")

            if label != "Normal":
                st.warning("Hasil ini **bukan diagnosis medis**. Segera konsultasikan ke dokter kulit untuk pemeriksaan lebih lanjut.")

# =========================================================
# MENU: INFORMASI
# =========================================================
elif "Informasi" in menu:

    st.markdown("## Informasi Penyakit Kulit")
    st.caption("Pilih kondisi kulit di bawah untuk melihat penjelasan lengkapnya.")

    # Tombol pilih penyakit
    btn_cols = st.columns(4, gap="small")
    for idx, cls in enumerate(DISEASE_INFO.keys()):
        with btn_cols[idx]:
            if st.button(cls, key=f"btn_{cls}", use_container_width=True):
                st.session_state["selected_info"] = cls
                st.rerun()

    st.divider()

    # Tampilkan detail jika sudah dipilih
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

            st.markdown("**Gejala Umum**")
            for g in info["gejala"]:
                st.write(f"— {g}")

            st.markdown("**Penanganan**")
            st.info(info["penanganan"])

            if info.get("jurnal"):
                st.markdown(f"[Lihat Jurnal Referensi]({info['jurnal']})")

    st.divider()
    st.warning("Informasi di atas bersifat **edukatif**. Selalu konsultasikan kondisi kulit Anda kepada tenaga medis profesional.")
