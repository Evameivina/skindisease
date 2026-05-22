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
            <div style="font-size:2.2rem">🔬</div>
            <div style="font-size:1.5rem;font-weight:700;color:#FFFFFF!important;letter-spacing:-0.02em;margin-top:0.3rem">SkinScan</div>
            <div style="font-size:0.7rem;color:#555870!important;letter-spacing:0.07em;text-transform:uppercase;margin-top:0.2rem">Skin Disease Detection</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:0 0 1.25rem'>", unsafe_allow_html=True)

    menu = st.radio(
        "NAVIGASI",
        ["🩺  Deteksi Penyakit Kulit", "📖  Informasi Penyakit"],
        label_visibility="visible"
    )



# =========================================================
# MENU: DETEKSI
# =========================================================
if "Deteksi" in menu:

    st.markdown("## 🩺 Deteksi Penyakit Kulit")
    st.markdown(
        "<p style='color:#6B7280;font-size:0.95rem;margin-top:-0.5rem;margin-bottom:1.75rem'>"
        "Unggah foto kulit untuk mendapatkan hasil klasifikasi dan confidence score dari model.</p>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Pilih gambar kulit (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.markdown("""
            <div style="
                border: 2px dashed #CBD5E1;
                border-radius: 16px;
                background: white;
                padding: 3.5rem 2rem;
                text-align: center;
            ">
                <div style="font-size:2.5rem;margin-bottom:0.75rem">🖼️</div>
                <div style="font-size:1rem;font-weight:600;color:#374151;margin-bottom:0.3rem">Seret & lepas gambar di sini</div>
                <div style="font-size:0.83rem;color:#9CA3AF">atau klik <b>Browse files</b> di atas &nbsp;·&nbsp; JPG, JPEG, PNG</div>
            </div>
        """, unsafe_allow_html=True)

    else:
        image = Image.open(uploaded_file)

        with st.spinner("Menganalisis gambar..."):
            model  = load_model()
            label, probs = predict(image, model)

        info = DISEASE_INFO[label]
        conf = float(probs[CLASSES.index(label)]) * 100

        # Layout terpusat vertikal
        _, center, _ = st.columns([1, 2, 1])

        with center:
            # Gambar
            st.image(image, use_container_width=True, caption="")
            st.markdown("<div style='margin-bottom:1.25rem'></div>", unsafe_allow_html=True)

            # Badge hasil
            st.markdown(f"""
                <div style="
                    background:{info['bg']};
                    border:1.5px solid {info['border']};
                    border-radius:16px;
                    padding:1.5rem 1.75rem;
                    margin-bottom:1.25rem;
                ">
                    <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;color:#8896AB;margin-bottom:0.5rem">
                        Hasil Klasifikasi
                    </div>
                    <div style="font-size:2rem;font-weight:700;color:{info['color']};letter-spacing:-0.02em;margin-bottom:0.15rem">
                        {info['icon']} {label}
                    </div>
                    <div style="font-size:0.85rem;color:#6B7280">
                        Confidence Score: &nbsp;
                        <span style="font-size:1.5rem;font-weight:700;color:{info['color']}">{conf:.2f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Deskripsi penyakit
            st.markdown(
                "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;"
                "color:#8896AB;margin:1.25rem 0 0.5rem'>Tentang Kondisi Ini</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size:0.88rem;color:#374151;line-height:1.7;margin-bottom:0.75rem'>"
                f"{info['deskripsi']}</div>",
                unsafe_allow_html=True
            )

            # Gejala
            st.markdown(
                "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;"
                "color:#8896AB;margin-bottom:0.45rem'>Gejala Umum</div>",
                unsafe_allow_html=True
            )
            gejala_html = "".join([
                f"<div style='display:flex;gap:0.45rem;align-items:flex-start;padding:0.22rem 0;"
                f"font-size:0.85rem;color:#374151'>"
                f"<div style='width:6px;height:6px;border-radius:50%;background:{info['color']};"
                f"margin-top:0.4rem;flex-shrink:0'></div><span>{g}</span></div>"
                for g in info["gejala"]
            ])
            st.markdown(gejala_html, unsafe_allow_html=True)

            # Penanganan
            st.markdown(
                "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;"
                "color:#8896AB;margin:0.85rem 0 0.45rem'>Penanganan</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='background:rgba(0,0,0,0.03);border-radius:8px;padding:0.7rem 0.9rem;"
                f"font-size:0.85rem;color:#4B5563;line-height:1.65'>{info['penanganan']}</div>",
                unsafe_allow_html=True
            )

            # Link jurnal
            if info.get("jurnal"):
                st.markdown(
                    f"<div style='margin-top:0.75rem'>"
                    f"<a href='{info['jurnal']}' target='_blank' style='font-size:0.83rem;"
                    f"color:{info['color']};font-weight:600;text-decoration:none'>"
                    f"📄 Lihat Jurnal Referensi →</a></div>",
                    unsafe_allow_html=True
                )

            # Disclaimer
            if label != "Normal":
                st.markdown("""
                    <div style="
                        margin-top:1rem;
                        background:#FFFBEB;
                        border:1px solid #FCD34D;
                        border-radius:10px;
                        padding:0.8rem 1rem;
                        font-size:0.82rem;
                        color:#92400E;
                    ">
                        ⚠️ Hasil ini <b>bukan diagnosis medis</b>.
                        Segera konsultasikan ke dokter kulit untuk pemeriksaan lebih lanjut.
                    </div>
                """, unsafe_allow_html=True)

# =========================================================
# MENU: INFORMASI
# =========================================================
elif "Informasi" in menu:

    st.markdown("## Informasi Penyakit Kulit")
    st.markdown(
        "<p style='color:#6B7280;font-size:0.95rem;margin-top:-0.5rem;margin-bottom:1.75rem'>"
        "Penjelasan singkat setiap kondisi kulit yang dapat dikenali oleh model.</p>",
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2, gap="medium")
    cols = [col_a, col_b]

    for idx, (cls, info) in enumerate(DISEASE_INFO.items()):

        gejala_html = "".join([
            f"""<div style="display:flex;gap:0.5rem;align-items:flex-start;padding:0.25rem 0;font-size:0.85rem;color:#374151">
                    <div style="width:7px;height:7px;border-radius:50%;background:{info['color']};margin-top:0.38rem;flex-shrink:0"></div>
                    <span>{g}</span>
                </div>"""
            for g in info["gejala"]
        ])

        jurnal_html = ""
        if info.get("jurnal"):
            jurnal_html = f"""
                <div style="margin-top:1rem">
                    <a href="{info['jurnal']}" target="_blank"
                       style="font-size:0.82rem;color:{info['color']};font-weight:600;text-decoration:none">
                        📄 Lihat Jurnal Referensi →
                    </a>
                </div>"""

        with cols[idx % 2]:
            st.markdown(f"""
                <div style="
                    background:{info['bg']};
                    border:1.5px solid {info['border']};
                    border-radius:16px;
                    padding:1.5rem;
                    margin-bottom:1.25rem;
                ">
                    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.9rem;padding-bottom:0.75rem;border-bottom:1px solid {info['border']}">
                        <span style="font-size:1.6rem">{info['icon']}</span>
                        <span style="font-size:1.15rem;font-weight:700;color:{info['color']};letter-spacing:-0.01em">{cls}</span>
                    </div>

                    <div style="font-size:0.87rem;color:#4B5563;line-height:1.7;margin-bottom:1rem">
                        {info['deskripsi']}
                    </div>

                    <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#8896AB;margin-bottom:0.4rem">
                        Gejala Umum
                    </div>
                    {gejala_html}

                    <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#8896AB;margin-top:0.9rem;margin-bottom:0.4rem">
                        Penanganan
                    </div>
                    <div style="background:rgba(0,0,0,0.03);border-radius:8px;padding:0.7rem 0.9rem;font-size:0.85rem;color:#4B5563;line-height:1.65">
                        {info['penanganan']}
                    </div>
                    {jurnal_html}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("""
        <div style="
            background:#FFFBEB;border:1px solid #FCD34D;
            border-radius:10px;padding:0.8rem 1rem;
            font-size:0.82rem;color:#92400E;margin-top:0.5rem
        ">
            ⚠️ Informasi di atas bersifat <b>edukatif</b>.
            Selalu konsultasikan kondisi kulit Anda kepada tenaga medis profesional.
        </div>
    """, unsafe_allow_html=True)
