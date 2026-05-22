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

    st.markdown("<hr style='border-color:#2A2D3E;margin:0 0 1rem 0'>", unsafe_allow_html=True)

    menu = st.radio(
        "MENU",
        ["🩺 Deteksi", "📖 Informasi"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2A2D3E;margin:1rem 0'>", unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size:0.72rem;color:#555870;line-height:1.7;padding:0 0.25rem">
            <b style="color:#8896AB">⚠️ Disclaimer</b><br>
            Aplikasi ini hanya untuk tujuan edukatif. Bukan pengganti diagnosis medis profesional.
        </div>
    """, unsafe_allow_html=True)


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

    # Tutorial & kelas yang didukung
    with st.expander("ℹ️ Cara Penggunaan & Kelas yang Didukung", expanded=False):
        col_t1, col_t2 = st.columns(2, gap="medium")
        with col_t1:
            st.markdown("""
                <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;
                text-transform:uppercase;color:#8896AB;margin-bottom:0.6rem">
                    🎯 Kelas yang Dapat Dideteksi
                </div>
                <div style="font-size:0.87rem;color:#374151;line-height:2">
                    🔴 &nbsp;<b>Eczema</b> — eksim / dermatitis<br>
                    🟠 &nbsp;<b>Herpes Zoster</b> — cacar api / shingles<br>
                    🟢 &nbsp;<b>Normal</b> — kulit sehat<br>
                    🟣 &nbsp;<b>Ringworm</b> — kurap / tinea corporis
                </div>
                <div style="margin-top:0.75rem;background:#FFF0F0;border-radius:8px;
                padding:0.6rem 0.8rem;font-size:0.82rem;color:#E05C5C">
                    ⚠️ Gambar di luar 4 kategori ini tetap akan diprediksi ke salah satu kelas,
                    namun hasilnya tidak dapat diandalkan.
                </div>
            """, unsafe_allow_html=True)
        with col_t2:
            st.markdown("""
                <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;
                text-transform:uppercase;color:#8896AB;margin-bottom:0.6rem">
                    📋 Tutorial Cara Pakai
                </div>
                <div style="font-size:0.87rem;color:#374151;line-height:2">
                    <b>1.</b> &nbsp;Siapkan foto kulit yang jelas dan cukup cahaya<br>
                    <b>2.</b> &nbsp;Pastikan area yang bermasalah terlihat jelas di foto<br>
                    <b>3.</b> &nbsp;Klik <b>Browse files</b> atau seret gambar ke area upload<br>
                    <b>4.</b> &nbsp;Tunggu hingga model selesai menganalisis<br>
                    <b>5.</b> &nbsp;Baca hasil prediksi dan confidence score<br>
                    <b>6.</b> &nbsp;Jika terdeteksi penyakit, konsultasikan ke dokter kulit
                </div>
                <div style="margin-top:0.75rem;background:#EFF6FF;border-radius:8px;
                padding:0.6rem 0.8rem;font-size:0.82rem;color:#1E40AF">
                    💡 Tips: Gunakan foto resolusi tinggi dengan latar belakang polos
                    untuk hasil terbaik.
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:0.75rem'></div>", unsafe_allow_html=True)

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
                <div style="font-size:1rem;font-weight:600;color:#374151;margin-bottom:0.3rem">Seret &amp; lepas gambar di sini</div>
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

            # Confidence bar semua kelas
            st.markdown(
                "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;"
                "color:#8896AB;margin:1rem 0 0.6rem'>Distribusi Confidence</div>",
                unsafe_allow_html=True
            )
            for i, cls in enumerate(CLASSES):
                cls_info  = DISEASE_INFO[cls]
                cls_conf  = float(probs[i]) * 100
                bar_width = max(cls_conf, 1)
                st.markdown(f"""
                    <div style="margin-bottom:0.55rem">
                        <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:0.2rem">
                            <span style="color:#374151;font-weight:500">{cls_info['icon']} {cls}</span>
                            <span style="color:{cls_info['color']};font-weight:600">{cls_conf:.1f}%</span>
                        </div>
                        <div style="background:#E5E9F0;border-radius:99px;height:7px">
                            <div style="width:{bar_width}%;background:{cls_info['color']};height:7px;border-radius:99px;transition:width 0.4s"></div>
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

    st.markdown("## 📖 Informasi Penyakit Kulit")
    st.markdown(
        "<p style='color:#6B7280;font-size:0.95rem;margin-top:-0.5rem;margin-bottom:1.5rem'>"
        "Pilih kondisi kulit di bawah untuk melihat penjelasan lengkapnya.</p>",
        unsafe_allow_html=True
    )

    # Tombol pilih penyakit
    btn_cols = st.columns(4, gap="small")
    selected = st.session_state.get("selected_info", None)

    for idx, (cls, info) in enumerate(DISEASE_INFO.items()):
        with btn_cols[idx]:
            is_active = selected == cls
            active_style = f"background:{info['color']};color:white;border-color:{info['color']};"
            inactive_style = f"background:{info['bg']};color:{info['color']};border-color:{info['border']};"
            st.markdown(f"""
                <div style="
                    {active_style if is_active else inactive_style}
                    border:1.5px solid;
                    border-radius:12px;
                    padding:0.75rem 0.5rem;
                    text-align:center;
                    cursor:pointer;
                    margin-bottom:0.25rem;
                ">
                    <div style="font-size:1.6rem">{info['icon']}</div>
                    <div style="font-size:0.82rem;font-weight:600;margin-top:0.25rem">{cls}</div>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"Pilih {cls}", key=f"btn_{cls}", use_container_width=True):
                st.session_state["selected_info"] = cls
                st.rerun()

    st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

    # Tampilkan detail jika sudah dipilih
    selected = st.session_state.get("selected_info", None)

    if selected is None:
        st.markdown("""
            <div style="
                border: 2px dashed #CBD5E1;
                border-radius: 16px;
                background: white;
                padding: 3rem 2rem;
                text-align: center;
            ">
                <div style="font-size:2.5rem;margin-bottom:0.75rem">👆</div>
                <div style="font-size:1rem;font-weight:600;color:#374151;margin-bottom:0.3rem">Pilih salah satu kondisi di atas</div>
                <div style="font-size:0.83rem;color:#9CA3AF">Informasi lengkap akan ditampilkan di sini</div>
            </div>
        """, unsafe_allow_html=True)

    else:
        info = DISEASE_INFO[selected]

        gejala_html = "".join([
            f"""<div style="display:flex;gap:0.5rem;align-items:flex-start;padding:0.3rem 0;font-size:0.87rem;color:#374151">
                    <div style="width:7px;height:7px;border-radius:50%;background:{info['color']};margin-top:0.42rem;flex-shrink:0"></div>
                    <span>{g}</span>
                </div>"""
            for g in info["gejala"]
        ])

        jurnal_html = ""
        if info.get("jurnal"):
            jurnal_html = f"""
                <div style="margin-top:1.25rem">
                    <a href="{info['jurnal']}" target="_blank"
                       style="font-size:0.83rem;color:{info['color']};font-weight:600;text-decoration:none">
                        📄 Lihat Jurnal Referensi →
                    </a>
                </div>"""

        _, center_info, _ = st.columns([1, 3, 1])
        with center_info:
            st.markdown(f"""
                <div style="
                    background:{info['bg']};
                    border:1.5px solid {info['border']};
                    border-radius:16px;
                    padding:1.75rem;
                    margin-bottom:1.25rem;
                ">
                    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;
                                padding-bottom:1rem;border-bottom:1.5px solid {info['border']}">
                        <span style="font-size:2rem">{info['icon']}</span>
                        <span style="font-size:1.4rem;font-weight:700;color:{info['color']};
                                     letter-spacing:-0.02em">{selected}</span>
                    </div>

                    <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.09em;
                                text-transform:uppercase;color:#8896AB;margin-bottom:0.5rem">
                        Deskripsi
                    </div>
                    <div style="font-size:0.88rem;color:#374151;line-height:1.75;margin-bottom:1.25rem">
                        {info['deskripsi']}
                    </div>

                    <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.09em;
                                text-transform:uppercase;color:#8896AB;margin-bottom:0.5rem">
                        Gejala Umum
                    </div>
                    {gejala_html}

                    <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.09em;
                                text-transform:uppercase;color:#8896AB;margin-top:1.1rem;margin-bottom:0.5rem">
                        Penanganan
                    </div>
                    <div style="background:rgba(0,0,0,0.04);border-radius:10px;padding:0.85rem 1rem;
                                font-size:0.87rem;color:#4B5563;line-height:1.7">
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
