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
IMG_SIZE   = 224
MODEL_PATH = "convnext_skin_state_dict.pth"
CLASSES    = ["Eczema", "Herpes Zoster", "Normal", "Ringworm"]

DISEASE_INFO = {
    "Eczema": {
        "icon"       : "🔴",
        "color"      : "#E05C5C",
        "deskripsi"  : (
            "Atopic dermatitis (AD), atau atopic eczema, adalah penyakit kulit inflamasi kronis yang bersifat heterogen secara fenotipik. "
            "Penyakit ini umumnya muncul akibat pemicu lingkungan pada individu yang memiliki predisposisi genetik. "
            "AD ditandai dengan pruritus — terutama memburuk pada malam hari — kulit kering dan menebal, serta papul yang terasa sangat gatal dan dapat mengeluarkan cairan jika digaruk. "
            "Xerosis (kekeringan kulit) dan iktiosis juga merupakan istilah yang sering dikaitkan dengan AD. "
            "Prevalensi AD sekitar 20% pada anak-anak dan 1–3% pada dewasa. "
            "AD menempati peringkat ke-15 dalam studi beban penyakit global (1990–2017) di antara semua penyakit non-fatal, dan berada di peringkat pertama di antara penyakit kulit berdasarkan disability-adjusted life-years (DALYs)."
        ),
        "gejala"     : [
            "Pruritus (gatal intens), terutama memburuk pada malam hari",
            "Kulit kering, menebal, dan bersisik (xerosis/iktiosis)",
            "Papul kemerahan yang dapat mengeluarkan cairan jika digaruk",
            "Lesi pada area fleksural (siku, lutut) atau wajah pada bayi",
            "Bersifat kronis dengan periode remisi dan eksaserbasi (flare)",
        ],
        "sumber"     : "Afshari et al., Front. Immunol. 2024",
        "link"       : "https://pmc.ncbi.nlm.nih.gov/articles/PMC10944924/",
    },
    "Herpes Zoster": {
        "icon"       : "🟡",
        "color"      : "#D4A017",
        "deskripsi"  : (
            "Herpes zoster (HZ) merupakan reaktivasi virus Varicella-Zoster (VZV) — agen kausal yang sama dengan cacar air (varisela). "
            "VZV bersifat laten di jaringan saraf setelah infeksi primer, dan reaktivasinya umumnya dipicu oleh penurunan imunitas seluler akibat usia lanjut, stres, infeksi lain, atau imunosupresi. "
            "Gejala klinis muncul dalam tiga tahap: fase pre-eruptif (nyeri atau rasa terbakar dalam dermaton yang terkena, minimal 2 hari sebelum ruam), fase eruptif akut (vesikel yang menyakitkan, berlangsung 2–4 minggu), dan fase kronis yang ditandai nyeri persisten lebih dari 4 minggu (postherpetic neuralgia/PHN). "
            "Insiden HZ meningkat seiring usia, berkisar 3,9–11,8 per 1.000 orang per tahun pada usia di atas 65 tahun."
        ),
        "gejala"     : [
            "Fase awal: nyeri, rasa terbakar, atau kesemutan pada satu sisi tubuh",
            "Vesikel multipel yang terasa nyeri mengikuti jalur dermaton",
            "Gejala prodromal: sakit kepala, malaise umum, dan fotofobia",
            "Nyeri yang dapat berlanjut setelah ruam sembuh (postherpetic neuralgia)",
        ],
        "sumber"     : "Patil et al., Viruses 2022",
        "link"       : "https://pmc.ncbi.nlm.nih.gov/articles/PMC8876683/",
    },
    "Normal": {
        "icon"       : "🟢",
        "color"      : "#1E8A5E",
        "deskripsi"  : (
            "Kulit merupakan organ terbesar tubuh manusia yang tersusun dari tiga lapisan utama: epidermis, dermis, dan hipodermis. "
            "Epidermis umumnya terdiri dari sekitar 40–50 lapisan sel epitel skuamosa yang terutama berasal dari keratinosit, dan terbagi menjadi empat lapisan: stratum basale, stratum spinosum, stratum granulosum, dan stratum corneum. "
            "Lapisan dermis yang terletak di bawah epidermis sebagian besar tersusun dari jaringan ikat yang mengandung serat kolagen dan elastin, sementara hipodermis merupakan area yang terdiri dari jaringan adiposa. "
            "Pada kondisi normal, kulit berfungsi optimal sebagai pelindung pertama tubuh terhadap patogen, radiasi UV, bahan kimia, dan cedera mekanis, sekaligus berperan dalam regulasi suhu tubuh."
        ),
        "gejala"     : [
            "Tidak ditemukan tanda-tanda kelainan atau penyakit kulit",
            "Fungsi barrier kulit optimal — mencegah kehilangan air dan invasi patogen",
            "Warna dan tekstur kulit merata, terhidrasi dengan baik",
            "Tidak ada lesi, kemerahan, bersisik, atau iritasi",
        ],
        "sumber"     : "Brito et al., Pharmaceutics 2024",
        "link"       : "https://pmc.ncbi.nlm.nih.gov/articles/PMC11597055/",
    },
    "Ringworm": {
        "icon"       : "🟠",
        "color"      : "#D4721A",
        "deskripsi"  : (
            "Tinea corporis, yang dikenal sebagai ringworm, adalah infeksi jamur superfisial pada kulit yang disebabkan oleh dermatofita. "
            "Trichophyton rubrum merupakan spesies dermatofita paling umum sebagai penyebabnya. "
            "Infeksi ini dapat terjadi melalui kontak dengan orang atau hewan yang terinfeksi, maupun melalui fomites seperti sisir, pakaian, handuk, dan alas lantai. "
            "Tinea corporis umumnya tampil sebagai papul dan plak anular (berbentuk cincin) yang terbatas jelas, terasa gatal, dengan hipopigmentasi sentral — tampilan inilah yang melahirkan nama 'ringworm'. "
            "Tinea infection merupakan kondisi kulit paling prevalen di dunia dan berada di peringkat keempat tertinggi dalam insiden penyakit pada tahun 2016, dengan estimasi risiko seumur hidup sebesar 10–20%."
        ),
        "gejala"     : [
            "Papul dan plak anular (cincin) berbatas jelas dengan tepi lebih menonjol",
            "Hipopigmentasi atau clearing pada bagian tengah lesi",
            "Skuama (sisik) pada tepi lesi yang aktif",
            "Rasa gatal (pruritus) pada area yang terinfeksi",
            "Pada kulit gelap, lesi dapat tampak violaseus atau hiperpigmentasi",
        ],
        "sumber"     : "Van Alfen et al., HCA Healthcare J Med 2026",
        "link"       : "https://pmc.ncbi.nlm.nih.gov/articles/PMC12971098/",
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    .stApp { background: #0D1117; color: #E6EDF3; }

    [data-testid="stSidebar"] {
        background: #161B22;
        border-right: 1px solid #30363D;
    }
    [data-testid="stSidebar"] * { color: #E6EDF3 !important; }

    h1, h2, h3 { font-family: 'Sora', sans-serif !important; font-weight: 700 !important; }

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
        font-size: 0.8rem;
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

    .source-tag {
        display: inline-block;
        margin-top: 0.6rem;
        font-size: 0.75rem;
        font-family: 'DM Mono', monospace;
        color: #6E7681;
        background: #21262D;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        text-decoration: none;
    }

    .source-tag:hover {
        color: #58A6FF;
        border-color: #58A6FF;
        text-decoration: none;
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
    .score-chip.top { border-color: #58A6FF; color: #58A6FF; }
    .score-chip .score-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        display: block;
        margin-bottom: 0.2rem;
        color: #6E7681;
    }
    .score-chip.top .score-label { color: #79C0FF; }
    .score-chip .score-val { font-size: 1rem; font-weight: 500; }

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
    .disclaimer strong { color: #F0883E; }

    [data-testid="stFileUploader"] {
        background: #161B22;
        border: 2px dashed #30363D;
        border-radius: 12px;
    }
    [data-testid="stFileUploader"]:hover { border-color: #58A6FF; }

    .stButton > button {
        background: linear-gradient(135deg, #1F6FEB, #58A6FF) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    [data-testid="stMetric"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #E6EDF3 !important; font-family: 'DM Mono', monospace !important; }

    #MainMenu, footer, header { display: none !important; }
    .block-container { padding-top: 2rem !important; max-width: 1100px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
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
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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
    return CLASSES[int(np.argmax(probs))], probs

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding:1.75rem 0 1.5rem; text-align:center;">
            <div style="font-size:1.5rem; font-weight:700; color:#FFFFFF; letter-spacing:-0.02em;">🔬 DermaScan</div>
            <div style="font-size:0.7rem; color:#555870; letter-spacing:0.07em; text-transform:uppercase; margin-top:0.3rem;">Skin Disease Detection</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#30363D; margin:0 0 1rem 0'>", unsafe_allow_html=True)

    menu = st.radio(
        "MENU",
        ["🩺 Deteksi Penyakit Kulit", "📚 Informasi Penyakit"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#30363D; margin:1rem 0'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='color:#6E7681; font-size:0.78rem; line-height:1.7;'>
        <b style='color:#8B949E;'>Kelas yang Didukung</b><br>
        • Eczema<br>
        • Herpes Zoster<br>
        • Normal<br>
        • Ringworm
    </div>
    <br>
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
            "Pilih gambar kulit (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Lebar", f"{image.width} px")
            with c2:
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
                    model_inst   = load_model()
                    pred_class, probs = predict(image, model_inst)
                    info         = DISEASE_INFO[pred_class]
                    top_conf     = float(probs[CLASSES.index(pred_class)]) * 100
                    color        = info["color"]

                    # Result box
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Hasil Klasifikasi</div>
                        <div class="result-class" style="color:{color};">{info['icon']} {pred_class}</div>
                        <div class="result-label" style="margin-top:0.75rem;">Confidence Score</div>
                        <div style="font-size:1.8rem; font-family:'DM Mono',monospace; font-weight:600; color:{color};">{top_conf:.1f}%</div>
                        <div class="confidence-bar-container">
                            <div style="height:100%; border-radius:100px; width:{top_conf:.1f}%;
                                        background:linear-gradient(90deg,{color}88,{color});"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # All class scores
                    chips_html = '<div class="all-scores-row">'
                    for i, cls in enumerate(CLASSES):
                        pct    = float(probs[i]) * 100
                        is_top = cls == pred_class
                        chips_html += f"""
                        <div class="score-chip {'top' if is_top else ''}">
                            <span class="score-label">{cls}</span>
                            <span class="score-val">{pct:.1f}%</span>
                        </div>"""
                    chips_html += '</div>'
                    st.markdown(chips_html, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Deskripsi
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>Deskripsi</h4>
                        <p style="color:#C9D1D9; font-size:0.88rem; line-height:1.7; margin:0;">
                            {info['deskripsi']}
                        </p>
                        <a href="{info['link']}" target="_blank" class="source-tag">📄 {info['sumber']} ↗</a>
                    </div>
                    """, unsafe_allow_html=True)

                    # Gejala
                    gejala_items = "".join([
                        f'<div class="symptom-item"><div class="symptom-dot" style="background:{color};"></div><span>{g}</span></div>'
                        for g in info["gejala"]
                    ])
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>Gejala Umum</h4>
                        {gejala_items}
                    </div>
                    """, unsafe_allow_html=True)

                    # Disclaimer
                    st.markdown("""
                    <div class="disclaimer">
                        <strong>⚠️ Disclaimer:</strong> Hasil ini merupakan output screening awal berbasis AI dan
                        <strong>tidak menggantikan diagnosis medis</strong> oleh tenaga kesehatan profesional.
                        Segera konsultasikan ke dokter untuk penanganan lebih lanjut.
                    </div>
                    """, unsafe_allow_html=True)

                except FileNotFoundError:
                    st.error(f"❌ File model tidak ditemukan: `{MODEL_PATH}`\n\nPastikan file berada di direktori yang sama dengan `app.py`.")
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
                <div style='font-size:0.9rem;'>Klik <strong style='color:#58A6FF;'>Analisis Sekarang</strong><br>untuk memulai klasifikasi</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: INFORMASI PENYAKIT
# ─────────────────────────────────────────────────────────────
elif "📚 Informasi" in menu:

    st.markdown('<p class="main-title">Informasi Penyakit Kulit</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Penjelasan singkat mengenai empat kondisi kulit yang dapat dideteksi oleh aplikasi ini</p>', unsafe_allow_html=True)

    for cls, info in DISEASE_INFO.items():
        color = info["color"]

        with st.expander(f"{info['icon']}  {cls}", expanded=(cls == "Normal")):
            st.markdown(f"""
            <p style="color:#C9D1D9; font-size:0.92rem; line-height:1.75; margin-bottom:0.75rem;">
                {info['deskripsi']}
            </p>
            <a href="{info['link']}" target="_blank" class="source-tag">📄 {info['sumber']} ↗</a>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{color}; font-weight:600; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; font-family:DM Mono,monospace; margin-bottom:0.25rem;'>Gejala Umum</p>", unsafe_allow_html=True)

            gejala_items = "".join([
                f'<div class="symptom-item"><div class="symptom-dot" style="background:{color};"></div><span style="font-size:0.88rem;">{g}</span></div>'
                for g in info["gejala"]
            ])
            st.markdown(gejala_items, unsafe_allow_html=True)
