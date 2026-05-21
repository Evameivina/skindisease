import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
MODEL_PATH = "convnext_skin_state_dict.pth"
CLASSES    = ["Eczema", "Herpes Zoster", "Normal", "Ringworm"]
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASE_INFO = {
    "Eczema": {
        "icon": "🔴", "color": "#E05C5C", "bg": "#FFF0F0",
        "deskripsi": "Kondisi kulit kronis yang menyebabkan peradangan, kemerahan, dan rasa gatal. Sering dipicu oleh alergen, stres, atau perubahan cuaca.",
        "gejala": ["Kulit kering dan gatal", "Kemerahan dan peradangan", "Kulit bersisik atau mengelupas", "Bentol-bentol kecil berisi cairan"],
        "penanganan": "Gunakan pelembap secara rutin, hindari pemicu alergi, dan konsultasikan ke dokter untuk krim kortikosteroid atau antihistamin.",
    },
    "Herpes Zoster": {
        "icon": "🟠", "color": "#E07A2F", "bg": "#FFF5EC",
        "deskripsi": "Infeksi virus akibat reaktivasi virus varisela-zoster. Ditandai ruam melepuh yang nyeri pada satu sisi tubuh.",
        "gejala": ["Nyeri, terbakar, atau kesemutan", "Sensitif terhadap sentuhan", "Ruam merah beberapa hari setelah nyeri", "Lepuhan berisi cairan yang pecah dan mengering"],
        "penanganan": "Segera konsultasi ke dokter untuk obat antivirus. Penanganan dini mengurangi keparahan dan durasi penyakit.",
    },
    "Normal": {
        "icon": "🟢", "color": "#2E9E6B", "bg": "#EDFAF3",
        "deskripsi": "Kulit terdeteksi dalam kondisi normal tanpa tanda-tanda penyakit kulit yang signifikan.",
        "gejala": ["Tidak ditemukan tanda penyakit kulit"],
        "penanganan": "Jaga kesehatan kulit dengan rutin membersihkan, melembapkan, dan melindungi dari paparan sinar matahari.",
    },
    "Ringworm": {
        "icon": "🟣", "color": "#7B61D4", "bg": "#F4F1FF",
        "deskripsi": "Infeksi jamur yang membentuk pola melingkar di kulit. Disebabkan oleh jamur, bukan cacing.",
        "gejala": ["Ruam melingkar berwarna merah", "Tepi ruam lebih menonjol", "Gatal pada area terinfeksi", "Kulit bersisik di dalam lingkaran"],
        "penanganan": "Gunakan krim antijamur dari apotek. Jaga kebersihan, keringkan kulit, hindari berbagi handuk atau pakaian.",
    },
}

# ── Model ─────────────────────────────────────────────────────────────────────
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
    state = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict(state)
    m.eval()
    return m.to(device)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = load_model()(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return CLASSES[probs.argmax()], probs

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinScan — Deteksi Penyakit Kulit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

section[data-testid="stSidebar"] {
    background: #0F1117;
    border-right: 1px solid #1E2130;
}
section[data-testid="stSidebar"] * { color: #E0E4F0 !important; }
section[data-testid="stSidebar"] .stSelectbox label {
    color: #5A6080 !important;
    font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase;
}
.sidebar-brand {
    text-align: center; padding: 1.5rem 0 2rem;
    border-bottom: 1px solid #1E2130; margin-bottom: 1.5rem;
}
.sidebar-brand .s-icon { font-size: 2.5rem; display: block; }
.sidebar-brand .s-name {
    font-family: 'DM Serif Display', serif; font-size: 1.6rem;
    color: #fff !important; letter-spacing: -0.02em; display: block; margin-top: 0.3rem;
}
.sidebar-brand .s-sub {
    font-size: 0.7rem; color: #5A6080 !important;
    letter-spacing: 0.07em; text-transform: uppercase;
}
.chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: #1E2130; border-radius: 99px;
    padding: 0.3rem 0.75rem; font-size: 0.78rem;
    color: #A0A8C0 !important; margin: 0.2rem 0.1rem;
}
.page-hd h1 {
    font-family: 'DM Serif Display', serif; font-size: 2rem;
    color: #111827; letter-spacing: -0.03em; margin: 0 0 0.35rem;
}
.page-hd p { color: #6B7280; font-size: 0.92rem; margin: 0 0 1.75rem; }
.upload-hint {
    border: 2px dashed #D1D5DB; border-radius: 16px;
    padding: 3rem 2rem; text-align: center; background: #FAFAFA;
}
.upload-hint .uh-icon { font-size: 2.5rem; display: block; margin-bottom: 0.6rem; }
.upload-hint h3 { font-size: 1rem; font-weight: 600; color: #374151; margin: 0 0 0.25rem; }
.upload-hint p { font-size: 0.82rem; color: #9CA3AF; margin: 0; }
.res-card {
    border-radius: 14px; padding: 1.5rem;
    border: 1.5px solid; margin-bottom: 1.25rem;
}
.res-lbl {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #9CA3AF; margin-bottom: 0.3rem;
}
.res-name { font-family: 'DM Serif Display', serif; font-size: 1.75rem; letter-spacing: -0.02em; margin: 0 0 0.15rem; }
.res-conf { font-size: 0.85rem; color: #6B7280; }
.res-conf strong { font-size: 1.4rem; font-weight: 700; }
.prob-lbl { font-size: 0.82rem; font-weight: 500; color: #374151; display: flex; justify-content: space-between; margin-bottom: 0.2rem; }
.prob-bg { height: 7px; border-radius: 99px; background: #F3F4F6; overflow: hidden; margin-bottom: 0.65rem; }
.prob-fill { height: 7px; border-radius: 99px; }
.disc {
    background: #FFFBEB; border: 1px solid #FCD34D;
    border-radius: 10px; padding: 0.8rem 1rem;
    font-size: 0.82rem; color: #92400E; margin-top: 1.25rem;
}
.icard { border-radius: 14px; padding: 1.5rem; border: 1.5px solid; margin-bottom: 1.25rem; }
.icard-hd {
    display: flex; align-items: center; gap: 0.65rem;
    padding-bottom: 0.75rem; margin-bottom: 0.9rem;
    border-bottom: 1px solid rgba(0,0,0,0.06);
}
.icard-icon { font-size: 1.6rem; }
.icard-name { font-family: 'DM Serif Display', serif; font-size: 1.2rem; letter-spacing: -0.02em; }
.icard-desc { font-size: 0.86rem; color: #4B5563; line-height: 1.65; margin-bottom: 0.85rem; }
.sec-ttl { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #9CA3AF; margin-bottom: 0.45rem; }
.gitem { font-size: 0.84rem; color: #374151; display: flex; gap: 0.45rem; align-items: flex-start; padding: 0.28rem 0; }
.gdot { width: 6px; height: 6px; border-radius: 50%; margin-top: 0.42rem; flex-shrink: 0; }
.pbox { background: rgba(0,0,0,0.03); border-radius: 8px; padding: 0.7rem 0.9rem; font-size: 0.84rem; color: #4B5563; line-height: 1.65; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="s-icon">🔬</span>
        <span class="s-name">SkinScan</span>
        <span class="s-sub">Skin Disease Detection</span>
    </div>
    """, unsafe_allow_html=True)

    menu = st.selectbox("MENU", ["🩺  Deteksi Penyakit Kulit", "📖  Informasi Penyakit"])

    st.markdown("---")
    st.markdown("""
    <div style="padding:0.25rem 0">
        <div style="font-size:0.68rem;color:#5A6080;letter-spacing:0.07em;text-transform:uppercase;margin-bottom:0.6rem">Model Info</div>
        <span class="chip">🧠 ConvNeXt-Tiny</span>
        <span class="chip">📊 4 Kelas</span>
        <span class="chip">✅ 99.57%</span>
    </div>
    """, unsafe_allow_html=True)

# ── Deteksi ───────────────────────────────────────────────────────────────────
if "Deteksi" in menu:
    st.markdown("""
    <div class="page-hd">
        <h1>Deteksi Penyakit Kulit</h1>
        <p>Unggah foto kulit untuk mendapatkan prediksi kondisi beserta confidence score dari model.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if not uploaded:
        st.markdown("""
        <div class="upload-hint">
            <span class="uh-icon">🖼️</span>
            <h3>Seret & lepas gambar di sini</h3>
            <p>atau klik Browse files di atas &nbsp;·&nbsp; JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded).convert("RGB")
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.image(image, use_container_width=True)
        with c2:
            with st.spinner("Menganalisis gambar..."):
                try:
                    label, probs = predict(image)
                    info = DISEASE_INFO[label]
                    conf = float(probs.max()) * 100

                    st.markdown(f"""
                    <div class="res-card" style="border-color:{info['color']};background:{info['bg']}">
                        <div class="res-lbl">Hasil Prediksi</div>
                        <div class="res-name" style="color:{info['color']}">{info['icon']} {label}</div>
                        <div class="res-conf">Confidence: <strong style="color:{info['color']}">{conf:.1f}%</strong></div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div class='res-lbl' style='margin-bottom:0.75rem'>Probabilitas per Kelas</div>", unsafe_allow_html=True)
                    for i, cls in enumerate(CLASSES):
                        pct = float(probs[i]) * 100
                        c   = DISEASE_INFO[cls]["color"]
                        st.markdown(f"""
                        <div class="prob-lbl"><span>{DISEASE_INFO[cls]['icon']} {cls}</span><span>{pct:.1f}%</span></div>
                        <div class="prob-bg"><div class="prob-fill" style="width:{pct}%;background:{c}"></div></div>
                        """, unsafe_allow_html=True)

                    if label != "Normal":
                        st.markdown("""
                        <div class="disc">⚠️ Hasil ini <strong>bukan diagnosis medis</strong>.
                        Segera konsultasikan ke dokter kulit untuk pemeriksaan lebih lanjut.</div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Gagal memuat model: {e}")
                    st.caption(f"Pastikan `{MODEL_PATH}` berada di folder yang sama dengan `app.py`.")

# ── Informasi ─────────────────────────────────────────────────────────────────
elif "Informasi" in menu:
    st.markdown("""
    <div class="page-hd">
        <h1>Informasi Penyakit Kulit</h1>
        <p>Penjelasan singkat setiap kondisi kulit yang dapat dikenali oleh model.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="medium")
    cols = [col_a, col_b]
    for idx, (cls, info) in enumerate(DISEASE_INFO.items()):
        gejala_html = "".join([
            f'<div class="gitem"><div class="gdot" style="background:{info["color"]}"></div><span>{g}</span></div>'
            for g in info["gejala"]
        ])
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="icard" style="border-color:{info['color']}30;background:{info['bg']}">
                <div class="icard-hd">
                    <span class="icard-icon">{info['icon']}</span>
                    <span class="icard-name" style="color:{info['color']}">{cls}</span>
                </div>
                <div class="icard-desc">{info['deskripsi']}</div>
                <div class="sec-ttl">Gejala Umum</div>
                {gejala_html}
                <div class="sec-ttl" style="margin-top:0.9rem">Penanganan</div>
                <div class="pbox">{info['penanganan']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disc">
        ⚠️ Informasi di atas bersifat <strong>edukatif</strong>.
        Selalu konsultasikan kondisi kulit Anda kepada tenaga medis profesional.
    </div>
    """, unsafe_allow_html=True)
