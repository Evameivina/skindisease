import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
CLASSES = ['ECZEMA', 'HERPES ZOSTER', 'NORMAL', 'RINGWORM']
MODEL_PATH = 'convnext_skin_disease_finetuned.pth'
NUM_CLASSES = len(CLASSES)

DISEASE_INFO = {
    'ECZEMA': {
        'nama': 'Eksim (Eczema)',
        'deskripsi': 'Eksim adalah kondisi kulit kronis yang menyebabkan peradangan, kemerahan, dan rasa gatal. Penyakit ini sering kambuh dan dapat dipicu oleh alergen, stres, atau perubahan cuaca.',
        'gejala': ['Kulit kering dan gatal', 'Kemerahan dan peradangan', 'Kulit bersisik atau mengelupas', 'Bentol-bentol kecil berisi cairan'],
        'penanganan': 'Gunakan pelembap secara rutin, hindari pemicu alergi, dan konsultasikan ke dokter untuk mendapatkan krim kortikosteroid atau antihistamin jika diperlukan.',
        'warna': '#FF6B6B'
    },
    'HERPES ZOSTER': {
        'nama': 'Herpes Zoster (Cacar Api)',
        'deskripsi': 'Herpes zoster adalah infeksi virus yang disebabkan oleh reaktivasi virus varisela-zoster (penyebab cacar air). Penyakit ini ditandai dengan ruam melepuh yang terasa nyeri pada satu sisi tubuh.',
        'gejala': ['Nyeri, terbakar, atau kesemutan', 'Sensitif terhadap sentuhan', 'Ruam merah yang muncul beberapa hari setelah nyeri', 'Lepuhan berisi cairan yang pecah dan mengering'],
        'penanganan': 'Segera konsultasi ke dokter untuk mendapatkan obat antivirus. Penanganan dini dapat mengurangi keparahan dan durasi penyakit.',
        'warna': '#FF9F43'
    },
    'NORMAL': {
        'nama': 'Kulit Normal',
        'deskripsi': 'Kulit terdeteksi dalam kondisi normal tanpa tanda-tanda penyakit kulit yang signifikan.',
        'gejala': ['Tidak ada gejala penyakit kulit yang terdeteksi'],
        'penanganan': 'Jaga kesehatan kulit dengan rutin membersihkan, melembapkan, dan melindungi kulit dari paparan sinar matahari berlebih.',
        'warna': '#26de81'
    },
    'RINGWORM': {
        'nama': 'Kurap (Ringworm)',
        'deskripsi': 'Kurap adalah infeksi jamur pada kulit yang membentuk pola melingkar berwarna merah. Meskipun namanya mengandung kata "worm", penyakit ini disebabkan oleh jamur, bukan cacing.',
        'gejala': ['Ruam melingkar berwarna merah', 'Bagian tepi ruam lebih menonjol', 'Gatal pada area yang terinfeksi', 'Kulit bersisik di dalam lingkaran'],
        'penanganan': 'Gunakan krim antijamur yang tersedia di apotek. Jaga kebersihan dan keringkan kulit dengan baik. Hindari berbagi handuk atau pakaian dengan orang lain.',
        'warna': '#A29BFE'
    }
}

# ── Model ────────────────────────────────────────────────────────────────────
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
        nn.Linear(256, NUM_CLASSES)
    )
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image: Image.Image, model):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    pred_idx = probs.argmax().item()
    return CLASSES[pred_idx], probs.numpy()

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Deteksi Penyakit Kulit',
    page_icon='🔬',
    layout='wide'
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3436;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #636e72;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: #f8f9fa;
        border-left: 5px solid;
        margin-top: 1rem;
    }
    .info-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: #f8f9fa;
        border-left: 5px solid;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔬 Aplikasi Deteksi Penyakit Kulit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Berbasis Deep Learning — ConvNeXt-Tiny | 4 Kelas: Eczema, Herpes Zoster, Normal, Ringworm</div>', unsafe_allow_html=True)

menu = st.sidebar.selectbox('📋 Menu', ['Deteksi Penyakit Kulit', 'Informasi Penyakit'])

# ── Menu 1: Deteksi ──────────────────────────────────────────────────────────
if menu == 'Deteksi Penyakit Kulit':
    st.subheader('📤 Unggah Citra Kulit')
    st.caption('Unggah foto kulit yang ingin dideteksi. Model akan memprediksi kondisi kulit beserta tingkat kepercayaan (confidence score).')

    uploaded = st.file_uploader('Pilih gambar...', type=['jpg', 'jpeg', 'png'])

    if uploaded:
        col1, col2 = st.columns([1, 1], gap='large')

        with col1:
            image = Image.open(uploaded).convert('RGB')
            st.image(image, caption='Gambar yang diunggah', use_container_width=True)

        with col2:
            with st.spinner('Menganalisis gambar...'):
                try:
                    model = load_model()
                    label, probs = predict(image, model)
                    info = DISEASE_INFO[label]
                    confidence = float(probs.max()) * 100

                    st.markdown(f"""
                    <div class="result-box" style="border-color:{info['warna']}">
                        <span class="badge" style="background:{info['warna']}">{info['nama']}</span>
                        <h3 style="margin:0.5rem 0">Hasil Prediksi</h3>
                        <p style="font-size:2rem;font-weight:700;color:{info['warna']};margin:0">{confidence:.2f}%</p>
                        <p style="color:#636e72;margin:0">Confidence Score</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('#### Confidence per Kelas')
                    for i, cls in enumerate(CLASSES):
                        pct = float(probs[i]) * 100
                        color = DISEASE_INFO[cls]['warna']
                        st.markdown(f'**{cls}**')
                        st.progress(float(probs[i]), text=f'{pct:.2f}%')

                    if label != 'NORMAL':
                        st.warning('⚠️ Hasil ini bukan diagnosis medis. Segera konsultasikan ke dokter kulit untuk pemeriksaan lebih lanjut.')

                except Exception as e:
                    st.error(f'Gagal memuat model: {e}')
                    st.info(f'Pastikan file `{MODEL_PATH}` berada di direktori yang sama dengan `app.py`.')

# ── Menu 2: Informasi ────────────────────────────────────────────────────────
elif menu == 'Informasi Penyakit':
    st.subheader('📚 Informasi Penyakit Kulit')
    st.caption('Penjelasan singkat mengenai setiap kondisi kulit yang dapat dideteksi oleh aplikasi ini.')

    for cls, info in DISEASE_INFO.items():
        with st.expander(f"**{info['nama']}**", expanded=(cls == 'ECZEMA')):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                <div class="info-card" style="border-color:{info['warna']}">
                    <p>{info['deskripsi']}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('**Gejala Umum:**')
                for g in info['gejala']:
                    st.markdown(f'- {g}')
                st.markdown('**Penanganan:**')
                st.info(info['penanganan'])
            with col2:
                st.markdown(f"""
                <div style="background:{info['warna']}22;border-radius:12px;padding:1rem;text-align:center">
                    <span style="font-size:3rem">{'🔴' if cls=='ECZEMA' else '🟠' if cls=='HERPES ZOSTER' else '🟢' if cls=='NORMAL' else '🟣'}</span>
                    <p style="font-weight:600;color:{info['warna']};margin-top:0.5rem">{info['nama']}</p>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    st.caption('⚠️ Informasi di atas bersifat edukatif. Selalu konsultasikan kondisi kulit Anda kepada tenaga medis profesional.')
