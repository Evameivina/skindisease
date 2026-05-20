# app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ======================================
# CONFIG
# ======================================
IMG_SIZE = 224
MODEL_PATH = "convnext_skin_disease_finetuned.pth"

classes = [
    "Eczema",
    "Herpes Zoster",
    "Normal",
    "Ringworm"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================
# TRANSFORM
# ======================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ======================================
# LOAD MODEL
# ======================================
@st.cache_resource
def load_model():

    model = models.convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features

    # HARUS sama dengan model training
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(classes))
    )

    # load model
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    return model

model = load_model()

# ======================================
# STREAMLIT UI
# ======================================
st.set_page_config(
    page_title="Deteksi Penyakit Kulit",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Aplikasi Deteksi Penyakit Kulit")

st.write("""
Aplikasi klasifikasi penyakit kulit berbasis Deep Learning
menggunakan arsitektur ConvNeXt-Tiny.
""")

st.write("### Kelas Penyakit")
st.write("- Eczema")
st.write("- Herpes Zoster")
st.write("- Normal")
st.write("- Ringworm")

uploaded_file = st.file_uploader(
    "Upload gambar kulit",
    type=["jpg", "jpeg", "png"]
)

# ======================================
# PREDICTION
# ======================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    # preprocessing
    img_tensor = transform(image).unsqueeze(0).to(device)

    # prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    # hasil
    st.success(f"Prediksi Penyakit: {predicted_class}")
    st.info(f"Confidence Score: {confidence_score:.2f}%")

    st.write("### Probabilitas Setiap Kelas")

    for i, cls in enumerate(classes):
        st.write(
            f"{cls}: {probabilities[0][i].item() * 100:.2f}%"
        )

# ======================================
# DISCLAIMER
# ======================================
st.warning("""
Aplikasi ini hanya digunakan sebagai alat bantu screening awal
dan tidak menggantikan diagnosis medis profesional.
""")
