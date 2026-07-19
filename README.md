# SkinScan — Skin Disease Classification using CNN Transfer Learning

Perbandingan performa tiga arsitektur *Convolutional Neural Network* (CNN) berbasis *transfer learning* — **MobileNetV3-Large**, **ResNet50**, dan **ConvNeXt-Tiny** — untuk klasifikasi citra penyakit kulit, serta implementasinya ke dalam aplikasi web **SkinScan** sebagai alat bantu *screening* awal.

> Skripsi — Program Studi Informatika, Fakultas Teknologi Industri, Universitas Gunadarma (2026)
> **Eva Meivina Dwiana** (50422472) — Pembimbing: Dr. Antonius Angga Kurniawan, ST., MMSI.

**Aplikasinya:** [skindiseaseproyek.streamlit.app](https://skindiseaseproyek.streamlit.app/)

---

## Ringkasan

Penyakit kulit menyumbang **4,60%–12,95%** dari seluruh kasus penyakit di Indonesia dan menempati peringkat ketiga penyakit terbanyak. Diagnosis manual rentan terhadap variasi hasil, terutama untuk penyakit dengan karakteristik visual yang mirip seperti **Ringworm**, **Herpes Zoster**, dan **Eczema**.

Proyek ini membandingkan tiga arsitektur CNN berbasis *transfer learning* untuk mengklasifikasikan citra kulit ke dalam **4 kelas**. Model terbaik (ConvNeXt-Tiny) kemudian diimplementasikan ke dalam aplikasi web **SkinScan** menggunakan **Streamlit**.

| Kelas | Deskripsi |
|---|---|
| **Eczema** | Kondisi peradangan kulit kronis dengan gejala kemerahan, gatal, dan kulit kering atau menebal. |
| **Herpes Zoster** | Infeksi akibat reaktivasi virus *Varicella-Zoster*, ditandai ruam kemerahan berisi lepuhan kecil yang nyeri dan biasanya memanjang mengikuti satu sisi tubuh. |
| **Normal** | Kondisi kulit sehat tanpa tanda kelainan. |
| **Ringworm** | Infeksi jamur pada permukaan kulit yang menimbulkan bercak kemerahan berbentuk melingkar dengan tepi lebih jelas dan sedikit menonjol dibanding bagian tengahnya. |

> ⚠️ Aplikasi hanya mendukung keempat kelas di atas. Jika citra yang diunggah berada di luar kategori ini, sistem tetap akan mengeluarkan salah satu dari empat prediksi tersebut, namun hasilnya **tidak valid/relevan**.

**Manfaat aplikasi:**
- Memberikan gambaran awal (*screening*) kepada masyarakat mengenai kemungkinan kondisi kulit yang dialami, khususnya untuk keempat kondisi di atas.
- Menjadi referensi bagi peneliti/akademisi dalam pengembangan sistem klasifikasi citra medis berbasis *deep learning*.
- Menunjukkan penerapan nyata hasil perbandingan arsitektur CNN berbasis *transfer learning* dalam bentuk aplikasi yang dapat digunakan langsung oleh pengguna.

## Hasil Utama

| Model | Accuracy | Precision | Recall | F1-Score | Rata-rata CV |
|---|---|---|---|---|---|
| MobileNetV3-Large | 97,05% | 97,15% | 97,21% | 97,17% | 91,87% ± 0,33% |
| ResNet50 | 96,05% | 96,11% | 96,24% | 96,15% | 94,08% ± 0,26% |
| **ConvNeXt-Tiny** | **99,57%** | **99,55%** | **99,56%** | **99,55%** | **97,74% ± 0,18%** |

**ConvNeXt-Tiny** terpilih sebagai model terbaik dan digunakan pada tahap implementasi aplikasi.

- Black-box testing: seluruh fitur aplikasi berfungsi sesuai kebutuhan
- Pengujian 20 citra data uji: kesesuaian **100%**
- Pengujian 20 citra eksternal (Google): kesesuaian **100%**
- Validasi 30 citra oleh dokter spesialis kulit dan kelamin (Sp.D.V.E): kesesuaian **96,67%** (29/30)

## Dataset

- **Sumber**: [Kaggle — Massive Skin Disease Balanced Dataset](https://www.kaggle.com) (lisensi MIT)
- **Total citra digunakan**: 30.260 citra (4 dari 34 kelas pada dataset asli)
- **Pembagian data**: 70% train, 20% validation, 10% test

| Kelas | Jumlah Citra | Persentase |
|---|---|---|
| Ringworm | 8.129 | 26,87% |
| Herpes Zoster | 8.082 | 26,71% |
| Eczema | 6.715 | 22,19% |
| Normal | 7.334 | 24,24% |

## Metodologi

Penelitian mengacu pada kerangka kerja **CRISP-DM**:

1. **Business Understanding** — menetapkan kriteria keberhasilan (akurasi/precision/recall/F1 ≥ 80%)
2. **Data Understanding** — eksplorasi karakteristik & distribusi dataset
3. **Data Preparation** — seleksi, pembersihan duplikat, split data, resize 224×224, normalisasi ImageNet, augmentasi (random crop, flip, rotation, color jitter, dll.)
4. **Modeling** — transfer learning dua tahap:
   - **Stage 1 (Warm-up Classifier)** — backbone dibekukan, hanya classifier dilatih
   - **Stage 2 (Fine-tuning)** — beberapa layer terakhir backbone dibuka dengan learning rate lebih kecil
5. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, Stratified K-Fold (3 fold)
6. **Deployment** — implementasi model ke aplikasi web Streamlit

### Hyperparameter

| Hyperparameter | Stage 1 | Stage 2 |
|---|---|---|
| Max epoch | 10 | 10 |
| Learning rate (classifier) | 1×10⁻⁵ | 1×10⁻⁶ |
| Learning rate (backbone) | – | 1×10⁻⁷ |
| Weight decay | – | 1×10⁻⁴ |
| Batch size | 32 | 32 |
| Optimizer | Adam | Adam |
| Loss function | CrossEntropyLoss (label smoothing 0.1) | CrossEntropyLoss (label smoothing 0.1) |

## Tech Stack

- **Bahasa**: Python 3.11.12 (aplikasi mendukung Python ≥ 3.9)
- **Deep Learning**: PyTorch & Torchvision (model ConvNeXt-Tiny)
- **Web App**: Streamlit
- **Data Processing**: NumPy, Pillow/PIL, Scikit-learn, Pandas
- **Visualisasi**: Matplotlib & Seaborn
- **Lainnya**: Gdown (mengunduh bobot model dari Google Drive)
- **Environment Training**: Google Colaboratory (GPU NVIDIA Tesla T4)
- **Version Control**: Git & GitHub

### Kebutuhan Hardware (menjalankan/melatih ulang secara lokal)

| Komponen | Spesifikasi Minimal | Rekomendasi |
|---|---|---|
| Prosesor | Intel Core i5 Gen 8 / setara | Core i7 / AMD Ryzen 5 ke atas |
| RAM | 8 GB | 16 GB (untuk pelatihan ulang model) |
| Penyimpanan | 20 GB ruang kosong | — |
| GPU | Opsional | NVIDIA CUDA-enabled GPU |
| Koneksi Internet | Diperlukan (unduh model dari Google Drive & menjalankan aplikasi) | — |

### Kebutuhan Software

- Sistem Operasi: Windows 10/11 (64-bit)
- Python 3.9 atau lebih baru
- Tools: Google Colaboratory / Jupyter Notebook, Visual Studio Code
- Git (opsional, jika project diunduh melalui repository)

## Akses Online (Tanpa Instalasi)

Aplikasi **SkinScan** sudah di-*deploy* menggunakan **Streamlit Community Cloud**, sehingga bisa langsung diakses tanpa instalasi apa pun:

1. Buka browser (Chrome, Firefox, Edge, dll.) dan pastikan perangkat terhubung internet.
2. Akses tautan: **https://skindiseaseproyek.streamlit.app/**
3. Tunggu proses pemuatan awal — aplikasi otomatis mengunduh berkas model dari Google Drive di sisi server (hanya terjadi sekali selama server aktif).
4. Halaman utama SkinScan akan tampil dan siap digunakan.

## Cara Penggunaan Aplikasi

### 1. Tampilan Awal
Setelah aplikasi terbuka, akan tampil *sidebar* dengan dua menu utama: **Deteksi** dan **Informasi**.

### 2. Menu Deteksi
1. Klik area unggah gambar, lalu pilih foto kulit dengan format **JPG/JPEG/PNG**.
2. Sistem akan otomatis menganalisis gambar menggunakan model **ConvNeXt-Tiny**.

### 3. Hasil Klasifikasi
- Hasil ditampilkan berupa nama kondisi kulit (**Eczema**, **Herpes Zoster**, **Normal**, atau **Ringworm**) beserta *confidence score* dalam persen.
- Di bawah hasil, ditampilkan deskripsi singkat mengenai kondisi tersebut beserta tautan sumber jurnal ilmiah terkait.

### 4. Menu Informasi
- Menampilkan empat tombol kondisi kulit (Eczema, Herpes Zoster, Normal, Ringworm).
- Klik salah satu tombol untuk menampilkan penjelasan lengkap mengenai kondisi tersebut tanpa perlu mengunggah gambar — cocok untuk keperluan edukasi.

### 5. Keluar dari Aplikasi
Aplikasi dapat ditutup dengan menutup tab browser.

## ⚠️ Batasan

- Aplikasi hanya menghasilkan **label klasifikasi** dan **confidence score** — tidak mencakup segmentasi lesi, deteksi lokasi, atau rekomendasi pengobatan.
- Aplikasi berfungsi sebagai **alat bantu screening awal** dan **tidak dimaksudkan untuk menggantikan diagnosis medis** oleh tenaga kesehatan profesional.
- Model dilatih dan diuji pada dataset publik (Kaggle); performa pada citra dari fasilitas kesehatan nyata di Indonesia belum diuji secara khusus.

## Pengembangan Selanjutnya

- Menambahkan lebih banyak jenis penyakit kulit dan dataset dari fasilitas kesehatan Indonesia
- Menguji model pada citra dengan variasi kualitas, pencahayaan, dan latar belakang yang lebih realistis
- Menambahkan fitur segmentasi lesi, deteksi area penyakit, dan rekomendasi penanganan awal
- Eksplorasi arsitektur terkini seperti Vision Transformer (ViT) atau pendekatan ensemble
- Validasi klinis skala lebih luas dengan lebih banyak tenaga medis
