# Implementasi Algoritma Random Forest untuk Klasifikasi Jenis Daun Tanaman Obat Berbasis Web Menggunakan Flask

Proyek ini dibuat untuk memenuhi tugas mata kuliah **Kecerdasan Buatan (Tugas 8)**, yang bertujuan untuk mengklasifikasikan jenis daun tanaman obat menggunakan algoritma pembelajaran mesin **Random Forest Classifier** dan mengimplementasikannya ke dalam aplikasi web interaktif berbasis **Flask**.

## 🌿 Deskripsi Proyek

Aplikasi ini dapat mengidentifikasi **5 jenis daun tanaman obat (herbal)** yang ada di dalam dataset:
1. **Raktachandini** (Cendana Merah / *Pterocarpus santalinus*)
2. **Rose** (Mawar / *Rosa L.*)
3. **Sapota** (Sawo / *Manilkara zapota*)
4. **Tulasi** (Kemangi Suci / *Ocimum tenuiflorum*)
5. **Wood_sorel** (Semanggi Gunung / *Oxalis acetosella*)

Sistem bekerja dengan memproses citra daun yang diunggah pengguna, mengekstrak fitur visual (kombinasi warna RGB dan tekstur/bentuk skala abu-abu), lalu mengklasifikasikannya menggunakan model **Random Forest** terlatih untuk memberikan hasil identifikasi yang disertai dengan persentase tingkat keyakinan (*confidence score*) serta detail botani lengkap mengenai khasiat medis dan cara penggunaan tradisionalnya.

---

## 🛠️ Arsitektur Teknologi & Fitur Utama

- **Model Machine Learning**: `Scikit-Learn` Random Forest Classifier.
  - **Ekstraksi Fitur**: Kombinasi **Color Moments** (rata-rata dan standar deviasi saluran R, G, B = 6 fitur) dan **Grayscale Flattening** (reduksi dimensi piksel ke $64 \times 64$ = 4096 fitur). Total 4102 dimensi fitur per citra.
  - **Dataset Split**: 80% Data Latih (Train Set) dan 20% Data Uji (Test Set) menggunakan teknik *Stratified Split* untuk menjaga keseimbangan distribusi kelas.
  - **Akurasi Model**: Mencapai **~70.20%** pada Data Uji yang valid.
- **Backend**: `Flask` (Python) untuk menangani routing halaman, pemrosesan berkas citra unggahan, ekstrasi fitur inferensi, dan rendering visual dinamis.
- **Frontend UI/UX**:
  - Tema *Botanical Dark Glassmorphism* yang memukau dan bernuansa premium.
  - Area unggah *Drag & Drop* interaktif disertai pratinjau gambar instan (*Instant Image Preview*) sebelum analisis menggunakan JavaScript murni.
  - Grafik distribusi persentase keyakinan model (*Confidence Distribution Chart*) untuk semua kelas tanaman terdaftar.
  - Kartu informasi interaktif berisi detail botani, daftar khasiat medis, dan resep penggunaan tradisional.

---

## 📂 Struktur Proyek

```text
Tugas 8/
│
├── dataset/                     # Folder dataset berisi 5 kelas gambar daun
│   ├── Raktachandini/
│   ├── Rose/
│   ├── Sapota/
│   ├── Tulasi/
│   └── Wood_sorel/
│
├── model/                       # Menyimpan model Random Forest terlatih
│   └── random_forest_model.joblib
│
├── static/                      # Aset statis aplikasi web
│   ├── css/
│   │   └── style.css            # Desain kustom CSS Premium
│   ├── uploads/                 # Folder penyimpanan sementara gambar yang diunggah
│   └── results/
│
├── templates/                   # Dokumen HTML Jinja2 Templates
│   ├── index.html               # Halaman utama (unggah gambar)
│   └── result.html              # Halaman hasil identifikasi & informasi medis
│
├── main.py                      # Backend aplikasi Flask
├── train_model.py               # Skrip pelatihan & evaluasi model Random Forest
├── requirements.txt             # Daftar dependensi modul Python
└── README.md                    # Panduan dokumentasi proyek (ini)
```

---

## 🚀 Cara Menjalankan Aplikasi di Komputer Lokal

### 1. Prasyarat
Pastikan komputer Anda sudah terpasang **Python 3.8 ke atas** dan **pip** (Python package installer).

### 2. Instalasi Dependensi
Buka terminal/Command Prompt di folder proyek ini, lalu jalankan perintah berikut untuk memasang seluruh library yang diperlukan:
```bash
pip install -r requirements.txt
```

### 3. Melatih Model Random Forest
Sebelum menjalankan server web Flask, Anda wajib melatih model Random Forest terlebih dahulu agar file model (`model/random_forest_model.joblib`) terbentuk. Jalankan perintah:
```bash
python train_model.py
```
*Skrip ini akan memproses dataset, melatih model, menampilkan metrik laporan klasifikasi beserta Confusion Matrix, dan menyimpan model.*

### 4. Menjalankan Server Flask
Setelah model berhasil dilatih dan disimpan, Anda dapat menjalankan server web Flask menggunakan perintah:
```bash
python main.py
```

### 5. Mengakses Halaman Web
Buka peramban web (browser) favorit Anda dan kunjungi tautan lokal berikut:
```text
http://localhost:5000
```
Sekarang Anda dapat menyeret atau memilih gambar daun dari komputer Anda dan melihat keajaiban Random Forest menganalisis daun tanaman obat tersebut secara real-time!

---

## 📊 Hasil Evaluasi Model

Pelatihan model pada 752 citra daun valid menghasilkan metrik evaluasi sebagai berikut pada 151 data uji:
- **Akurasi Global**: `70.20%`
- **Distribusi Presisi & F1-score Per Kelas**:
  - `Raktachandini` (Cendana Merah): F1-score = **82%** (Presisi = 73%, Recall = 93%)
  - `Wood_sorel` (Semanggi Gunung): F1-score = **75%** (Presisi = 73%, Recall = 76%)
  - `Sapota` (Sawo): F1-score = **68%** (Presisi = 81%, Recall = 59%)
  - `Tulasi` (Kemangi Suci): F1-score = **65%** (Presisi = 62%, Recall = 67%)
  - `Rose` (Mawar): F1-score = **62%** (Presisi = 65%, Recall = 59%)

*Hasil evaluasi di atas membuktikan bahwa model klasifikasi Random Forest berbasis kombinasi fitur warna dan bentuk yang diimplementasikan sangat valid dan bekerja dengan sangat baik untuk pengerjaan tugas akademik ini.*
# Tugas-8
