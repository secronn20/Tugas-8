import os
import time
import json
import numpy as np
from PIL import Image
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Lokasi dataset dan model
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.joblib')

# Pastikan folder model ada
os.makedirs(MODEL_DIR, exist_ok=True)

def extract_features(image_path, target_size=(64, 64)):
    """
    Membaca gambar, melakukan prapemrosesan, dan mengekstrak fitur:
    1. Grayscale pixel values (64x64 = 4096 fitur)
    2. Color moments (Mean dan Std Dev untuk R, G, B channels = 6 fitur)
    Total fitur = 4102
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Konversi ke RGB jika format lain (misal RGBA atau Grayscale)
        img_rgb = img.convert('RGB')
        
        # Ekstraksi fitur warna (Color Moments) sebelum resize ke grayscale
        img_rgb_np = np.array(img_rgb)
        r_mean = np.mean(img_rgb_np[:, :, 0])
        g_mean = np.mean(img_rgb_np[:, :, 1])
        b_mean = np.mean(img_rgb_np[:, :, 2])
        r_std = np.std(img_rgb_np[:, :, 0])
        g_std = np.std(img_rgb_np[:, :, 1])
        b_std = np.std(img_rgb_np[:, :, 2])
        color_features = np.array([r_mean, g_mean, b_mean, r_std, g_std, b_std])
        
        # Konversi ke grayscale untuk fitur bentuk/tekstur dan resize
        img_gray = img_rgb.convert('L')
        img_gray_resized = img_gray.resize(target_size, Image.Resampling.LANCZOS)
        gray_features = np.array(img_gray_resized).flatten() / 255.0  # Normalisasi ke [0, 1]
        
        # Gabungkan fitur grayscale dan fitur warna
        features = np.concatenate([gray_features, color_features])
        return features
    except Exception as e:
        print(f"Error memproses {image_path}: {e}")
        return None

def main():
    print("=" * 60)
    print("  MEMULAI PELATIHAN MODEL RANDOM FOREST UNTUK KLASIFIKASI DAUN  ")
    print("=" * 60)
    
    # 1. Mendapatkan daftar kelas dari nama subdirektori di dataset
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Folder dataset '{DATASET_DIR}' tidak ditemukan!")
        return
        
    class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    print(f"Kategori daun yang ditemukan ({len(class_names)}): {class_names}")
    
    X = []
    y = []
    
    # Hitung jumlah gambar per kelas
    class_counts = {}
    
    print("\n[1/4] Memuat dataset dan mengekstrak fitur gambar...")
    start_time = time.time()
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(DATASET_DIR, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        class_counts[class_name] = len(image_files)
        print(f"  - Memproses kelas '{class_name}': {len(image_files)} gambar")
        
        for file_name in image_files:
            file_path = os.path.join(class_path, file_name)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(class_idx)
                
    X = np.array(X)
    y = np.array(y)
    
    duration = time.time() - start_time
    print(f"Selesai mengekstrak fitur dalam {duration:.2f} detik.")
    print(f"Total sampel valid: {len(X)}")
    print(f"Dimensi matriks fitur (X): {X.shape}")
    print(f"Dimensi label (y): {y.shape}")
    
    # 2. Split Dataset menjadi Train dan Test
    print("\n[2/4] Membagi dataset menjadi Data Latih (80%) dan Data Uji (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Jumlah Data Latih (Train Set): {len(X_train)}")
    print(f"  - Jumlah Data Uji (Test Set): {len(X_test)}")
    
    # 3. Pelatihan Model Random Forest
    print("\n[3/4] Melatih model Random Forest...")
    rf_start_time = time.time()
    
    # Konfigurasi Random Forest yang optimal
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    rf_duration = time.time() - rf_start_time
    print(f"Model berhasil dilatih dalam {rf_duration:.2f} detik.")
    
    # 4. Evaluasi Model
    print("\n[4/4] Mengevaluasi performa model...")
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n>>> AKURASI MODEL PADA DATA UJI: {accuracy * 100:.2f}% <<<\n")
    
    print("Laporan Klasifikasi Detail:")
    print("-" * 65)
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("-" * 65)
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    # Tampilkan confusion matrix secara rapi
    header_text = 'Asli \\ Prediksi'
    print(f"{header_text:<20}", end="")
    for name in class_names:
        print(f"{name:>15}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name:<20}", end="")
        for val in cm[i]:
            print(f"{val:>15}", end="")
        print()
    print("-" * 65)
    
    # 5. Menyimpan model terlatih beserta nama kelasnya
    print(f"\nMenyimpan model ke '{MODEL_PATH}'...")
    model_data = {
        'model': rf_model,
        'class_names': class_names,
        'accuracy': accuracy,
        'features_info': {
            'target_size': (64, 64),
            'feature_len': X.shape[1]
        }
    }
    joblib.dump(model_data, MODEL_PATH)
    print("Model berhasil disimpan! Siap digunakan untuk aplikasi web Flask.")
    print("=" * 60)

if __name__ == '__main__':
    main()
