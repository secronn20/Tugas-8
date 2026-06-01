import os
import uuid
import numpy as np
from PIL import Image
import joblib
from flask import Flask, request, render_template, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'tugas8_randomforest_secret_key_botanical'

# Folder Konfigurasi
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Maksimum file 5MB

# Path Model
MODEL_PATH = os.path.join(app.root_path, 'model', 'random_forest_model.joblib')

# Data Detail Tanaman Obat (Botanical Database)
LEAF_DETAILS = {
    'Raktachandini': {
        'common_name': 'Cendana Merah (Raktachandini / Red Sandalwood)',
        'scientific_name': 'Pterocarpus santalinus',
        'description': 'Cendana merah adalah pohon kecil yang terkenal dengan kayu terasnya yang berwarna merah tua, bertekstur keras, dan tidak berbau wangi seperti cendana putih biasa. Daunnya berbentuk bulat oval dengan susunan majemuk dan permukaan halus.',
        'benefits': [
            'Perawatan Kulit: Membantu meredakan jerawat, bekas luka, hiperpigmentasi, dan memberikan efek mendinginkan kulit wajah.',
            'Anti-inflamasi: Mengurangi pembengkakan, peradangan akut, dan nyeri pada persendian.',
            'Kesehatan Pencernaan: Digunakan dalam pengobatan tradisional untuk mengobati diare, disentri, dan maag.',
            'Antioksidan Tinggi: Memiliki kandungan polifenol yang tinggi guna menangkal efek radikal bebas.'
        ],
        'usage': 'Untuk kulit, kayu teras yang dihaluskan menjadi bubuk dicampur dengan air mawar atau madu untuk dijadikan masker wajah. Untuk pengobatan dalam, harus dikonsultasikan secara hati-hati dengan herbalis medis profesional.'
    },
    'Rose': {
        'common_name': 'Mawar (Rose)',
        'scientific_name': 'Rosa L.',
        'description': 'Mawar adalah tanaman semak berdunga indah dengan kelopak berlapis-lapis dan duri tajam pada batangnya. Daun mawar umumnya berbentuk menyirip ganjil (pinnate) dengan tepi bergerigi halus.',
        'benefits': [
            'Kesehatan Kulit: Air mawar bertindak sebagai toner alami yang menyeimbangkan pH kulit dan mengurangi kemerahan akibat iritasi.',
            'Menenangkan Pikiran: Teh kelopak mawar atau minyak esensial mawar membantu mengurangi kecemasan, stres, dan memperbaiki kualitas tidur.',
            'Kaya Vitamin C: Kelopak bunga dan daunnya mengandung vitamin C alami yang tinggi untuk meningkatkan kekebalan tubuh.',
            'Anti-bakteri: Membantu mendisinfeksi dan mempercepat penyembuhan luka luar ringan.'
        ],
        'usage': 'Kelopak mawar kering dapat diseduh dengan air panas mendidih selama 5 menit menjadi teh herbal penenang. Air hasil penyulingan mawar (Rose Water) dapat disemprotkan langsung sebagai penyegar wajah.'
    },
    'Sapota': {
        'common_name': 'Sawo (Sapota / Sapodilla)',
        'scientific_name': 'Manilkara zapota',
        'description': 'Sawo adalah pohon buah tropis yang hijau sepanjang tahun. Daun sawo berbentuk oval memanjang (oblong), tebal seperti kulit (coriaceous), berwarna hijau tua mengkilap di bagian atas, dan tersusun spiral di ujung cabang.',
        'benefits': [
            'Sumber Energi Alami: Mengandung kadar fruktosa dan glukosa tinggi yang cepat memulihkan stamina.',
            'Kesehatan Pencernaan: Serat tinggi dan kandungan tanin bertindak sebagai agen anti-inflamasi alami pada lambung dan usus.',
            'Kesehatan Tulang: Kaya akan kalsium, fosfor, dan zat besi yang sangat penting dalam memperkuat kepadatan tulang.',
            'Ekspektoran Alami: Sifat ekspektoran membantu meredakan batuk dan pilek dengan mengeluarkan lendir di saluran napas.'
        ],
        'usage': 'Buah sawo dikonsumsi segar atau diolah menjadi jus. Air rebusan daun sawo muda secara tradisional digunakan untuk berkumur guna meredakan sariawan, radang gusi, dan sakit gigi karena kaya akan zat tanin astringen.'
    },
    'Tulasi': {
        'common_name': 'Tulasi (Kemangi Suci / Holy Basil)',
        'scientific_name': 'Ocimum tenuiflorum',
        'description': 'Tulasi adalah tanaman aromatik suci dalam pengobatan Ayurveda yang dikenal sebagai "Ratu Herbal". Daunnya kecil berbentuk oval, berbulu halus, memiliki tepi bergerigi halus, dan mengeluarkan aroma mint-cengkeh yang khas.',
        'benefits': [
            'Adaptogen Kuat (Anti-Stres): Membantu tubuh beradaptasi secara fisiologis dan psikologis terhadap stres emosional.',
            'Pengobatan Respirasi: Sangat efektif meredakan batuk, pilek, asma, bronkitis, serta radang tenggorokan.',
            'Peningkat Imunitas (Immunomodulator): Senyawa fitokimianya memperkuat respons sel imun dalam melawan virus dan bakteri.',
            'Regulasi Gula Darah: Berfungsi membantu mengontrol dan menurunkan kadar glukosa darah puasa.'
        ],
        'usage': 'Daun tulasi segar dapat dikunyah langsung setelah dicuci bersih, atau diseduh bersama air panas selama 10 menit untuk diminum sebagai teh Tulasi hangat guna melegakan pernapasan.'
    },
    'Wood_sorel': {
        'common_name': 'Wood Sorel (Semanggi Gunung / Calincing)',
        'scientific_name': 'Oxalis acetosella',
        'description': 'Wood Sorel adalah tanaman herbal kecil berakar rimpang dengan daun trifoliate (tiga helai) mirip semanggi. Daunnya berbentuk hati terbalik (obcordate) yang sensitif menutup pada malam hari atau cuaca mendung.',
        'benefits': [
            'Sumber Vitamin C: Memiliki rasa asam lemon segar yang secara historis ampuh mengatasi sariawan dan skurvi.',
            'Efek Mendinginkan (Cooling Effect): Membantu menurunkan suhu tubuh saat demam tinggi dan meredakan haus ekstrim.',
            'Diuretik & Detoksifikasi: Melancarkan buang air kecil sehingga membantu ginjal menyaring racun dari tubuh.',
            'Pereda Ruam Kulit: Air remasan daun segar dapat dioleskan pada ruam gatal, eksim ringan, atau sengatan serangga.'
        ],
        'usage': 'Daun segar dapat dicampurkan sedikit pada salad untuk rasa asam lemon yang alami. Namun hindari konsumsi berlebihan bagi penderita batu ginjal atau rematik karena kadar asam oksalatnya cukup tinggi.'
    }
}

def extract_features(image_path, target_size=(64, 64)):
    """
    Ekstraksi fitur yang sama persis dengan train_model.py
    """
    try:
        img = Image.open(image_path)
        img_rgb = img.convert('RGB')
        
        # 1. Color Moments
        img_rgb_np = np.array(img_rgb)
        r_mean = np.mean(img_rgb_np[:, :, 0])
        g_mean = np.mean(img_rgb_np[:, :, 1])
        b_mean = np.mean(img_rgb_np[:, :, 2])
        r_std = np.std(img_rgb_np[:, :, 0])
        g_std = np.std(img_rgb_np[:, :, 1])
        b_std = np.std(img_rgb_np[:, :, 2])
        color_features = np.array([r_mean, g_mean, b_mean, r_std, g_std, b_std])
        
        # 2. Grayscale Resized & Flattened
        img_gray = img_rgb.convert('L')
        img_gray_resized = img_gray.resize(target_size, Image.Resampling.LANCZOS)
        gray_features = np.array(img_gray_resized).flatten() / 255.0
        
        return np.concatenate([gray_features, color_features])
    except Exception as e:
        print(f"Error mengekstrak fitur: {e}")
        return None

def get_model():
    """
    Memuat model dari folder model/
    """
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Gagal memuat model: {e}")
            return None
    return None

@app.route('/')
def index():
    model_data = get_model()
    model_status = {
        'trained': model_data is not None,
        'accuracy': f"{model_data['accuracy'] * 100:.2f}%" if model_data else "Belum Dilatih",
        'class_names': model_data['class_names'] if model_data else []
    }
    return render_template('index.html', model_status=model_status, leaf_list=LEAF_DETAILS)

@app.route('/predict', methods=['POST'])
def predict():
    # Periksa apakah model sudah dilatih
    model_data = get_model()
    if model_data is None:
        flash("Model klasifikasi belum dilatih atau tidak ditemukan! Silakan jalankan 'train_model.py' terlebih dahulu.", "danger")
        return redirect(url_for('index'))
        
    # Periksa apakah ada file gambar dalam request
    if 'image' not in request.files:
        flash("Tidak ada bagian file gambar dalam form.", "warning")
        return redirect(url_for('index'))
        
    file = request.files['image']
    if file.filename == '':
        flash("Tidak ada berkas gambar yang dipilih untuk diunggah.", "warning")
        return redirect(url_for('index'))
        
    if file:
        # Cek ekstensi
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
            flash("Format berkas tidak didukung! Harap unggah berkas bertipe JPG, JPEG, PNG, atau WEBP.", "danger")
            return redirect(url_for('index'))
            
        # Simpan file dengan nama unik menggunakan uuid
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Ekstraksi fitur
        features = extract_features(filepath)
        if features is None:
            flash("Terjadi kesalahan saat memproses gambar. Pastikan berkas gambar tidak rusak.", "danger")
            return redirect(url_for('index'))
            
        # Bentuk ulang untuk prediksi (single sample)
        features = features.reshape(1, -1)
        
        # Klasifikasi menggunakan model
        rf_model = model_data['model']
        class_names = model_data['class_names']
        
        # Dapatkan prediksi probabilitas (Confidence)
        probabilities = rf_model.predict_proba(features)[0]
        prediction_idx = np.argmax(probabilities)
        confidence = probabilities[prediction_idx]
        
        predicted_class = class_names[prediction_idx]
        
        # Siapkan probabilitas per kelas untuk visualisasi grafik
        class_probs = []
        for idx, prob in enumerate(probabilities):
            class_probs.append({
                'name': class_names[idx],
                'display_name': LEAF_DETAILS.get(class_names[idx], {}).get('common_name', class_names[idx]),
                'prob_pct': round(prob * 100, 2)
            })
        # Urutkan berdasarkan probabilitas tertinggi
        class_probs = sorted(class_probs, key=lambda x: x['prob_pct'], reverse=True)
        
        # Ambil informasi detail botani
        details = LEAF_DETAILS.get(predicted_class, {
            'common_name': predicted_class,
            'scientific_name': 'Unknown',
            'description': 'Informasi tidak tersedia untuk jenis tanaman obat ini.',
            'benefits': [],
            'usage': 'Tidak ada panduan.'
        })
        
        # Path relatif gambar untuk digunakan di template
        image_url = url_for('static', filename=f"uploads/{unique_filename}")
        
        result_data = {
            'class': predicted_class,
            'confidence_pct': round(confidence * 100, 2),
            'image_url': image_url,
            'details': details,
            'class_probs': class_probs
        }
        
        return render_template('result.html', result=result_data)
        
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Jalankan Flask app di port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
