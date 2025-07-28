# BERT Sentiment Analysis Web App

Aplikasi web untuk prediksi sentimen menggunakan model BERT yang telah dilatih.

## Fitur

- Prediksi sentimen (negatif, netral, positif)
- Interface web yang user-friendly
- Model BERT yang telah dilatih untuk analisis sentimen

## Deployment di Render.com

### Langkah-langkah Deployment:

1. **Push kode ke GitHub**

   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy di Render.com**
   - Buka [Render.com](https://render.com)
   - Klik "New +" → "Web Service"
   - Connect dengan repository GitHub Anda
   - Render akan otomatis mendeteksi konfigurasi dari `.render.yaml`

### Konfigurasi yang Telah Disiapkan:

- **Runtime**: Python 3.9.18
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
- **Environment Variables**:
  - `PYTHON_VERSION`: 3.9.18
  - `PORT`: 10000

## Struktur Proyek

```
bert_webapp/
├── app.py                 # Aplikasi Flask utama
├── requirements.txt       # Dependensi Python
├── runtime.txt           # Versi Python
├── .render.yaml          # Konfigurasi Render.com
├── Procfile             # Konfigurasi deployment
├── templates/
│   └── index.html       # Template HTML
└── model/               # Model BERT yang telah dilatih
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

## Penggunaan

1. Masukkan teks yang ingin dianalisis sentimennya
2. Klik "Prediksi Sentimen"
3. Hasil akan ditampilkan (negatif, netral, atau positif)

## Troubleshooting

Jika terjadi error saat deployment:

1. **Periksa log di Render.com** - Lihat bagian "Logs" untuk detail error
2. **Pastikan semua file model ada** - File `model.safetensors` dan file terkait harus ada di folder `model/`
3. **Periksa requirements.txt** - Pastikan semua dependensi tercantum dengan versi yang benar
4. **Restart deployment** - Kadang restart dapat mengatasi masalah

## Kontak

Dibuat oleh Rifqi - 2025
