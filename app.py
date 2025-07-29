from flask import Flask, request, render_template, send_file, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import os
import pandas as pd
import io
import base64
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from collections import Counter
import json

app = Flask(__name__)

# === Load model/tokenizer dari folder model ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

label_map_inv = {0: 'negatif', 1: 'netral', 2: 'positif'}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text):
    text = preprocess(text)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        # Get probabilities using softmax
        probabilities = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(outputs.logits, dim=1).item()
        
        # Convert to percentages
        prob_percentages = {
            'negatif': round(probabilities[0][0].item() * 100, 2),
            'netral': round(probabilities[0][1].item() * 100, 2),
            'positif': round(probabilities[0][2].item() * 100, 2)
        }
        
        return label_map_inv[pred], prob_percentages

def process_file_data(df, text_column):
    """Process file data and return results with sentiment analysis"""
    results = []
    
    for index, row in df.iterrows():
        text = str(row[text_column])
        sentiment, probabilities = predict_sentiment(text)
        
        results.append({
            'index': index + 1,
            'text': text[:100] + '...' if len(text) > 100 else text,
            'sentiment': sentiment,
            'negatif_prob': probabilities['negatif'],
            'netral_prob': probabilities['netral'],
            'positif_prob': probabilities['positif']
        })
    
    return results

def calculate_analytics(results):
    """Calculate analytics from sentiment results"""
    if not results:
        return {
            'total_analyzed': 0,
            'sentiment_distribution': {'negatif': 0, 'netral': 0, 'positif': 0},
            'sentiment_percentages': {'negatif': 0, 'netral': 0, 'positif': 0},
            'average_probabilities': {'negatif': 0, 'netral': 0, 'positif': 0},
            'confidence_scores': [],
            'top_keywords': []
        }
    
    # Basic statistics
    total_analyzed = len(results)
    sentiment_counts = Counter([r['sentiment'] for r in results])
    sentiment_distribution = dict(sentiment_counts)
    
    # Calculate percentages
    sentiment_percentages = {
        'negatif': round((sentiment_distribution.get('negatif', 0) / total_analyzed) * 100, 2),
        'netral': round((sentiment_distribution.get('netral', 0) / total_analyzed) * 100, 2),
        'positif': round((sentiment_distribution.get('positif', 0) / total_analyzed) * 100, 2)
    }
    
    # Average probabilities
    avg_negatif = sum(r['negatif_prob'] for r in results) / total_analyzed
    avg_netral = sum(r['netral_prob'] for r in results) / total_analyzed
    avg_positif = sum(r['positif_prob'] for r in results) / total_analyzed
    
    average_probabilities = {
        'negatif': round(avg_negatif, 2),
        'netral': round(avg_netral, 2),
        'positif': round(avg_positif, 2)
    }
    
    # Confidence scores (max probability for each prediction)
    confidence_scores = []
    for r in results:
        max_prob = max(r['negatif_prob'], r['netral_prob'], r['positif_prob'])
        confidence_scores.append({
            'index': r['index'],
            'confidence': round(max_prob, 2),
            'sentiment': r['sentiment']
        })
    
    # Sort by confidence (descending)
    confidence_scores.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Extract keywords (simple approach - most common words)
    all_text = ' '.join([r['text'] for r in results])
    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
    word_counts = Counter(words)
    # Remove common Indonesian stop words
    stop_words = {'yang', 'dan', 'dengan', 'dari', 'untuk', 'dalam', 'pada', 'ke', 'di', 'sebagai', 'oleh', 'itu', 'ini', 'atau', 'juga', 'akan', 'bisa', 'dapat', 'sudah', 'masih', 'sangat', 'lebih', 'kurang', 'sama', 'lain', 'lainnya', 'semua', 'setiap', 'beberapa', 'banyak', 'sedikit', 'tidak', 'bukan', 'tanpa', 'dengan', 'oleh', 'karena', 'jika', 'ketika', 'sebelum', 'sesudah', 'selama', 'sampai', 'hingga', 'melalui', 'terhadap', 'tentang', 'mengenai', 'berdasarkan', 'menurut', 'seperti', 'bagi', 'untuk', 'kepada', 'dari', 'oleh', 'dengan', 'tanpa', 'melalui', 'terhadap', 'tentang', 'mengenai', 'berdasarkan', 'menurut', 'seperti', 'bagi', 'untuk', 'kepada'}
    filtered_words = {word: count for word, count in word_counts.items() if word not in stop_words}
    top_keywords = [{'word': word, 'count': count} for word, count in sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    return {
        'total_analyzed': total_analyzed,
        'sentiment_distribution': sentiment_distribution,
        'sentiment_percentages': sentiment_percentages,
        'average_probabilities': average_probabilities,
        'confidence_scores': confidence_scores[:10],  # Top 10 most confident
        'top_keywords': top_keywords
    }

def read_file_data(file):
    """Read file data based on file extension"""
    filename = file.filename.lower()
    
    if filename.endswith('.csv'):
        return pd.read_csv(file)
    elif filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV, XLS, atau XLSX.")

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    probabilities = None
    if request.method == "POST":
        input_text = request.form["text"]
        result, probabilities = predict_sentiment(input_text)
    return render_template("index.html", result=result, probabilities=probabilities, header_title="Prediksi Sentimen Argumen Sampah")

@app.route("/analytic")
def analytic():
    """Dashboard analytics page"""
    return render_template("analytic.html", header_title="Dashboard Analytics")

@app.route("/api/analytics", methods=["POST"])
def api_analytics():
    """API endpoint for analytics calculation"""
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    
    try:
        # Read file data
        df = read_file_data(file)
        
        # Automatically use 'text' column or first column if 'text' doesn't exist
        if 'text' in df.columns:
            text_column = 'text'
        else:
            text_column = df.columns[0]
        
        # Process data
        results = process_file_data(df, text_column)
        
        # Calculate analytics
        analytics = calculate_analytics(results)
        
        return jsonify({
            'success': True,
            'analytics': analytics,
            'results': results,
            'filename': file.filename,
            'text_column': text_column
        })
        
    except Exception as e:
        return jsonify({'error': f'Error memproses file: {str(e)}'}), 400

@app.route("/download-template")
def download_template():
    """Download CSV template"""
    # Create sample data for template
    sample_data = {
        'text': [
            'Sampah plastik sangat merusak lingkungan kita',
            'Pengelolaan sampah yang baik sangat penting',
            'Masyarakat harus sadar akan pentingnya kebersihan',
            'Tempat sampah yang tersedia sudah cukup memadai',
            'Kampanye anti sampah plastik sangat efektif'
        ],
        'kategori': [
            'lingkungan',
            'pengelolaan',
            'kesadaran',
            'fasilitas',
            'kampanye'
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Create CSV in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.seek(0)
    
    # Create BytesIO object for sending file
    csv_bytes = io.BytesIO()
    csv_bytes.write(csv_buffer.getvalue().encode('utf-8'))
    csv_bytes.seek(0)
    
    return send_file(
        csv_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name='template_sentimen_analisis.csv'
    )

@app.route("/download-template-excel")
def download_template_excel():
    """Download Excel template"""
    # Create sample data for template
    sample_data = {
        'text': [
            'Sampah plastik sangat merusak lingkungan kita',
            'Pengelolaan sampah yang baik sangat penting',
            'Masyarakat harus sadar akan pentingnya kebersihan',
            'Tempat sampah yang tersedia sudah cukup memadai',
            'Kampanye anti sampah plastik sangat efektif'
        ],
        'kategori': [
            'lingkungan',
            'pengelolaan',
            'kesadaran',
            'fasilitas',
            'kampanye'
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Create Excel file in memory
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Data']
        
        # Style the header row
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        
        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    excel_buffer.seek(0)
    
    return send_file(
        excel_buffer,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='template_sentimen_analisis.xlsx'
    )

@app.route("/multi-argument", methods=["GET", "POST"])
def file_upload():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("multi-argument.html", error="Tidak ada file yang dipilih", header_title="Upload File - Prediksi Sentimen")
        
        file = request.files['file']
        if file.filename == '':
            return render_template("multi-argument.html", error="Tidak ada file yang dipilih", header_title="Upload File - Prediksi Sentimen")
        
        # Check file extension
        filename = file.filename.lower()
        if not (filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls')):
            return render_template("multi-argument.html", error="File harus berformat CSV, XLS, atau XLSX", header_title="Upload File - Prediksi Sentimen")
        
        try:
            # Read file data
            df = read_file_data(file)
            
            # Get column names for selection
            columns = df.columns.tolist()
            
            # Automatically use 'text' column or first column if 'text' doesn't exist
            if 'text' in df.columns:
                text_column = 'text'
            else:
                text_column = df.columns[0]  # Use first column as fallback
            
            results = process_file_data(df, text_column)
            return render_template("multi-argument.html", results=results, filename=file.filename, text_column=text_column, header_title="Upload File - Prediksi Sentimen")
                
        except Exception as e:
            return render_template("multi-argument.html", error=f"Error membaca file: {str(e)}", header_title="Upload File - Prediksi Sentimen")
    
    return render_template("multi-argument.html", header_title="Upload File - Prediksi Sentimen")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 1000))
    app.run(host="0.0.0.0", port=port, debug=False)
