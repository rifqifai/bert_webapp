from flask import Flask, request, render_template, send_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import os
import pandas as pd
import io
import base64
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

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
