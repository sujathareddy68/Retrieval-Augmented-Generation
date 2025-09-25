from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from rag_chain import answer_question_from_pdf, check_ollama_connection

app = Flask(__name__)
CORS(app)
from flask import render_template

@app.route('/')
def home():
    return render_template('index.html')


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['pdf']
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)
    return jsonify({'message': 'PDF uploaded successfully', 'pdf_path': filename})
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    pdf_path = data.get('pdf_path')
    question = data.get('question')

    if not pdf_path or not question:
        return jsonify({'error': 'Missing pdf_path or question'}), 400

    try:
        result = answer_question_from_pdf(pdf_path, question)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    ok = check_ollama_connection()
    if ok:
        return jsonify({'status': 'ok', 'ollama': 'reachable'})
    else:
        return jsonify({'status': 'error', 'ollama': 'unreachable'}), 503
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

#curl -X POST -F "pdf=@sample.pdf" http://localhost:5000/upload
#curl -X POST http://localhost:5000/ask \
#-H "Content-Type: application/json" \
#-d '{"pdf_path": "uploads/sample.pdf", "question": "What is the summary?"}'

