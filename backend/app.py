from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from utils.image_processing import flood_fill_highlight

app = Flask(__name__, template_folder='../frontend/public', static_folder='../frontend/public')

# Конфигурация
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Проверка допустимых форматов файла
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'Ошибка: файл не найден.'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return jsonify({'message': 'Изображение загружено успешно!', 'image_url': f"/uploads/{filename}"})
    else:
        return jsonify({'message': 'Ошибка: Неверный формат файла.'}), 400


@app.route('/process-click', methods=['POST'])
def process_click():
    data = request.json
    x, y = int(data['x']), int(data['y'])
    image_path = os.path.join(UPLOAD_FOLDER, data['image'])

    # Вызываем функцию flood_fill для выделения области
    mask_path = flood_fill_highlight(image_path, x, y)

    if not mask_path:
        return jsonify({'message': 'Ошибка выделения области.'}), 500

    return jsonify({'mask_url': f"/uploads/{os.path.basename(mask_path)}"})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(app.static_folder, 'css'), filename)


if __name__ == '__main__':
    app.run(debug=True)
