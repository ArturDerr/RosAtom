from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from utils.image_processing import process_image, auto_select_area, highlight_front_door, highlight_trunk, \
    highlight_hood, analyze_and_highlight

app = Flask(__name__, template_folder='../frontend/public', static_folder='../frontend/public')

# Конфигурация
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
current_image_path = None  # Переменная для хранения пути к текущему загруженному изображению


# Проверка допустимых форматов файла
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global current_image_path  # Глобальная переменная для пути
    if 'file' not in request.files:
        return jsonify({'message': 'Ошибка: файл не найден.'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Обновляем глобальную переменную
        current_image_path = filepath

        # Возвращаем ссылку на загруженное изображение
        return jsonify({
            "message": "Файл успешно загружен.",
            "original_image_url": f"/uploads/{filename}",
        })

    return jsonify({'message': 'Ошибка: Неверный формат файла.'}), 400


@app.route('/select_area', methods=['POST'])
def select_area():
    data = request.json
    if not data or 'part' not in data:
        return jsonify({'message': 'Не указана часть автомобиля для выделения.'}), 400

    part = data['part']

    if not current_image_path:
        return jsonify({'message': 'Файл еще не загружен.'}), 400

    try:
        processed_image_path, rect = analyze_and_highlight(current_image_path, part)

        if not rect:
            return jsonify({'message': 'Выбранная часть автомобиля не найдена.'}), 404

        return jsonify({
            'message': 'Область успешно выделена!',
            'processed_image_url': '/' + processed_image_path,
            'rect': rect  # Координаты выделенной области
        })
    except Exception as e:
        return jsonify({'message': f'Ошибка при выделении области: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(app.static_folder, 'css'), filename)


if __name__ == '__main__':
    app.run(debug=True)
