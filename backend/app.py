from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import os
from utils.image_processing import process_image

# Инициализация Flask-приложения
app = Flask(__name__,
            template_folder='../frontend/public',  # Указываем, где искать шаблоны
            static_folder='../frontend/public')  # Указываем папку для статики (если нужно)

# Конфигурация
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Убедимся, что папка для загрузок существует
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

        # Обрабатываем изображение
        result = process_image(filepath)
        if "error" in result:
            return jsonify({'message': f"Ошибка: {result['error']}"}), 400

        # Подсчитываем площадь
        white_area_percentage = result.get("white_area_percentage", 0)

        # Возвращаем путь к изображению для отображения
        image_url = f"/uploads/{filename}"

        return jsonify({
            'message': f"Изображение загружено успешно! Площадь обработки: {white_area_percentage:.2f}%",
            'image_url': image_url  # Путь к изображению
        })
    else:
        return jsonify({'message': 'Ошибка: Неверный формат файла.'}), 400


# Статический маршрут для обслуживания загруженных изображений
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(app.static_folder, 'css'), filename)


if __name__ == '__main__':
    app.run(debug=True)
