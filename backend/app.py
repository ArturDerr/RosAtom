from flask import Flask, request, jsonify, send_from_directory, render_template, send_file
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from utils.image_processing import process_image, auto_select_area, highlight_front_door, highlight_trunk, \
    highlight_hood, analyze_and_highlight, detect_hood

app = Flask(__name__, template_folder='../frontend/public', static_folder='../frontend/public')

# Конфигурация
model = YOLO("yolov8n.pt")
PROCESSED_FOLDER = './processed_files'
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


def is_car_detected(image_path):
    """
    Проверяет, содержит ли изображение автомобиль, используя YOLOv8.
    """
    # Выполняем детекцию объектов
    results = model(image_path)

    # Проверяем, есть ли класс "car" в результатах
    for box in results[0].boxes:
        cls = int(box.cls[0])  # Индекс класса объекта
        if model.names[cls].lower() == "car":
            return True  # Если найден автомобиль
    return False  # Если автомобили не найдены


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'Файл не найден.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'Имя файла пустое.'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Проверяем, есть ли автомобиль на фото
    if not is_car_detected(filepath):
        return jsonify({'message': 'На фото не обнаружено автомобиля.'}), 400

    return jsonify({
        'message': 'Файл успешно загружен.',
        'original_image_url': f'/uploads/{file.filename}'
    })


@app.route('/select_area', methods=['POST'])
def select_area():
    data = request.json
    if not data or 'part' not in data:
        return jsonify({'message': 'Не указана часть автомобиля для выделения.'}), 400

    part = data['part']

    if not current_image_path:
        return jsonify({'message': 'Файл еще не загружен.'}), 400

    try:
        if part == "hood":
            processed_image_path, rect = detect_hood(current_image_path)
        else:
            return jsonify({'message': f'Детекция для части {part} не поддерживается.'}), 400

        if not rect:
            return jsonify({'message': 'Капот не найден.'}), 404

        return jsonify({
            'message': 'Область успешно выделена!',
            'processed_image_url': '/' + processed_image_path,
            'rect': rect  # Координаты выделенной области
        })
    except Exception as e:
        return jsonify({'message': f'Ошибка при выделении области: {str(e)}'}), 500


@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    if not data:
        return jsonify({'message': 'Нет данных для расчета.'}), 400

    try:
        # Параметры от пользователя
        torch_width = float(data.get('torchWidth', 0.1))
        torch_extension = float(data.get('torchExtension', 0.08))
        layers = int(data.get('layers', 2))
        consumption = float(data.get('consumption', 0.3))
        paint_cost = float(data.get('paintCost', 0))
        hour_cost = float(data.get('hourCost', 0))
        rect = data.get('rect', {})
        front_door_area = float(data.get('frontDoorArea', 0.8))  # м²
        trunk_area = float(data.get('trunkArea', 0.9))  # м²
        hood_area = float(data.get('hoodArea', 1.2))  # м²

        # Размеры выделенной области (пиксели)
        rect_width = float(rect.get('width', 0))
        rect_height = float(rect.get('height', 0))
        if rect_width <= 0 or rect_height <= 0:
            return jsonify({'message': 'Некорректные размеры выделенной области.'}), 400

        # Эталонный элемент (например, дверь)
        reference_width_pixels = 200  # Замените на реальную ширину в пикселях
        reference_height_pixels = 400  # Замените на реальную высоту в пикселях
        reference_area_m2 = front_door_area  # Эталонная площадь двери (м²)

        # Расчёт площади выделенной области
        physical_area = reference_area_m2 * (
                (rect_width * rect_height) /
                (reference_width_pixels * reference_height_pixels)
        )

        # Расход ЛКМ
        total_paint = physical_area * layers * consumption
        total_paint_cost = total_paint * paint_cost
        total_labor_cost = physical_area * layers * hour_cost
        total_cost = total_paint_cost + total_labor_cost

        # Генерация отчёта
        report_path = os.path.join(UPLOAD_FOLDER, "report.txt")
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write("Отчет о расчете физических параметров\n")
            report_file.write("=" * 40 + "\n")
            report_file.write(f"Ширина факела: {torch_width} м\n")
            report_file.write(f"Вылет факела за границы: {torch_extension} м\n")
            report_file.write(f"Количество слоев: {layers}\n")
            report_file.write(f"Расход ЛКМ на 1 слой: {consumption} л/м²\n")
            report_file.write(f"Площадь передней двери: {front_door_area} м²\n")
            report_file.write(f"Площадь крышки багажника: {trunk_area} м²\n")
            report_file.write(f"Площадь капота: {hood_area} м²\n")
            report_file.write("\nРезультаты расчета:\n")
            report_file.write(f"Физическая площадь: {round(physical_area, 2)} м²\n")
            report_file.write(f"Общий расход ЛКМ: {round(total_paint, 2)} л\n")
            report_file.write(f"Стоимость ЛКМ: {round(total_paint_cost, 2)} руб.\n")
            report_file.write(f"Стоимость трудоёмкости: {round(total_labor_cost, 2)} руб.\n")
            report_file.write(f"Общая стоимость: {round(total_cost, 2)} руб.\n")

        return send_file(report_path, as_attachment=True)

    except Exception as e:
        return jsonify({'message': f'Ошибка при расчете: {str(e)}'}), 500


@app.route('/locales/<filename>')
def get_translation_file(filename):
    filepath = os.path.join('locales', filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'message': 'Файл переводов не найден'}), 404


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(app.static_folder, 'css'), filename)


if __name__ == '__main__':
    app.run(debug=True)
