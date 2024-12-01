from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image


def process_image(image_path):
    try:
        
        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output_image = image.copy()
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

        output_path = image_path.replace(".jpg", "_processed.jpg")
        cv2.imwrite(output_path, output_image)

        areas = [cv2.contourArea(c) for c in contours]

        return {
            "processed_image_path": output_path,
            "areas": areas
        }
    except Exception as e:
        return {"error": str(e)}


def auto_select_area(image_path, x, y):
    """
    Выделяет объект на изображении, ближайший к заданной точке (x, y).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Не удалось загрузить изображение.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        closest_contour = None
        min_distance = float('inf')

        for contour in contours:
            for point in contour:
                px, py = point[0]
                distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour

        if closest_contour is None:
            raise RuntimeError("Не найдено подходящих объектов для выделения.")

        cv2.drawContours(img, [closest_contour], -1, (0, 0, 255), 2)  # Красный контур

        output_path = "static/processed_image.png"
        cv2.imwrite(output_path, img)

        x, y, w, h = cv2.boundingRect(closest_contour)

        return output_path, (x, y, w, h)  # Возвращаем путь и координаты области
    except Exception as e:
        raise RuntimeError(f"Ошибка авто-выделения: {str(e)}")


def highlight_front_door(image_path, x, y):
    """
    Выделяет переднюю дверь на изображении.
    """
    img = cv2.imread(image_path)
    rect = (50, 100, 200, 300)  # Пример ограничивающего прямоугольника
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    processed_image_path = "static/processed_image_front_door.png"
    cv2.imwrite(processed_image_path, img)
    return processed_image_path, rect


def highlight_hood(image_path, x, y):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Не удалось загрузить изображение.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Найдено контуров: {len(contours)}")

        for contour in contours:
            x_min, y_min, width, height = cv2.boundingRect(contour)
            if y_min < img.shape[0] // 3:  
                cv2.rectangle(img, (x_min, y_min), (x_min + width, y_min + height), (0, 0, 255), 3)
                processed_image_path = "static/processed_image_hood.png"
                if not cv2.imwrite(processed_image_path, img):
                    raise RuntimeError("Не удалось сохранить обработанное изображение.")
                return processed_image_path, (x_min, y_min, width, height)

        return None, None
    except Exception as e:
        raise RuntimeError(f"Ошибка выделения капота: {str(e)}")


def analyze_and_highlight(image_path, part):
    """
    Анализирует изображение автомобиля и выделяет выбранную часть (например, капот).
    """
    try:
        model = YOLO("yolov8n.pt")  

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Не удалось загрузить изображение.")

        results = model(image_path)
        detections = results[0].boxes

        part_to_class = {
            "front_door": ["car door", "door"],
            "hood": ["hood", "bonnet"],  # Возможные метки для капота
            "trunk": ["trunk", "boot"]
        }
        target_classes = part_to_class.get(part, [])

        highlighted = False
        for box in detections:
            cls = results[0].names[int(box.cls)]
            if cls in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy)

                if part == "hood" and y1 > img.shape[0] * 0.5:
                    continue

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                highlighted = True

                output_path = f"static/processed_{part}.png"
                cv2.imwrite(output_path, img)
                return output_path, (x1, y1, x2 - x1, y2 - y1)

        if not highlighted:
            return None, None

    except Exception as e:
        raise RuntimeError(f"Ошибка анализа изображения: {str(e)}")



def highlight_trunk(image_path, x, y):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Не удалось загрузить изображение.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x_min, y_min, width, height = cv2.boundingRect(contour)
            if y_min > img.shape[0] * 2 // 3:  # Фильтр для нижней трети изображения
                cv2.rectangle(img, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 3)
                processed_image_path = "static/processed_image_trunk.png"
                cv2.imwrite(processed_image_path, img)
                return processed_image_path, (x_min, y_min, width, height)

        return None, None
    except Exception as e:
        raise RuntimeError(f"Ошибка выделения крышки багажника: {str(e)}")

def detect_hood(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Не удалось загрузить изображение.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = img.shape[:2]
        hood_contours = []
        for contour in contours:
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            if y < h // 2 and w_contour > w * 0.3:  # Фильтруем большие объекты в верхней половине
                hood_contours.append(contour)

        if not hood_contours:
            return None, None  

        largest_contour = max(hood_contours, key=cv2.contourArea)
        x, y, w_contour, h_contour = cv2.boundingRect(largest_contour)
        cv2.rectangle(img, (x, y), (x + w_contour, y + h_contour), (0, 255, 0), 3)

        processed_image_path = "static/processed_hood.png"
        cv2.imwrite(processed_image_path, img)

        return processed_image_path, (x, y, w_contour, h_contour)
    except Exception as e:
        raise RuntimeError(f"Ошибка обработки изображения: {str(e)}")
