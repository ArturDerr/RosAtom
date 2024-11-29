from PIL import Image, ImageOps


def process_image(image_path):
    """
    Обрабатывает изображение для подсчета площади выделенных объектов.

    Args:
        image_path (str): Путь к изображению.

    Returns:
        dict: Результаты обработки изображения.
    """
    try:
        # Открываем изображение
        image = Image.open(image_path).convert("L")  # Конвертируем в градации серого

        # Бинаризация (пороговое преобразование)
        threshold = 127
        binary_image = image.point(lambda x: 255 if x > threshold else 0, '1')

        # Подсчет белых пикселей (обрабатываемая площадь)
        pixel_data = binary_image.getdata()
        white_pixels = sum(1 for pixel in pixel_data if pixel == 255)

        # Размер изображения
        width, height = binary_image.size
        total_pixels = width * height

        # Возвращаем площади
        return {
            "total_pixels": total_pixels,
            "white_pixels": white_pixels,
            "white_area_percentage": (white_pixels / total_pixels) * 100
        }
    except Exception as e:
        return {"error": f"Ошибка обработки изображения: {str(e)}"}
