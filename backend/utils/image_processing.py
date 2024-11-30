import cv2
import numpy as np
import os


def flood_fill_highlight(image_path, x, y):
    try:
        # Загружаем изображение
        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        # Создаем пустую маску
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Выполняем заливку
        flood_fill_flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        cv2.floodFill(image, mask, (x, y), (255, 255, 255), (10, 10, 10), (10, 10, 10), flood_fill_flags)

        # Создаем цветную маску
        mask = mask[1:-1, 1:-1]
        mask_rgb = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

        # Накладываем маску на оригинальное изображение
        highlighted = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)

        # Сохраняем результат
        result_path = image_path.replace(".jpg", "_highlighted.png")
        cv2.imwrite(result_path, highlighted)

        return result_path
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

