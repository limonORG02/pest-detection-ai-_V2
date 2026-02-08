from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template_string, request

from src.utils.config import IMAGE_SIZE, MODEL_PATH

app = Flask(__name__)

MODEL: Optional[tf.keras.Model] = None

HTML_TEMPLATE = """
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <title>Pest Detection AI</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; }
      .card { max-width: 520px; padding: 24px; border: 1px solid #ddd; border-radius: 8px; }
      .result { margin-top: 16px; padding: 12px; border-radius: 6px; background: #f5f5f5; }
      .error { color: #b00020; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Pest Detection AI</h1>
      <p>Загрузите изображение листа для определения наличия вредителей.</p>
      <form method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required />
        <button type="submit">Проверить</button>
      </form>
      {% if message %}
        <div class="result {{ 'error' if is_error else '' }}">{{ message }}</div>
      {% endif %}
    </div>
  </body>
</html>
"""


def get_model() -> tf.keras.Model:
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Модель не найдена по пути {MODEL_PATH}. Сначала обучите модель."
            )
        MODEL = tf.keras.models.load_model(MODEL_PATH)
    return MODEL


def prepare_image(file_bytes: bytes) -> np.ndarray:
    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось прочитать изображение.")
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    message = ""
    is_error = False
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            message = "Пожалуйста, выберите изображение."
            is_error = True
        else:
            try:
                image = prepare_image(file.read())
                pred = get_model().predict(image)[0][0]
                message = (
                    "Обнаружен вредитель" if pred > 0.5 else "Вредителей не найдено"
                )
            except Exception as exc:  # noqa: BLE001
                message = f"Ошибка обработки изображения: {exc}"
                is_error = True
    return render_template_string(HTML_TEMPLATE, message=message, is_error=is_error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
