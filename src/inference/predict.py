import sys
import cv2
import numpy as np
import tensorflow as tf

from src.utils.config import *

if len(sys.argv) < 2:
    print("Использование: python predict.py image.jpg")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

img = cv2.imread(sys.argv[1])
img = cv2.resize(img, IMAGE_SIZE)
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]

if pred > 0.5:
    print("Обнаружен вредитель")
else:
    print("Вредителей не найдено")
