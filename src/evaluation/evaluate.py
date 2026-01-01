import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.utils.config import *

model = tf.keras.models.load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_directory(
    "data/test",
    target_size=IMAGE_SIZE,
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

preds = model.predict(test_gen)
preds = (preds > 0.5).astype(int)

report = classification_report(
    test_gen.classes,
    preds,
    target_names=CLASSES
)

with open("reports/metrics.txt", "w") as f:
    f.write(report)

print("Метрики:")
print(report)
