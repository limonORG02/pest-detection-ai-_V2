import os
import shutil
import random

RAW_DIR = "data/raw"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
SPLIT_RATIO = 0.8

def prepare():
    for cls in os.listdir(RAW_DIR):
        cls_path = os.path.join(RAW_DIR, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [f for f in os.listdir(cls_path)
                  if os.path.isfile(os.path.join(cls_path, f))]

        random.shuffle(images)
        split = int(len(images) * SPLIT_RATIO)

        train_imgs = images[:split]
        test_imgs = images[split:]

        for folder, imgs in [(TRAIN_DIR, train_imgs), (TEST_DIR, test_imgs)]:
            dst = os.path.join(folder, cls)
            os.makedirs(dst, exist_ok=True)

            for img in imgs:
                shutil.copy(
                    os.path.join(cls_path, img),
                    os.path.join(dst, img)
                )

    print(" Данные подготовлены")

if __name__ == "__main__":
    prepare()
