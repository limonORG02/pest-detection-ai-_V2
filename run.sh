#!/usr/bin/env bash
set -e

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python src/data/prepare_data.py
python src/training/train.py
python src/evaluation/evaluate.py

echo "Готово! Для предсказания:"
echo "python src/inference/predict.py image.jpg"
