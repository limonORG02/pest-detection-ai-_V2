# Pest Detection AI (Updated Version)

Интеллектуальная система обнаружения вредителей
на основе сверточных нейронных сетей.

## Цель
Автоматизация выявления вредителей на листьях растений
для повышения эффективности сельского хозяйства.

## СТРУКТУРА ПРОЕКТА
```
pest-detection-ai-_V2/
├── data/
│   ├── raw/
│   │   ├── pests/
│   │   └── no_pests/
│   ├── train/
│   └── test/
├── models/
│   └── best_model.keras
├── reports/
│   └── metrics.txt
├── src/
│   ├── data/
│   │   └── prepare_data.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── inference/
│   │   └── predict.py
│   └── utils/
│       └── config.py
├── requirements.txt
├── run.sh
├── README.md
├── .gitignore
└── venv/

```

## Технологии
- Python
- TensorFlow / Keras
- EfficientNet (Transfer Learning)
- OpenCV
- Scikit-learn

## Установка и запуск

```bash
git clone <git@github.com:limonORG02/pest-detection-ai-_V2.git>
cd pest-detection-ai-update
./run.sh
```
## Предсказание
```
python src/inference/predict.py image.jpg
```
## Результаты

Метрики сохраняются в reports/metrics.txt

## Применение

Проект может быть использован в системах точного земледелия
и как основа для мобильных и web-приложений.

---
