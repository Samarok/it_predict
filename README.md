# IT Project Delay Prediction

Machine learning project for predicting IT project delays using CatBoost models.

## Project Structure

- `synthetic_dataset_generator/generate_dataset.py` — generates synthetic dataset
- `model_training/train_model.py` — trains classification and regression models
- `model_training/predict_demo.py` — tests trained models

## Quick Start

### 1. Install Dependencies

```bash
cd synthetic_dataset_generator
pip install -r requirements.txt

cd ../model_training
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
cd synthetic_dataset_generator
python generate_dataset.py
```

### 3. Train Models

```bash
cd model_training
python train_model.py
```

### 4. Test Predictions

```bash
cd model_training
python predict_demo.py
```
