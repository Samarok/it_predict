import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, mean_absolute_error,
                             mean_squared_error)
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
import os

# 1. Загрузка данных
df = pd.read_csv('data/projects_dataset.csv')

# 2. Определение признаков и целевых переменных
feature_columns = ['domain', 'client_type', 'allocated_time', 'budget_adequacy',
                   'team_size_adequacy', 'methodology', 'tz_quality',
                   'stakeholder_involvement', 'risk_skill_gap']
X = df[feature_columns]
y_class = df['completed_on_time']          # классификация
y_reg = df['delay_percentage']             # регрессия

# 3. Указание категориальных признаков для CatBoost
categorical_features = ['domain', 'client_type', 'methodology']

# 4. Разделение на train/test
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
)

# 5. Обучение классификатора
clf = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_features,
    verbose=100,
    random_seed=42
)
clf.fit(X_train, y_class_train)

# 6. Оценка классификатора
y_class_pred = clf.predict(X_test)
y_class_proba = clf.predict_proba(X_test)[:, 1]

print("Классификация (completed_on_time):")
print(f"Accuracy:  {accuracy_score(y_class_test, y_class_pred):.4f}")
print(f"Precision: {precision_score(y_class_test, y_class_pred):.4f}")
print(f"Recall:    {recall_score(y_class_test, y_class_pred):.4f}")
print(f"F1-score:  {f1_score(y_class_test, y_class_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_class_test, y_class_proba):.4f}")

# 7. Обучение регрессора
reg = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_features,
    verbose=100,
    random_seed=42
)
reg.fit(X_train, y_reg_train)

# 8. Оценка регрессора
y_reg_pred = reg.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
# sMAPE (симметричная средняя абсолютная процентная ошибка)
smape = 100 * np.mean(2 * np.abs(y_reg_test - y_reg_pred) / (np.abs(y_reg_test) + np.abs(y_reg_pred)))

print("\nРегрессия (delay_percentage):")
print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"sMAPE: {smape:.2f}%")

# 9. Сохранение моделей
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/catboost_classifier.pkl')
joblib.dump(reg, 'models/catboost_regressor.pkl')
print("\nМодели сохранены в папку 'models/'")

# 10. (Опционально) Визуализация важности признаков
import matplotlib.pyplot as plt

importance = clf.get_feature_importance()
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, importance)
plt.xlabel('Важность')
plt.title('Важность признаков (классификатор)')
plt.tight_layout()
plt.savefig('models/feature_importance.png')
plt.show()