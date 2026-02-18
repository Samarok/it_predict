import joblib
import pandas as pd

# Загрузка моделей
clf = joblib.load('models/catboost_classifier.pkl')
reg = joblib.load('models/catboost_regressor.pkl')

# Пример ввода данных (можно заменить на input() для интерактива)
new_project = pd.DataFrame([{
    'domain': 'web_development',
    'client_type': 'large_corporate',
    'allocated_time': 20,
    'budget_adequacy': 1.2,
    'team_size_adequacy': 1.0,
    'methodology': 'agile',
    'tz_quality': 0.8,
    'stakeholder_involvement': 0.7,
    'risk_skill_gap': False
}])

# Предсказания
prob_delay = clf.predict_proba(new_project)[0, 1]  # вероятность срыва
pred_delay_percent = reg.predict(new_project)[0]

print(f"Вероятность срыва срока: {prob_delay:.2%}")
print(f"Прогнозируемая задержка: {pred_delay_percent:.2f}%")