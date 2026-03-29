import joblib
import pandas as pd

# Load models
clf = joblib.load('models/catboost_classifier.pkl')
reg = joblib.load('models/catboost_regressor.pkl')

# Example input data (can be replaced with input() for interactive use)
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

# Predictions
prob_delay = clf.predict_proba(new_project)[0, 1]  # probability of delay
pred_delay_percent = reg.predict(new_project)[0]

print(f"Probability of schedule delay: {prob_delay:.2%}")
print(f"Predicted delay: {pred_delay_percent:.2f}%")