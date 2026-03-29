import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, mean_absolute_error,
                             mean_squared_error, confusion_matrix, median_absolute_error)
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
import os
import matplotlib.pyplot as plt

# 1. Loading data
df = pd.read_csv('data/projects_dataset.csv')

# 2. Define features and target variables
feature_columns = ['domain', 'client_type', 'allocated_time', 'budget_adequacy',
                   'team_size_adequacy', 'methodology', 'tz_quality',
                   'stakeholder_involvement', 'risk_skill_gap']
X = df[feature_columns]
y_class = df['completed_on_time']          # classification

# Process regression target: negative delays (early delivery) = 0
# Early delivery is considered "on time", we only care about delays
y_reg_raw = df['delay_percentage'].values
y_reg = np.maximum(y_reg_raw, 0)  # All negative -> 0
y_reg = pd.Series(y_reg, index=df.index)

# Check class imbalance for scale_pos_weight
class_counts = y_class.value_counts()
success_count = class_counts.get(True, 0)    # "successful" projects (minority class)
delay_count = class_counts.get(False, 0)     # "delays" (majority class)
# scale_pos_weight = weight for minority class (successful projects)
scale_pos_weight = delay_count / success_count if success_count > 0 else 1.0

print("=" * 60)
print("DATA PREPARATION")
print("=" * 60)
print(f"\nTarget variable distribution:")
print(f"  Successful projects (completed_on_time=True):  {success_count} ({success_count/len(y_class)*100:.1f}%)")
print(f"  Delays (completed_on_time=False):              {delay_count} ({delay_count/len(y_class)*100:.1f}%)")
print(f"  scale_pos_weight for balancing:                {scale_pos_weight:.2f}")

# Target distribution analysis for regression
print("\n" + "-" * 40)
print("Target distribution analysis (regression):")
print(f"  Projects with delay=0%: {(y_reg==0).mean()*100:.1f}%")
print(f"  Projects with delay>0%: {(y_reg>0).mean()*100:.1f}%")
print(f"  Median delay (only >0): {np.median(y_reg[y_reg>0]):.2f}%")
print(f"  Mean delay (only >0): {np.mean(y_reg[y_reg>0]):.2f}%")
print(f"  Maximum delay: {y_reg.max():.2f}%")
print(f"  Minimum delay: {y_reg.min():.2f}%")
print(f"  Negative delays (early delivery): {(y_reg_raw < 0).sum()} ({(y_reg_raw < 0).mean()*100:.1f}%)")
print(f"  After processing: all negative -> 0")

# 3. Categorical features for CatBoost
categorical_features = ['domain', 'client_type', 'methodology']

# ============================================================
# Cross-Validation
# ============================================================
print("=" * 60)
print("CROSS-VALIDATION (5-Fold)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation for classifier
print("\n1. Cross-Validation for Classifier:")
print("-" * 40)

cv_scores = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'roc_auc': []
}

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_class), 1):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y_class.iloc[train_idx]
    y_val_fold = y_class.iloc[val_idx]

    # Enhanced regularization to prevent overfitting
    clf_fold = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=15,
        min_data_in_leaf=20,
        subsample=0.65,
        colsample_bylevel=0.65,
        cat_features=categorical_features,
        scale_pos_weight=scale_pos_weight,
        verbose=False,
        random_seed=42,
        early_stopping_rounds=50
    )
    clf_fold.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold))

    y_pred_fold = clf_fold.predict(X_val_fold)
    # P(True) = P(success), we need P(delay) = 1 - P(success)
    y_proba_fold_delay = 1 - clf_fold.predict_proba(X_val_fold)[:, 1]
    y_val_fold_delay = (y_val_fold == False).astype(int)
    y_pred_fold_delay = (y_pred_fold == False).astype(int)

    cv_scores['accuracy'].append(accuracy_score(y_val_fold_delay, y_pred_fold_delay))
    cv_scores['precision'].append(precision_score(y_val_fold_delay, y_pred_fold_delay, zero_division=0))
    cv_scores['recall'].append(recall_score(y_val_fold_delay, y_pred_fold_delay, zero_division=0))
    cv_scores['f1'].append(f1_score(y_val_fold_delay, y_pred_fold_delay, zero_division=0))
    cv_scores['roc_auc'].append(roc_auc_score(y_val_fold_delay, y_proba_fold_delay))

    print(f"Fold {fold}: Acc={cv_scores['accuracy'][-1]:.4f}, "
          f"Recall={cv_scores['recall'][-1]:.4f}, "
          f"ROC-AUC={cv_scores['roc_auc'][-1]:.4f}")

print("\nCV Results (Classification):")
print(f"  Accuracy:  {np.mean(cv_scores['accuracy']):.4f} (+/- {np.std(cv_scores['accuracy']):.4f})")
print(f"  Precision: {np.mean(cv_scores['precision']):.4f} (+/- {np.std(cv_scores['precision']):.4f})")
print(f"  Recall:    {np.mean(cv_scores['recall']):.4f} (+/- {np.std(cv_scores['recall']):.4f})")
print(f"  F1-score:  {np.mean(cv_scores['f1']):.4f} (+/- {np.std(cv_scores['f1']):.4f})")
print(f"  ROC-AUC:   {np.mean(cv_scores['roc_auc']):.4f} (+/- {np.std(cv_scores['roc_auc']):.4f})")

# 4. Split into train/test
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
)

# 5. Threshold tuning on validation
print("\n" + "-" * 40)
print("Threshold tuning on validation:")

# Split validation from train for threshold tuning
X_train_clf, X_val_clf, y_class_train_clf, y_class_val_clf = train_test_split(
    X_train, y_class_train, test_size=0.2, random_state=42, stratify=y_class_train
)

clf_tune = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=15,
    min_data_in_leaf=20,
    subsample=0.65,
    colsample_bylevel=0.65,
    cat_features=categorical_features,
    scale_pos_weight=scale_pos_weight,
    verbose=False,
    random_seed=42,
    early_stopping_rounds=50
)
clf_tune.fit(X_train_clf, y_class_train_clf, eval_set=(X_val_clf, y_class_val_clf))

y_val_proba = clf_tune.predict_proba(X_val_clf)[:, 1]
y_val_proba_delay = 1 - y_val_proba

# Find optimal threshold
best_threshold = 0.5
best_f1 = 0
for threshold in [0.25, 0.3, 0.35, 0.4, 0.5]:
    y_pred_delay = (y_val_proba_delay >= threshold).astype(int)
    y_true_delay = (y_class_val_clf == False).astype(int)
    f1 = f1_score(y_true_delay, y_pred_delay, zero_division=0)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold} (F1={best_f1:.4f})")

# 6. Train classifier
clf = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=15,
    min_data_in_leaf=20,
    subsample=0.65,
    colsample_bylevel=0.65,
    cat_features=categorical_features,
    scale_pos_weight=scale_pos_weight,
    verbose=100,
    random_seed=42,
    early_stopping_rounds=50
)
clf.fit(X_train, y_class_train, eval_set=(X_test, y_class_test))

# 7. Evaluate classifier
y_class_proba = clf.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("2. Test Set Evaluation")
print("=" * 60)

# P(True) = P(success), P(delay) = 1 - P(success)
y_class_proba_delay = 1 - y_class_proba

# Apply optimal threshold for "delay" class
y_class_pred_delay = (y_class_proba_delay >= best_threshold).astype(int)
y_class_test_delay = (y_class_test == False).astype(int)

print("\nClassification (project delays):")
print(f"  Accuracy:  {accuracy_score(y_class_test_delay, y_class_pred_delay):.4f}")
print(f"  Precision: {precision_score(y_class_test_delay, y_class_pred_delay, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_class_test_delay, y_class_pred_delay, zero_division=0):.4f}")
print(f"  F1-score:  {f1_score(y_class_test_delay, y_class_pred_delay, zero_division=0):.4f}")

# ROC-AUC for "delay" class
roc_auc_delay = roc_auc_score(y_class_test_delay, y_class_proba_delay)
print(f"  ROC-AUC:   {roc_auc_delay:.4f}")

# Confusion matrix
cm = confusion_matrix(y_class_test_delay, y_class_pred_delay)
print("\nConfusion Matrix:")
print("            Predicted")
print("            On-time  Delay")
print(f"Actual On-time:  {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Delay:    {cm[1,0]:4d}  {cm[1,1]:4d}")

# Business metrics
total_delays = y_class_test_delay.sum()
predicted_delays = y_class_pred_delay.sum()
true_positives = cm[1, 1]
false_positives = cm[0, 1]
false_negatives = cm[1, 0]

print("\n" + "-" * 40)
print("Business Metrics:")
print(f"  Delays detected: {true_positives} out of {total_delays} ({true_positives/total_delays*100 if total_delays > 0 else 0:.1f}% recall)")
print(f"  False alarms: {false_positives} ({false_positives/predicted_delays*100 if predicted_delays > 0 else 0:.1f}% of predicted delays)")
print(f"  Missed delays: {false_negatives} ({false_negatives/total_delays*100 if total_delays > 0 else 0:.1f}%)")

# 8. Train regressor
print("\n" + "=" * 60)
print("3. Regression Model")
print("=" * 60)

reg = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=5,
    l2_leaf_reg=10,
    min_data_in_leaf=10,
    cat_features=categorical_features,
    verbose=False,
    random_seed=42,
    early_stopping_rounds=50
)
reg.fit(X_train, y_reg_train.values, eval_set=(X_test, y_reg_test.values))
y_reg_pred = reg.predict(X_test)

mae_pct = mean_absolute_error(y_reg_test.values, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test.values, y_reg_pred))
hit_rate_15 = (np.abs(y_reg_test.values - y_reg_pred) <= 15).mean() * 100
print(f"  MAE: {mae_pct:.4f}%")
print(f"  RMSE: {rmse:.4f}")
print(f"  Hit Rate <=15%: {hit_rate_15:.1f}%")

# Metrics for projects with delay
mask_delayed_test = y_reg_test.values > 0
mae_delayed = mean_absolute_error(y_reg_test.values[mask_delayed_test], y_reg_pred[mask_delayed_test])
hit_rate_15_delayed = (np.abs(y_reg_test.values[mask_delayed_test] - y_reg_pred[mask_delayed_test]) <= 15).mean() * 100

print(f"\nModel: CatBoost (MAE={mae_pct:.4f}%)")

# 9. Save models
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/catboost_classifier.pkl')
joblib.dump(reg, 'models/catboost_regressor.pkl')
print(f"\nModels saved to 'models/' folder")

# 10. Detailed regression metrics

# MAE in absolute weeks
allocated_test = df.loc[X_test.index, 'allocated_time'].values
actual_test_weeks = allocated_test * (1 + y_reg_test.values / 100)
predicted_test_weeks = allocated_test * (1 + y_reg_pred / 100)
mae_weeks = mean_absolute_error(actual_test_weeks, predicted_test_weeks)

# Hit rate metric
errors = np.abs(y_reg_test.values - y_reg_pred)
hit_rate_15 = (errors <= 15).mean() * 100
hit_rate_10 = (errors <= 10).mean() * 100

# Error only for projects with ACTUAL delay (y_test > 0)
delayed_mask = y_reg_test.values > 0
mae_delayed = mean_absolute_error(
    y_reg_test.values[delayed_mask],
    y_reg_pred[delayed_mask]
) if delayed_mask.sum() > 0 else 0

# MedAE (median absolute error)
medae = median_absolute_error(y_reg_test.values, y_reg_pred)

print(f"\nModel: CatBoost")
print(f"  MAE (percentage):     {mae_pct:.4f}%")
print(f"  MAE (weeks):          {mae_weeks:.2f} weeks")
print(f"  MAE (delay>0):        {mae_delayed:.4f}%")
print(f"  RMSE:                 {rmse:.4f}")
print(f"  MedAE (median):       {medae:.4f}%")
print(f"  Hit Rate <=10%: {hit_rate_10:.1f}%")
print(f"  Hit Rate <=15%: {hit_rate_15:.1f}%")

# 11. Overfitting check

# Use same parameters as main model
clf_check = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=15,
    min_data_in_leaf=20,
    subsample=0.65,
    colsample_bylevel=0.65,
    cat_features=categorical_features,
    scale_pos_weight=scale_pos_weight,
    verbose=False,
    random_seed=42,
    early_stopping_rounds=50
)
clf_check.fit(X_train, y_class_train, eval_set=(X_test, y_class_test))

evals_result = clf_check.get_evals_result()
train_loss = evals_result['learn']['Logloss']
val_loss = evals_result['validation']['Logloss']

print(f"Final Logloss on train: {train_loss[-1]:.4f}")
print(f"Final Logloss on validation: {val_loss[-1]:.4f}")
logloss_diff = abs(train_loss[-1] - val_loss[-1])
print(f"Difference: {logloss_diff:.4f}")

print(f"\nOverfitting check: {logloss_diff:.4f} (acceptable range: <0.15)")

# 12. Feature importance

importance = clf.get_feature_importance()
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': importance
}).sort_values('importance', ascending=False)

print(feature_importance.to_string(index=False))

# Feature correlation with target
print("\nFeature correlation with target (delays):")
print("-" * 40)
df_analysis = df.copy()
df_analysis['target_delay'] = (df['completed_on_time'] == False).astype(int)

correlations = {}
for feat in feature_columns:
    if df_analysis[feat].dtype in ['float64', 'int64']:
        corr = df_analysis[feat].corr(df_analysis['target_delay'])
        correlations[feat] = corr
        print(f"  {feat}: {corr:.4f}")

max_imp = feature_importance['importance'].max()
max_feature = feature_importance.iloc[0]['feature']

print(f"\nMaximum feature importance: {max_imp:.3f} ({max_feature})")

# 13. Final summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
Classification (project delays):
  Cross-Validation Accuracy:  {np.mean(cv_scores['accuracy']):.4f} (+/- {np.std(cv_scores['accuracy']):.4f})
  Cross-Validation Recall:    {np.mean(cv_scores['recall']):.4f} (+/- {np.std(cv_scores['recall']):.4f})
  Cross-Validation ROC-AUC:   {np.mean(cv_scores['roc_auc']):.4f}
  Test Set Accuracy:          {accuracy_score(y_class_test_delay, y_class_pred_delay):.4f}
  Test Set Recall:            {recall_score(y_class_test_delay, y_class_pred_delay, zero_division=0):.4f}
  Test Set Precision:         {precision_score(y_class_test_delay, y_class_pred_delay, zero_division=0):.4f}
  Test Set ROC-AUC:           {roc_auc_delay:.4f}
  Optimal Threshold:          {best_threshold}

Regression (delay_percentage):
  Model:                      CatBoost
  MAE (percentage):           {mae_pct:.4f}%
  MAE (weeks):                {mae_weeks:.2f} weeks
  MAE (delay>0):              {mae_delayed:.4f}%
  RMSE:                       {rmse:.4f}
  MedAE:                      {medae:.4f}%
  Hit Rate <=10%:             {hit_rate_10:.1f}%
  Hit Rate <=15%:             {hit_rate_15:.1f}%

Target Metrics:
  Classification Recall:      {recall_score(y_class_test_delay, y_class_pred_delay, zero_division=0):.4f} (target: >=0.70)
  Regression MAE:             {mae_pct:.4f}% (median error: {medae:.2f}%)
  Hit Rate <=15%:             {hit_rate_15:.1f}% (for delayed projects: {hit_rate_15_delayed:.1f}%)
  Overfitting:                {logloss_diff:.4f}
""")

# 14. Feature importance visualization
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance (Classifier)')
plt.tight_layout()
plt.savefig('models/feature_importance.png')
plt.show()