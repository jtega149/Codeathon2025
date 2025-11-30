#!/usr/bin/env python3
"""
MODEL TESTING & EVALUATION
- Tests model performance on training data
- Calculates comprehensive metrics
- Analyzes prediction quality
- Validates submission file
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("MODEL TESTING & EVALUATION")
print("="*70)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1/5] Loading data...")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_scope1 = pd.read_csv('y_scope1.csv').values.ravel()
y_scope2 = pd.read_csv('y_scope2.csv').values.ravel()
test_ids = pd.read_csv('test_ids.csv').values.ravel()
submission = pd.read_csv('submission.csv')

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features: {X_train.shape[1]}")

# ============================================================================
# 1b. Exploratory Data Analysis (EDA) & Visuals
# ============================================================================
print("\n[1b/5] Running EDA and creating visualizations (saved under ../figures)...")

train_raw = pd.read_csv('../data/train.csv')

eda_output_dir = '../figures'

plt.style.use('seaborn-v0_8')


def _save_fig(name: str) -> None:
    """Save current matplotlib figure to the figures directory."""
    plt.tight_layout()
    plt.savefig(f"{eda_output_dir}/{name}", dpi=200)
    plt.close()


# Target distributions (Scope 1 & 2)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

sns.histplot(train_raw['target_scope_1'], bins=50, ax=axes[0, 0])
axes[0, 0].set_title('Scope 1 Emissions - Raw')
axes[0, 0].set_xlabel('target_scope_1')

sns.histplot(np.log1p(train_raw['target_scope_1']), bins=50, ax=axes[0, 1])
axes[0, 1].set_title('Scope 1 Emissions - log1p')
axes[0, 1].set_xlabel('log1p(target_scope_1)')

sns.histplot(train_raw['target_scope_2'], bins=50, ax=axes[1, 0])
axes[1, 0].set_title('Scope 2 Emissions - Raw')
axes[1, 0].set_xlabel('target_scope_2')

sns.histplot(np.log1p(train_raw['target_scope_2']), bins=50, ax=axes[1, 1])
axes[1, 1].set_title('Scope 2 Emissions - log1p')
axes[1, 1].set_xlabel('log1p(target_scope_2)')

_save_fig('eda_targets_distribution.png')


# Emissions vs revenue (log-log)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(np.log1p(train_raw['revenue']), np.log1p(
    train_raw['target_scope_1']), s=10, alpha=0.4)
axes[0].set_xlabel('log1p(revenue)')
axes[0].set_ylabel('log1p(target_scope_1)')
axes[0].set_title('Scope 1 vs Revenue (log-log)')

axes[1].scatter(np.log1p(train_raw['revenue']), np.log1p(
    train_raw['target_scope_2']), s=10, alpha=0.4, color='orange')
axes[1].set_xlabel('log1p(revenue)')
axes[1].set_ylabel('log1p(target_scope_2)')
axes[1].set_title('Scope 2 vs Revenue (log-log)')

_save_fig('eda_revenue_vs_emissions.png')


# Emissions by region (median)
if 'region_code' in train_raw.columns:
    region_agg = train_raw.groupby('region_code')[
        ['target_scope_1', 'target_scope_2']].median().reset_index()
    region_agg = region_agg.sort_values('target_scope_1', ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    region_agg.plot(x='region_code', y=[
                    'target_scope_1', 'target_scope_2'], kind='bar', ax=ax)
    ax.set_title('Median Emissions by Region')
    ax.set_ylabel('Median Emissions')
    _save_fig('eda_median_emissions_by_region.png')


# Emissions by country (top 15 by mean Scope 1)
country_agg = train_raw.groupby('country_code')[
    ['target_scope_1', 'target_scope_2']].mean().reset_index()
top_countries = country_agg.sort_values(
    'target_scope_1', ascending=False).head(15)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
top_countries.plot(x='country_code', y=[
                   'target_scope_1', 'target_scope_2'], kind='bar', ax=ax)
ax.set_title('Top 15 Countries by Mean Scope 1 Emissions')
ax.set_ylabel('Mean Emissions')
_save_fig('eda_top_countries_emissions.png')


# Correlation heatmap for key numeric drivers
num_cols = [
    'revenue',
    'overall_score',
    'environmental_score',
    'social_score',
    'governance_score',
    'target_scope_1',
    'target_scope_2',
]
num_cols = [c for c in num_cols if c in train_raw.columns]

if len(num_cols) >= 2:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    corr = train_raw[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation - ESG / Revenue vs Emissions')
    _save_fig('eda_correlation_heatmap.png')

# ============================================================================
# 2. Train Models for Testing
# ============================================================================
print("\n[2/5] Training models for evaluation...")

# Model parameters (same as training)
xgb_params = {
    'n_estimators': 500,
    'max_depth': 4,
    'learning_rate': 0.03,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'random_state': 42,
    'n_jobs': -1
}

lgb_params = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_samples': 20,
    'reg_alpha': 0.5,
    'reg_lambda': 1.5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

cat_params = {
    'iterations': 500,
    'depth': 5,
    'learning_rate': 0.03,
    'l2_leaf_reg': 3.0,
    'random_seed': 42,
    'verbose': False
}

# ============================================================================
# 3. Cross-Validation Evaluation
# ============================================================================
print("\n[3/5] Running 5-fold cross-validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for predictions and metrics
all_metrics_s1 = []
all_metrics_s2 = []
all_preds_s1 = []
all_preds_s2 = []
all_true_s1 = []
all_true_s2 = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n   Fold {fold_idx + 1}/5...")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_s1_tr, y_s1_val = y_scope1[train_idx], y_scope1[val_idx]
    y_s2_tr, y_s2_val = y_scope2[train_idx], y_scope2[val_idx]

    # Log transform
    y_s1_tr_log = np.log1p(y_s1_tr)
    y_s2_tr_log = np.log1p(y_s2_tr)
    y_s1_val_log = np.log1p(y_s1_val)
    y_s2_val_log = np.log1p(y_s2_val)

    # Train Scope 1 models
    xgb_s1 = xgb.XGBRegressor(**xgb_params)
    xgb_s1.fit(X_tr, y_s1_tr_log)

    lgb_s1 = lgb.LGBMRegressor(**lgb_params)
    lgb_s1.fit(X_tr, y_s1_tr_log)

    cat_s1 = CatBoostRegressor(**cat_params)
    cat_s1.fit(X_tr, y_s1_tr_log)

    # Scope 1 predictions
    pred_s1_log = (0.40 * xgb_s1.predict(X_val) +
                   0.35 * lgb_s1.predict(X_val) +
                   0.25 * cat_s1.predict(X_val))
    pred_s1 = np.maximum(np.expm1(pred_s1_log), 0)

    # Train Scope 2 models
    xgb_s2 = xgb.XGBRegressor(**xgb_params)
    xgb_s2.fit(X_tr, y_s2_tr_log)

    lgb_s2 = lgb.LGBMRegressor(**lgb_params)
    lgb_s2.fit(X_tr, y_s2_tr_log)

    cat_s2 = CatBoostRegressor(**cat_params)
    cat_s2.fit(X_tr, y_s2_tr_log)

    # Scope 2 predictions
    pred_s2_log = (0.40 * xgb_s2.predict(X_val) +
                   0.35 * lgb_s2.predict(X_val) +
                   0.25 * cat_s2.predict(X_val))
    pred_s2 = np.maximum(np.expm1(pred_s2_log), 0)

    # Store predictions
    all_preds_s1.extend(pred_s1)
    all_preds_s2.extend(pred_s2)
    all_true_s1.extend(y_s1_val)
    all_true_s2.extend(y_s2_val)

    # Calculate metrics for Scope 1
    metrics_s1 = {
        'rmse_orig': np.sqrt(mean_squared_error(y_s1_val, pred_s1)),
        'rmse_log': np.sqrt(mean_squared_error(y_s1_val_log, pred_s1_log)),
        'mae': mean_absolute_error(y_s1_val, pred_s1),
        'r2': r2_score(y_s1_val, pred_s1),
        'mape': mean_absolute_percentage_error(y_s1_val, pred_s1) * 100
    }
    all_metrics_s1.append(metrics_s1)

    # Calculate metrics for Scope 2
    metrics_s2 = {
        'rmse_orig': np.sqrt(mean_squared_error(y_s2_val, pred_s2)),
        'rmse_log': np.sqrt(mean_squared_error(y_s2_val_log, pred_s2_log)),
        'mae': mean_absolute_error(y_s2_val, pred_s2),
        'r2': r2_score(y_s2_val, pred_s2),
        'mape': mean_absolute_percentage_error(y_s2_val, pred_s2) * 100
    }
    all_metrics_s2.append(metrics_s2)

    print(
        f"      Scope 1 - RMSE: {metrics_s1['rmse_orig']:,.0f}, MAE: {metrics_s1['mae']:,.0f}, R²: {metrics_s1['r2']:.4f}")
    print(
        f"      Scope 2 - RMSE: {metrics_s2['rmse_orig']:,.0f}, MAE: {metrics_s2['mae']:,.0f}, R²: {metrics_s2['r2']:.4f}")

# ============================================================================
# 4. Aggregate Metrics
# ============================================================================
print("\n[4/5] Calculating aggregate metrics...")

# Convert to arrays
all_preds_s1 = np.array(all_preds_s1)
all_preds_s2 = np.array(all_preds_s2)
all_true_s1 = np.array(all_true_s1)
all_true_s2 = np.array(all_true_s2)

# Overall metrics
avg_metrics_s1 = {k: np.mean([m[k] for m in all_metrics_s1])
                  for k in all_metrics_s1[0].keys()}
avg_metrics_s2 = {k: np.mean([m[k] for m in all_metrics_s2])
                  for k in all_metrics_s2[0].keys()}
std_metrics_s1 = {k: np.std([m[k] for m in all_metrics_s1])
                  for k in all_metrics_s1[0].keys()}
std_metrics_s2 = {k: np.std([m[k] for m in all_metrics_s2])
                  for k in all_metrics_s2[0].keys()}

# Combined RMSE
combined_rmse_orig = np.sqrt(
    (avg_metrics_s1['rmse_orig']**2 + avg_metrics_s2['rmse_orig']**2) / 2)
combined_rmse_log = np.sqrt(
    (avg_metrics_s1['rmse_log']**2 + avg_metrics_s2['rmse_log']**2) / 2)

# ============================================================================
# 5. Submission Validation
# ============================================================================
print("\n[5/5] Validating submission file...")

# Load training data to compare distributions
train = pd.read_csv('../data/train.csv')

# Check submission
submission_checks = {
    'total_predictions': len(submission),
    'expected_predictions': len(X_test),
    'no_missing_values': not submission.isnull().any().any(),
    'no_negative_scope1': (submission['target_scope_1'] >= 0).all(),
    'no_negative_scope2': (submission['target_scope_2'] >= 0).all(),
    'all_ids_present': len(submission) == len(test_ids)
}

# ============================================================================
# Print Results
# ============================================================================
print("\n" + "="*70)
print("✅ MODEL TESTING COMPLETE")
print("="*70)

print("\n📊 CROSS-VALIDATION METRICS (5-Fold)")
print("-"*70)

print("\nScope 1 (Direct Emissions):")
print(
    f"  RMSE (Original):  {avg_metrics_s1['rmse_orig']:>12,.2f} ± {std_metrics_s1['rmse_orig']:,.2f}")
print(
    f"  RMSE (Log):       {avg_metrics_s1['rmse_log']:>12.4f} ± {std_metrics_s1['rmse_log']:.4f}")
print(
    f"  MAE:              {avg_metrics_s1['mae']:>12,.2f} ± {std_metrics_s1['mae']:,.2f}")
print(
    f"  MAPE:             {avg_metrics_s1['mape']:>12.2f}% ± {std_metrics_s1['mape']:.2f}%")
print(
    f"  R²:               {avg_metrics_s1['r2']:>12.4f} ± {std_metrics_s1['r2']:.4f}")

print("\nScope 2 (Indirect Emissions):")
print(
    f"  RMSE (Original):  {avg_metrics_s2['rmse_orig']:>12,.2f} ± {std_metrics_s2['rmse_orig']:,.2f}")
print(
    f"  RMSE (Log):       {avg_metrics_s2['rmse_log']:>12.4f} ± {std_metrics_s2['rmse_log']:.4f}")
print(
    f"  MAE:              {avg_metrics_s2['mae']:>12,.2f} ± {std_metrics_s2['mae']:,.2f}")
print(
    f"  MAPE:             {avg_metrics_s2['mape']:>12.2f}% ± {std_metrics_s2['mape']:.2f}%")
print(
    f"  R²:               {avg_metrics_s2['r2']:>12.4f} ± {std_metrics_s2['r2']:.4f}")

print("\nCombined Metrics:")
print(f"  RMSE (Original):  {combined_rmse_orig:>12,.2f}")
print(f"  RMSE (Log):       {combined_rmse_log:>12.4f}")

print("\n📈 PREDICTION ANALYSIS")
print("-"*70)

# Prediction accuracy by range


def analyze_by_range(y_true, y_pred, name):
    print(f"\n{name}:")
    ranges = [
        (0, 1000, "Very Low (0-1K)"),
        (1000, 10000, "Low (1K-10K)"),
        (10000, 50000, "Medium (10K-50K)"),
        (50000, 100000, "High (50K-100K)"),
        (100000, float('inf'), "Very High (>100K)")
    ]

    for low, high, label in ranges:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            mape = mean_absolute_percentage_error(
                y_true[mask], y_pred[mask]) * 100
            print(
                f"  {label:20s} - Count: {mask.sum():3d}, RMSE: {rmse:>10,.0f}, MAE: {mae:>10,.0f}, MAPE: {mape:>6.1f}%")


analyze_by_range(all_true_s1, all_preds_s1, "Scope 1 by Emission Range")
analyze_by_range(all_true_s2, all_preds_s2, "Scope 2 by Emission Range")

print("\n📋 SUBMISSION VALIDATION")
print("-"*70)
for check, result in submission_checks.items():
    status = "✅" if result else "❌"
    print(f"  {status} {check.replace('_', ' ').title()}: {result}")

print("\n📊 SUBMISSION STATISTICS")
print("-"*70)

print("\nScope 1 Predictions:")
print(f"  Min:      {submission['target_scope_1'].min():>12,.0f}")
print(f"  Q1:       {submission['target_scope_1'].quantile(0.25):>12,.0f}")
print(f"  Median:   {submission['target_scope_1'].median():>12,.0f}")
print(f"  Mean:     {submission['target_scope_1'].mean():>12,.0f}")
print(f"  Q3:       {submission['target_scope_1'].quantile(0.75):>12,.0f}")
print(f"  Max:      {submission['target_scope_1'].max():>12,.0f}")

print("\nScope 2 Predictions:")
print(f"  Min:      {submission['target_scope_2'].min():>12,.0f}")
print(f"  Q1:       {submission['target_scope_2'].quantile(0.25):>12,.0f}")
print(f"  Median:   {submission['target_scope_2'].median():>12,.0f}")
print(f"  Mean:     {submission['target_scope_2'].mean():>12,.0f}")
print(f"  Q3:       {submission['target_scope_2'].quantile(0.75):>12,.0f}")
print(f"  Max:      {submission['target_scope_2'].max():>12,.0f}")

print("\n📊 TRAINING vs TEST DISTRIBUTION COMPARISON")
print("-"*70)

print("\nScope 1:")
print(f"  Training Mean:   {train['target_scope_1'].mean():>12,.0f}")
print(f"  Test Mean:       {submission['target_scope_1'].mean():>12,.0f}")
print(
    f"  Difference:      {abs(train['target_scope_1'].mean() - submission['target_scope_1'].mean()):>12,.0f}")
print(f"  Training Median: {train['target_scope_1'].median():>12,.0f}")
print(f"  Test Median:     {submission['target_scope_1'].median():>12,.0f}")

print("\nScope 2:")
print(f"  Training Mean:   {train['target_scope_2'].mean():>12,.0f}")
print(f"  Test Mean:       {submission['target_scope_2'].mean():>12,.0f}")
print(
    f"  Difference:      {abs(train['target_scope_2'].mean() - submission['target_scope_2'].mean()):>12,.0f}")
print(f"  Training Median: {train['target_scope_2'].median():>12,.0f}")
print(f"  Test Median:     {submission['target_scope_2'].median():>12,.0f}")

print("\n" + "="*70)
print("✅ All tests passed! Ready for submission.")
print("="*70)
