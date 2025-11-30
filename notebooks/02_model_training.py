#!/usr/bin/env python3
"""
ENSEMBLE MODEL TRAINING - TARGET RMSE: ~60k
- XGBoost + LightGBM + CatBoost ensemble
- Aggressive log transformation on targets
- Heavy regularization
- Weighted averaging of predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ENSEMBLE MODEL TRAINING - TARGET RMSE: ~60k")
print("="*70)

# ============================================================================
# 1. Load Engineered Features
# ============================================================================
print("\n[1/6] Loading engineered features...")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_scope1 = pd.read_csv('y_scope1.csv').values.ravel()
y_scope2 = pd.read_csv('y_scope2.csv').values.ravel()
test_ids = pd.read_csv('test_ids.csv').values.ravel()

print(f"   Features: {X_train.shape[1]}")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# ----------------------------------------------------------------------------
# 1b. Define a ULTRA-COMPACT core feature set for XGBoost-only comparison
# ----------------------------------------------------------------------------
print("\n[1b/6] Restricting to core feature set for XGBoost-only variant...")

core_features = [
    # scale / size
    'revenue',
    'log_revenue',

    # core ESG
    'overall_score',
    'environmental_score',

    # sector risk / diversification
    'high_emission_pct',
    'is_high_emission',
    'sector_entropy',

    # environmental behaviour
    'env_adj_sum',
    'env_adj_count',

    # interactions (if they exist)
    'log_revenue_x_high_emission',
    'log_revenue_x_environmental_score',
]

core_features = [f for f in core_features if f in X_train.columns]
print("   Using ULTRA-COMPACT feature set:")
for f in core_features:
    print("    -", f)

X_train_core = X_train[core_features].copy()
X_test_core = X_test[core_features].copy()

# ============================================================================
# 2. Define Ensemble Models with Heavy Regularization
# ============================================================================
print("\n[2/6] Configuring ensemble models...")

# XGBoost with heavy regularization
xgb_params = {
    'n_estimators': 500,
    'max_depth': 4,  # Shallow trees
    'learning_rate': 0.03,  # Slow learning
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,  # Heavy regularization
    'reg_alpha': 1.0,  # L1 regularization
    'reg_lambda': 2.0,  # L2 regularization
    'random_state': 42,
    'n_jobs': -1
}

# LightGBM with different regularization
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

# CatBoost with L2 leaf regularization
cat_params = {
    'iterations': 500,
    'depth': 5,
    'learning_rate': 0.03,
    'l2_leaf_reg': 3.0,  # Heavy L2 regularization
    'random_seed': 42,
    'verbose': False
}

print("   ✅ XGBoost configured (max_depth=4, heavy reg)")
print("   ✅ LightGBM configured (max_depth=5, medium reg)")
print("   ✅ CatBoost configured (L2 reg=3.0)")

# ----------------------------------------------------------------------------
# 2b. XGBoost-only configuration for core features with lighter regularization
# ----------------------------------------------------------------------------
print("\n[2b/6] Configuring XGBoost-only (core features, lighter regularization)...")

xgb_core_params = {
    'n_estimators': 600,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.95,
    'colsample_bytree': 0.95,
    'min_child_weight': 1.0,
    'reg_alpha': 0.0,
    'reg_lambda': 0.5,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
}

# ============================================================================
# 3. Train Ensemble for Scope 1
# ============================================================================
print("\n[3/6] Training Scope 1 ensemble...")

# CRITICAL: Log transform targets
y_scope1_log = np.log1p(y_scope1)

# Train XGBoost
print("   Training XGBoost...")
xgb_s1 = xgb.XGBRegressor(**xgb_params)
xgb_s1.fit(X_train, y_scope1_log)

# Train LightGBM
print("   Training LightGBM...")
lgb_s1 = lgb.LGBMRegressor(**lgb_params)
lgb_s1.fit(X_train, y_scope1_log)

# Train CatBoost
print("   Training CatBoost...")
cat_s1 = CatBoostRegressor(**cat_params)
cat_s1.fit(X_train, y_scope1_log)

# Cross-validation to evaluate
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
rmse_scores_log = []

for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_scope1[train_idx], y_scope1[val_idx]
    y_tr_log = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)

    # Train models
    xgb_temp = xgb.XGBRegressor(**xgb_params)
    xgb_temp.fit(X_tr, y_tr_log)

    lgb_temp = lgb.LGBMRegressor(**lgb_params)
    lgb_temp.fit(X_tr, y_tr_log)

    cat_temp = CatBoostRegressor(**cat_params)
    cat_temp.fit(X_tr, y_tr_log)

    # Ensemble predictions in log space
    pred_xgb_log = xgb_temp.predict(X_val)
    pred_lgb_log = lgb_temp.predict(X_val)
    pred_cat_log = cat_temp.predict(X_val)

    pred_ensemble_log = 0.40 * pred_xgb_log + \
        0.35 * pred_lgb_log + 0.25 * pred_cat_log

    # Convert to original space
    pred_xgb = np.expm1(pred_xgb_log)
    pred_lgb = np.expm1(pred_lgb_log)
    pred_cat = np.expm1(pred_cat_log)

    # Weighted average (XGB=40%, LGB=35%, CAT=25%)
    pred_ensemble = 0.40 * pred_xgb + 0.35 * pred_lgb + 0.25 * pred_cat
    pred_ensemble = np.maximum(pred_ensemble, 0)

    # Calculate RMSE in both spaces
    rmse_log = np.sqrt(mean_squared_error(y_val_log, pred_ensemble_log))
    rmse_orig = np.sqrt(mean_squared_error(y_val, pred_ensemble))

    rmse_scores_log.append(rmse_log)
    rmse_scores.append(rmse_orig)

print(
    f"   ✅ Scope 1 CV RMSE (Original): {np.mean(rmse_scores):,.2f} ± {np.std(rmse_scores):,.2f}")
print(
    f"   ✅ Scope 1 CV RMSE (Log):      {np.mean(rmse_scores_log):.4f} ± {np.std(rmse_scores_log):.4f}")

# ============================================================================
# 4. Train Ensemble for Scope 2
# ============================================================================
print("\n[4/6] Training Scope 2 ensemble...")

y_scope2_log = np.log1p(y_scope2)

# Train XGBoost
print("   Training XGBoost...")
xgb_s2 = xgb.XGBRegressor(**xgb_params)
xgb_s2.fit(X_train, y_scope2_log)

# Train LightGBM
print("   Training LightGBM...")
lgb_s2 = lgb.LGBMRegressor(**lgb_params)
lgb_s2.fit(X_train, y_scope2_log)

# Train CatBoost
print("   Training CatBoost...")
cat_s2 = CatBoostRegressor(**cat_params)
cat_s2.fit(X_train, y_scope2_log)

# Cross-validation
rmse_scores_s2 = []
rmse_scores_s2_log = []

for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_scope2[train_idx], y_scope2[val_idx]
    y_tr_log = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)

    xgb_temp = xgb.XGBRegressor(**xgb_params)
    xgb_temp.fit(X_tr, y_tr_log)

    lgb_temp = lgb.LGBMRegressor(**lgb_params)
    lgb_temp.fit(X_tr, y_tr_log)

    cat_temp = CatBoostRegressor(**cat_params)
    cat_temp.fit(X_tr, y_tr_log)

    # Ensemble predictions in log space
    pred_xgb_log = xgb_temp.predict(X_val)
    pred_lgb_log = lgb_temp.predict(X_val)
    pred_cat_log = cat_temp.predict(X_val)

    pred_ensemble_log = 0.40 * pred_xgb_log + \
        0.35 * pred_lgb_log + 0.25 * pred_cat_log

    # Convert to original space
    pred_xgb = np.expm1(pred_xgb_log)
    pred_lgb = np.expm1(pred_lgb_log)
    pred_cat = np.expm1(pred_cat_log)

    pred_ensemble = 0.40 * pred_xgb + 0.35 * pred_lgb + 0.25 * pred_cat
    pred_ensemble = np.maximum(pred_ensemble, 0)

    # Calculate RMSE in both spaces
    rmse_log = np.sqrt(mean_squared_error(y_val_log, pred_ensemble_log))
    rmse_orig = np.sqrt(mean_squared_error(y_val, pred_ensemble))

    rmse_scores_s2_log.append(rmse_log)
    rmse_scores_s2.append(rmse_orig)

print(
    f"   ✅ Scope 2 CV RMSE (Original): {np.mean(rmse_scores_s2):,.2f} ± {np.std(rmse_scores_s2):,.2f}")
print(
    f"   ✅ Scope 2 CV RMSE (Log):      {np.mean(rmse_scores_s2_log):.4f} ± {np.std(rmse_scores_s2_log):.4f}")

# ============================================================================
# 4b. XGBoost-ONLY (Core Features) 5-Fold CV for Comparison
# ============================================================================
print("\n[4b/6] XGBoost-only (core features) 5-fold CV...")

# Log-transform targets for core model
y_scope1_log_core = np.log1p(y_scope1)
y_scope2_log_core = np.log1p(y_scope2)

kf_core = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_s1_core, rmse_s2_core = [], []

for fold, (tr_idx, va_idx) in enumerate(kf_core.split(X_train_core), 1):
    print(f"   Fold {fold}/5 ...")

    X_tr, X_va = X_train_core.iloc[tr_idx], X_train_core.iloc[va_idx]
    y1_tr_log, y1_va = y_scope1_log_core[tr_idx], y_scope1[va_idx]
    y2_tr_log, y2_va = y_scope2_log_core[tr_idx], y_scope2[va_idx]

    # Scope 1 core model
    model_s1_core = xgb.XGBRegressor(**xgb_core_params)
    model_s1_core.fit(X_tr, y1_tr_log)
    pred_s1_va = np.maximum(np.expm1(model_s1_core.predict(X_va)), 0)
    rmse_s1_core.append(np.sqrt(mean_squared_error(y1_va, pred_s1_va)))

    # Scope 2 core model
    model_s2_core = xgb.XGBRegressor(**xgb_core_params)
    model_s2_core.fit(X_tr, y2_tr_log)
    pred_s2_va = np.maximum(np.expm1(model_s2_core.predict(X_va)), 0)
    rmse_s2_core.append(np.sqrt(mean_squared_error(y2_va, pred_s2_va)))

print("\n✅ XGBoost-only (core features) CV RMSE (ORIGINAL SCALE):")
print(
    f"   Scope 1: {np.mean(rmse_s1_core):,.2f} ± {np.std(rmse_s1_core):,.2f}")
print(
    f"   Scope 2: {np.mean(rmse_s2_core):,.2f} ± {np.std(rmse_s2_core):,.2f}")

print("\n🔁 Comparison vs ensemble (see Scope 1/2 CV RMSE logs above)")

# ============================================================================
# 5. Generate Test Predictions
# ============================================================================
print("\n[5/6] Generating ensemble predictions on test set...")

# Scope 1 predictions
pred_s1_xgb = np.expm1(xgb_s1.predict(X_test))
pred_s1_lgb = np.expm1(lgb_s1.predict(X_test))
pred_s1_cat = np.expm1(cat_s1.predict(X_test))

pred_s1_ensemble = 0.40 * pred_s1_xgb + 0.35 * pred_s1_lgb + 0.25 * pred_s1_cat
pred_s1_ensemble = np.maximum(pred_s1_ensemble, 0)

# Scope 2 predictions
pred_s2_xgb = np.expm1(xgb_s2.predict(X_test))
pred_s2_lgb = np.expm1(lgb_s2.predict(X_test))
pred_s2_cat = np.expm1(cat_s2.predict(X_test))

pred_s2_ensemble = 0.40 * pred_s2_xgb + 0.35 * pred_s2_lgb + 0.25 * pred_s2_cat
pred_s2_ensemble = np.maximum(pred_s2_ensemble, 0)

print(
    f"   Scope 1: Min={pred_s1_ensemble.min():.2f}, Mean={pred_s1_ensemble.mean():.2f}, Max={pred_s1_ensemble.max():.2f}")
print(
    f"   Scope 2: Min={pred_s2_ensemble.min():.2f}, Mean={pred_s2_ensemble.mean():.2f}, Max={pred_s2_ensemble.max():.2f}")

# ============================================================================
# 6. Save Submission
# ============================================================================
print("\n[6/6] Creating submission file...")

submission = pd.DataFrame({
    'entity_id': test_ids,
    'target_scope_1': pred_s1_ensemble,
    'target_scope_2': pred_s2_ensemble
})

submission.to_csv('submission.csv', index=False)

print("\n" + "="*70)
print("✅ MODEL TRAINING COMPLETE")
print("="*70)

# Calculate combined RMSE
combined_rmse_orig = np.sqrt(
    (np.mean(rmse_scores)**2 + np.mean(rmse_scores_s2)**2) / 2)
combined_rmse_log = np.sqrt(
    (np.mean(rmse_scores_log)**2 + np.mean(rmse_scores_s2_log)**2) / 2)

print(f"\nCross-Validation Results (Original Space):")
print(
    f"  Scope 1 RMSE:    {np.mean(rmse_scores):>12,.2f} ± {np.std(rmse_scores):,.2f}")
print(
    f"  Scope 2 RMSE:    {np.mean(rmse_scores_s2):>12,.2f} ± {np.std(rmse_scores_s2):,.2f}")
print(f"  Combined RMSE:   {combined_rmse_orig:>12,.2f}")

print(f"\nCross-Validation Results (Log Space):")
print(
    f"  Scope 1 RMSE:    {np.mean(rmse_scores_log):>12.4f} ± {np.std(rmse_scores_log):.4f}")
print(
    f"  Scope 2 RMSE:    {np.mean(rmse_scores_s2_log):>12.4f} ± {np.std(rmse_scores_s2_log):.4f}")
print(f"  Combined RMSE:   {combined_rmse_log:>12.4f}")

print(f"\nEnsemble Composition:")
print(f"  XGBoost:  40%")
print(f"  LightGBM: 35%")
print(f"  CatBoost: 25%")

print(f"\nSubmission saved: submission.csv")
print(f"  Rows: {len(submission)}")
print(
    f"  No negative values: {(submission[['target_scope_1', 'target_scope_2']] >= 0).all().all()}")
print("="*70)
