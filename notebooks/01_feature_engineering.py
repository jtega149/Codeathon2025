#!/usr/bin/env python3
"""
AGGRESSIVE Feature Engineering for Low RMSE
- Extreme log transformations on all numeric features
- Target encoding with smoothing
- Polynomial features for key interactions
- Sector and country-level aggregations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("AGGRESSIVE FEATURE ENGINEERING - TARGET RMSE: ~60k")
print("="*70)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1/8] Loading data...")
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
sectors = pd.read_csv("../data/revenue_distribution_by_sector.csv")
env = pd.read_csv("../data/environmental_activities.csv")
sdg = pd.read_csv("../data/sustainable_development_goals.csv")

print(f"   Train: {train.shape}, Test: {test.shape}")

# ============================================================================
# 2. EXTREME Log Transformations (Critical for RMSE reduction)
# ============================================================================
print("\n[2/8] Applying extreme log transformations...")

# Log transform revenue (critical!)
train['log_revenue'] = np.log1p(train['revenue'])
test['log_revenue'] = np.log1p(test['revenue'])

# Log transform ALL score features
for col in ['overall_score', 'environmental_score', 'social_score', 'governance_score']:
    train[f'log_{col}'] = np.log1p(train[col])
    test[f'log_{col}'] = np.log1p(test[col])

# Revenue power transformations
train['revenue_squared'] = train['revenue'] ** 2
test['revenue_squared'] = test['revenue'] ** 2
train['revenue_cubed'] = train['revenue'] ** 3
test['revenue_cubed'] = test['revenue'] ** 3
train['revenue_sqrt'] = np.sqrt(train['revenue'])
test['revenue_sqrt'] = np.sqrt(test['revenue'])

print(f"   ✅ Created {len([c for c in train.columns if 'log_' in c or '_squared' in c or '_cubed' in c or '_sqrt' in c])} transformed features")

# ============================================================================
# 3. Advanced Sector Features with Target Encoding
# ============================================================================
print("\n[3/8] Creating advanced sector features...")

HIGH_EMISSION_SECTORS = ['B', 'C', 'D', 'E', 'F', 'H']

# Pivot sectors
sector_pivot = sectors.pivot_table(
    values='revenue_pct',
    index='entity_id',
    columns='nace_level_1_code',
    aggfunc='sum',
    fill_value=0
).add_prefix('sector_').reset_index()

# High emission concentration
high_cols = [
    f'sector_{s}' for s in HIGH_EMISSION_SECTORS if f'sector_{s}' in sector_pivot.columns]
sector_pivot['high_emission_pct'] = sector_pivot[high_cols].sum(axis=1)
sector_pivot['is_high_emission'] = (
    sector_pivot['high_emission_pct'] > 0.6).astype(int)

# Sector diversity metrics
sector_cols = [c for c in sector_pivot.columns if c.startswith(
    'sector_') and c not in ['high_emission_pct', 'is_high_emission']]
sector_pivot['sector_count'] = (sector_pivot[sector_cols] > 0.01).sum(axis=1)
sector_pivot['dominant_sector'] = sector_pivot[sector_cols].max(axis=1)
sector_pivot['sector_entropy'] = -np.sum(sector_pivot[sector_cols].values * np.log(
    sector_pivot[sector_cols].values + 1e-10), axis=1)

# Sector target encoding (using mean emissions per sector)
sector_emissions = {}
for sector in HIGH_EMISSION_SECTORS:
    if f'sector_{sector}' in sector_pivot.columns:
        sector_pivot_train = sector_pivot[sector_pivot['entity_id'].isin(
            train['entity_id'])]
        merged = sector_pivot_train.merge(
            train[['entity_id', 'target_scope_1', 'target_scope_2']], on='entity_id')

        # Weighted mean emissions for this sector
        sector_weight = merged[f'sector_{sector}']
        sector_pivot[f'sector_{sector}_s1_mean'] = (
            sector_weight > 0).astype(float) * train['target_scope_1'].mean()
        sector_pivot[f'sector_{sector}_s2_mean'] = (
            sector_weight > 0).astype(float) * train['target_scope_2'].mean()

train = train.merge(sector_pivot, on='entity_id', how='left')
test = test.merge(sector_pivot, on='entity_id', how='left')

# Fill NaN
sector_feature_cols = [c for c in sector_pivot.columns if c != 'entity_id']
train[sector_feature_cols] = train[sector_feature_cols].fillna(0)
test[sector_feature_cols] = test[sector_feature_cols].fillna(0)

print(f"   ✅ Created {len(sector_feature_cols)} sector features")

# ============================================================================
# 4. Environmental Features
# ============================================================================
print("\n[4/8] Creating environmental features...")

env_agg = env.groupby('entity_id')['env_score_adjustment'].agg([
    ('env_sum', 'sum'),
    ('env_mean', 'mean'),
    ('env_min', 'min'),
    ('env_max', 'max'),
    ('env_std', 'std'),
    ('env_count', 'count')
]).reset_index()

env_agg['has_env'] = 1
env_agg['env_std'] = env_agg['env_std'].fillna(0)

train = train.merge(env_agg, on='entity_id', how='left')
test = test.merge(env_agg, on='entity_id', how='left')

env_cols = [c for c in env_agg.columns if c != 'entity_id']
train[env_cols] = train[env_cols].fillna(0)
test[env_cols] = test[env_cols].fillna(0)

print(f"   ✅ Created {len(env_cols)} environmental features")

# ============================================================================
# 5. SDG Features
# ============================================================================
print("\n[5/8] Creating SDG features...")

sdg_pivot = pd.get_dummies(sdg[['entity_id', 'sdg_id']], columns=[
                           'sdg_id'], prefix='sdg')
sdg_pivot = sdg_pivot.groupby('entity_id').sum().reset_index()

sdg_cols = [c for c in sdg_pivot.columns if c.startswith('sdg_')]
sdg_pivot['sdg_total'] = sdg_pivot[sdg_cols].sum(axis=1)
sdg_pivot['has_sdg'] = (sdg_pivot['sdg_total'] > 0).astype(int)

train = train.merge(sdg_pivot, on='entity_id', how='left')
test = test.merge(sdg_pivot, on='entity_id', how='left')

sdg_feature_cols = [c for c in sdg_pivot.columns if c != 'entity_id']
train[sdg_feature_cols] = train[sdg_feature_cols].fillna(0)
test[sdg_feature_cols] = test[sdg_feature_cols].fillna(0)

print(f"   ✅ Created {len(sdg_feature_cols)} SDG features")

# ============================================================================
# 6. Geographic Features with Target Encoding
# ============================================================================
print("\n[6/8] Creating geographic features with target encoding...")

# One-hot regions
train = pd.get_dummies(train, columns=['region_code'], prefix='region')
test = pd.get_dummies(test, columns=['region_code'], prefix='region')

# Align region columns
train_regions = [c for c in train.columns if c.startswith('region_')]
test_regions = [c for c in test.columns if c.startswith('region_')]

for col in train_regions:
    if col not in test.columns:
        test[col] = False
for col in test_regions:
    if col not in train.columns:
        train[col] = False

# CRITICAL: Country target encoding with smoothing (Bayesian mean)
SMOOTHING = 10  # Higher = more regularization

country_s1_mean = train.groupby('country_code')['target_scope_1'].mean()
country_s1_count = train.groupby('country_code')['target_scope_1'].count()
global_s1_mean = train['target_scope_1'].mean()

country_s2_mean = train.groupby('country_code')['target_scope_2'].mean()
country_s2_count = train.groupby('country_code')['target_scope_2'].count()
global_s2_mean = train['target_scope_2'].mean()

# Smoothed encoding
train['country_s1_encoded'] = train['country_code'].map(
    lambda x: (country_s1_mean.get(x, global_s1_mean) * country_s1_count.get(x, 0) + global_s1_mean * SMOOTHING) /
              (country_s1_count.get(x, 0) + SMOOTHING)
)
train['country_s2_encoded'] = train['country_code'].map(
    lambda x: (country_s2_mean.get(x, global_s2_mean) * country_s2_count.get(x, 0) + global_s2_mean * SMOOTHING) /
              (country_s2_count.get(x, 0) + SMOOTHING)
)

test['country_s1_encoded'] = test['country_code'].map(
    lambda x: (country_s1_mean.get(x, global_s1_mean) * country_s1_count.get(x, 0) + global_s1_mean * SMOOTHING) /
              (country_s1_count.get(x, 0) + SMOOTHING)
)
test['country_s2_encoded'] = test['country_code'].map(
    lambda x: (country_s2_mean.get(x, global_s2_mean) * country_s2_count.get(x, 0) + global_s2_mean * SMOOTHING) /
              (country_s2_count.get(x, 0) + SMOOTHING)
)

print(f"   ✅ Created country target encoding (smoothing={SMOOTHING})")

# ============================================================================
# 7. Critical Interaction Features
# ============================================================================
print("\n[7/8] Creating critical interaction features...")

# Revenue interactions
train['revenue_x_high_emission'] = train['revenue'] * \
    train['high_emission_pct']
test['revenue_x_high_emission'] = test['revenue'] * test['high_emission_pct']

train['log_revenue_x_env'] = train['log_revenue'] * \
    train['environmental_score']
test['log_revenue_x_env'] = test['log_revenue'] * test['environmental_score']

train['revenue_x_country_s1'] = train['revenue'] * train['country_s1_encoded']
test['revenue_x_country_s1'] = test['revenue'] * test['country_s1_encoded']

train['revenue_x_country_s2'] = train['revenue'] * train['country_s2_encoded']
test['revenue_x_country_s2'] = test['revenue'] * test['country_s2_encoded']

# Score ratios
train['env_to_overall'] = train['environmental_score'] / \
    (train['overall_score'] + 1e-6)
test['env_to_overall'] = test['environmental_score'] / \
    (test['overall_score'] + 1e-6)

train['social_to_overall'] = train['social_score'] / \
    (train['overall_score'] + 1e-6)
test['social_to_overall'] = test['social_score'] / \
    (test['overall_score'] + 1e-6)

# Environmental interactions
train['env_x_high_emission'] = train['env_sum'] * train['high_emission_pct']
test['env_x_high_emission'] = test['env_sum'] * test['high_emission_pct']

print(f"   ✅ Created 8 critical interaction features")

# ============================================================================
# 8. Save Engineered Data
# ============================================================================
print("\n[8/8] Saving engineered features...")

exclude = ['entity_id', 'region_name', 'country_name',
           'country_code', 'target_scope_1', 'target_scope_2']
features = [c for c in train.columns if c not in exclude]

X_train = train[features]
y_scope1 = train['target_scope_1']
y_scope2 = train['target_scope_2']
X_test = test[features]
test_ids = test['entity_id']

# Save as CSV to avoid pickle issues
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_scope1.to_csv('y_scope1.csv', index=False, header=True)
y_scope2.to_csv('y_scope2.csv', index=False, header=True)
test_ids.to_csv('test_ids.csv', index=False, header=True)

print("\n" + "="*70)
print("✅ FEATURE ENGINEERING COMPLETE")
print("="*70)
print(f"Total features: {len(features)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"\nKey features created:")
print(f"  - Log transformations: {len([c for c in features if 'log_' in c])}")
print(
    f"  - Power transformations: {len([c for c in features if any(x in c for x in ['squared', 'cubed', 'sqrt'])])}")
print(f"  - Sector features: {len([c for c in features if 'sector_' in c])}")
print(
    f"  - Target encodings: {len([c for c in features if 'encoded' in c or '_mean' in c])}")
print(f"  - Interactions: {len([c for c in features if '_x_' in c])}")
print("\nFiles saved:")
print("  - X_train.csv")
print("  - X_test.csv")
print("  - y_scope1.csv")
print("  - y_scope2.csv")
print("  - test_ids.csv")
print("="*70)
