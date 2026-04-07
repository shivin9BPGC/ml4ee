import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def rmspe(y_true, y_pred):
    """Root Mean Squared Percentage Error"""
    mask = y_true != 0
    percentage_errors = ((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2
    return np.sqrt(np.mean(percentage_errors)) * 100

# Load data
print("Loading datasets...")
train_df = pd.read_csv('../data/q1/train.csv')
test_df = pd.read_csv('../data/q1/test.csv')
col_df = pd.read_csv('../data/q1/cost_of_living.csv')

# Clean training data
train_df = train_df.dropna(subset=['salary_average'])

# Process cost of living data
print("Processing cost of living data...")
cost_cols = [col for col in col_df.columns if col.startswith('col_')]

# Fill missing with median
for col in cost_cols:
    col_df[col] = col_df[col].fillna(col_df[col].median())

# Create cost aggregates
col_df['cost_basic'] = col_df[['col_1', 'col_2', 'col_3', 'col_14', 'col_15']].mean(axis=1)
col_df['cost_overall'] = col_df[cost_cols[:10]].mean(axis=1)

# Merge with training data
print("Creating features...")
train_merged = pd.merge(train_df, col_df[['country', 'state', 'city', 'cost_basic', 'cost_overall']], 
                       on=['country', 'state', 'city'], how='left')

# Fill missing cost values
for feature in ['cost_basic', 'cost_overall']:
    train_median = train_merged[feature].median()
    train_merged[feature] = train_merged[feature].fillna(train_median)

# Role frequency encoding
role_counts = train_merged['role'].value_counts()
role_freq = role_counts / len(train_merged)
train_merged['role_freq'] = train_merged['role'].map(role_freq)

# Create interaction feature
train_merged['cost_role_interaction'] = train_merged['cost_overall'] * train_merged['role_freq']

# Prepare features
features = ['cost_overall', 'cost_basic', 'role_freq', 'cost_role_interaction']

# Remove any remaining NaN values
for feature in features:
    train_merged[feature] = train_merged[feature].fillna(train_merged[feature].median())

X = train_merged[features].values
y = train_merged['salary_average'].values

print(f"Dataset shape: {X.shape}")
print(f"Features: {features}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
print("\nTraining models...")
models = {}

# Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
models['Linear'] = (lr, scaler)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
models['Ridge'] = (ridge, scaler)

# Evaluate models
print("\nValidation results:")
best_model = None
best_score = float('inf')
best_name = None

for name, (model, model_scaler) in models.items():
    val_pred = model.predict(X_val_scaled)
    val_rmspe = rmspe(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"{name}: RMSPE={val_rmspe:.2f}%, R²={val_r2:.4f}")
    
    if val_rmspe < best_score:
        best_score = val_rmspe
        best_model = (model, model_scaler)
        best_name = name

print(f"\nBest model: {best_name} (RMSPE: {best_score:.2f}%)")

# Process test data
print("\nProcessing test data...")
test_merged = pd.merge(test_df, col_df[['country', 'state', 'city', 'cost_basic', 'cost_overall']], 
                      on=['country', 'state', 'city'], how='left')

# Use training medians for missing values
for feature in ['cost_basic', 'cost_overall']:
    test_median = train_merged[feature].median()
    test_merged[feature] = test_merged[feature].fillna(test_median)

# Apply same role frequency encoding
test_merged['role_freq'] = test_merged['role'].map(role_freq)
test_merged['role_freq'] = test_merged['role_freq'].fillna(role_freq.median())

# Create interaction feature
test_merged['cost_role_interaction'] = test_merged['cost_overall'] * test_merged['role_freq']

# Prepare test features
X_test = test_merged[features].values

# Fill any remaining NaN
for i, feature in enumerate(features):
    col_median = train_merged[feature].median()
    X_test[:, i] = np.where(np.isnan(X_test[:, i]), col_median, X_test[:, i])

# Scale and predict
model, scaler = best_model
X_test_scaled = scaler.transform(X_test)
test_predictions = model.predict(X_test_scaled)

# Ensure reasonable predictions
test_predictions = np.maximum(test_predictions, 1000)

# Create submission
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'salary_average': test_predictions
})

# Save predictions
submission.to_csv('baseline_predictions.csv', index=False)

# Load ground truth and evaluate
try:
    ground_truth = pd.read_csv('./q1_solution.csv')
    merged_eval = pd.merge(submission, ground_truth, on='ID')
    
    final_rmspe = rmspe(merged_eval['salary_average_y'], merged_eval['salary_average_x'])
    final_rmse = np.sqrt(mean_squared_error(merged_eval['salary_average_y'], merged_eval['salary_average_x']))
    final_r2 = r2_score(merged_eval['salary_average_y'], merged_eval['salary_average_x'])
    
    print(f"\n=== BASELINE RESULTS ===")
    print(f"Test RMSPE: {final_rmspe:.2f}%")
    print(f"Test RMSE: {final_rmse:.2f}")
    print(f"Test R²: {final_r2:.4f}")
    print(f"Number of predictions: {len(submission)}")
    
except FileNotFoundError:
    print(f"\nPredictions saved to 'baseline_predictions.csv'")
    print(f"Number of predictions: {len(submission)}")

print("\nBaseline model training completed!")