# https://chat.deepseek.com/share/0wdclfjicfaxks6whf
# https://chat.deepseek.com/share/pnl8d6dvf8qp1l89nr


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('./data/q1/train.csv')
col_df = pd.read_csv('./data/q1/cost_of_living.csv')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('./data/q1/train.csv')
col_df = pd.read_csv('./data/q1/cost_of_living.csv')

# Clean training data
train_df = train_df.dropna(subset=['salary_average'])

# Preprocess cost of living data
# Create aggregated cost indices while avoiding data leakage
cost_cols = [col for col in col_df.columns if col.startswith('col_')]

# Calculate robust cost statistics per city (median and IQR)
for col in cost_cols:
    # Fill missing with median of each column (from cost_of_living data only)
    col_df[col] = col_df[col].fillna(col_df[col].median())

# Create meaningful cost aggregates (avoiding arbitrary combinations)
col_df['cost_basic'] = col_df[['col_1', 'col_2', 'col_3', 'col_14', 'col_15']].mean(axis=1)  # Basic necessities
col_df['cost_housing'] = col_df[[c for c in cost_cols if 'rent' in c.lower() or any(x in c.lower() for x in ['housing', 'mortgage', 'property'])]]\
    .mean(axis=1, skipna=True) if any('rent' in c.lower() for c in cost_cols) else col_df['col_1']
col_df['cost_transport'] = col_df[[c for c in cost_cols if 'transport' in c.lower() or 'gas' in c.lower() or 'auto' in c.lower()]]\
    .mean(axis=1, skipna=True) if any('transport' in c.lower() for c in cost_cols) else col_df['col_2']

# Handle cases where specific aggregates might be empty
for col in ['cost_housing', 'cost_transport']:
    if col_df[col].isnull().all():
        col_df[col] = col_df['cost_basic']

col_df['cost_overall'] = col_df[['cost_basic', 'cost_housing', 'cost_transport']].mean(axis=1)

# Merge with training data
train_merged = pd.merge(train_df, col_df[['country', 'state', 'city', 'cost_basic', 'cost_housing', 
                                          'cost_transport', 'cost_overall']], 
                       on=['country', 'state', 'city'], how='left')

# Calculate median cost values from training data only (to avoid leakage)
cost_features = ['cost_basic', 'cost_housing', 'cost_transport', 'cost_overall']
for feature in cost_features:
    train_median = train_merged[feature].median()
    train_merged[feature] = train_merged[feature].fillna(train_median)

# Encode role using frequency encoding (less prone to leakage than one-hot)
role_counts = train_merged['role'].value_counts()
role_freq = role_counts / len(train_merged)
train_merged['role_freq'] = train_merged['role'].map(role_freq)

# Create interaction features (cost of living adjusted by role frequency)
train_merged['cost_role_interaction'] = train_merged['cost_overall'] * train_merged['role_freq']

# Prepare feature matrix
features = ['cost_overall', 'cost_basic', 'cost_housing', 'cost_transport', 
            'role_freq', 'cost_role_interaction']

# Remove any remaining NaN values
for feature in features:
    train_merged[feature] = train_merged[feature].fillna(train_merged[feature].median())

X = train_merged[features].values
y = train_merged['salary_average'].values

# Custom RMSPE function
def rmspe(y_true, y_pred):
    """Root Mean Squared Percentage Error"""
    mask = y_true != 0  # Avoid division by zero
    percentage_errors = ((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2
    return np.sqrt(np.mean(percentage_errors)) * 100

# Cross-validation to check for overfitting
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models = []

print("Cross-validation RMSPE scores:")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Standardize features (fit on train only to avoid leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    models.append((model, scaler))
    
    # Predict and evaluate
    y_pred = model.predict(X_val_scaled)
    score = rmspe(y_val, y_pred)
    cv_scores.append(score)
    print(f"  Fold {fold+1}: {score:.2f}%")

print(f"\nAverage CV RMSPE: {np.mean(cv_scores):.2f}%")
print(f"Std Dev: {np.std(cv_scores):.2f}%")

# Train final model on all data with proper standardization
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
final_model = LinearRegression()
final_model.fit(X_scaled, y)

# Display model coefficients
print("\nModel coefficients:")
for feature, coef in zip(features, final_model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {final_model.intercept_:.2f}")

# Load and preprocess test data
test_df = pd.read_csv('./data/q1/test.csv')

# Merge with cost of living data
test_merged = pd.merge(test_df, col_df[['country', 'state', 'city', 'cost_basic', 'cost_housing', 
                                        'cost_transport', 'cost_overall']], 
                      on=['country', 'state', 'city'], how='left')

# Use training medians for missing values (no leakage)
for feature in cost_features:
    test_merged[feature] = test_merged[feature].fillna(train_median)

# Apply same role frequency encoding (with fallback for unseen roles)
test_merged['role_freq'] = test_merged['role'].map(role_freq)
test_merged['role_freq'] = test_merged['role_freq'].fillna(role_freq.median())  # Use median for unseen roles

# Create interaction feature
test_merged['cost_role_interaction'] = test_merged['cost_overall'] * test_merged['role_freq']

# Prepare test features
X_test = test_merged[features].values

# Fill any remaining NaN with training medians
for i, feature in enumerate(features):
    col_median = train_merged[feature].median()
    X_test[:, i] = np.where(np.isnan(X_test[:, i]), col_median, X_test[:, i])

# Scale test features using the training scaler
X_test_scaled = final_scaler.transform(X_test)

# Make predictions
test_predictions = final_model.predict(X_test_scaled)

# Ensure predictions are reasonable (no negative salaries)
test_predictions = np.maximum(test_predictions, 1000)  # Minimum reasonable salary

# Create submission
submission = pd.DataFrame({
    'ID': test_merged['ID'],
    'salary_average': test_predictions
})

# Save predictions
submission.to_csv('predictions.csv', index=False)
print(f"\nPredictions saved to predictions.csv")
print(f"Number of predictions: {len(submission)}")

# Check feature importance
print("\nFeature importance (absolute coefficients):")
coef_df = pd.DataFrame({
    'feature': features,
    'coefficient': final_model.coef_,
    'abs_coef': np.abs(final_model.coef_)
}).sort_values('abs_coef', ascending=False)
print(coef_df[['feature', 'coefficient']].to_string(index=False))

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def calculate_rmpse(y_true, y_pred):
    """Calculate Root Mean Percentage Square Error (RMPSE)"""
    percentage_errors = ((y_true - y_pred) / y_true)
    mspse = np.mean(percentage_errors ** 2)
    return np.sqrt(mspse)

def preprocess_and_merge_data(train_df, col_df, test_df=None):
    """
    Preprocess and merge data while avoiding leakage
    """
    # Create a copy to avoid modifying originals
    train_processed = train_df.copy()
    col_processed = col_df.copy()
    
    # Drop ID columns and target column from features
    train_processed = train_processed.drop(['ID'], axis=1)
    
    # Handle missing target values
    train_processed = train_processed.dropna(subset=['salary_average'])
    
    # Separate target
    y = train_processed['salary_average']
    X = train_processed.drop(['salary_average'], axis=1)
    
    # Process cost of living data
    # Aggregate cost of living features to avoid city-level leakage
    # Calculate median cost of living indices by state (not city!)
    col_state_agg = col_processed.groupby(['country', 'state']).agg({
        'col_1': 'median',
        'col_2': 'median',
        'col_3': 'median',
        'col_4': 'median',
        'col_5': 'median',
        'col_6': 'median',
        'col_7': 'median',
        'col_8': 'median',
        'col_9': 'median',
        'col_10': 'median'
    }).reset_index()
    
    # Fill missing values in col features with state median
    for col in [f'col_{i}' for i in range(1, 11)]:
        if col in col_state_agg.columns:
            col_state_agg[col] = col_state_agg[col].fillna(col_state_agg[col].median())
    
    # Merge with training data on country and state (NOT city to avoid leakage)
    X_merged = pd.merge(
        X, 
        col_state_agg, 
        on=['country', 'state'], 
        how='left',
        suffixes=('', '_col')
    )
    
    # Drop city column to prevent city-based leakage
    X_merged = X_merged.drop(['city'], axis=1)
    
    # If test data provided, process it similarly
    if test_df is not None:
        test_processed = test_df.copy()
        test_ids = test_processed['ID']
        test_processed = test_processed.drop(['ID'], axis=1)
        
        # Merge test data with aggregated col data
        test_merged = pd.merge(
            test_processed,
            col_state_agg,
            on=['country', 'state'],
            how='left',
            suffixes=('', '_col')
        )
        
        # Drop city column
        test_merged = test_merged.drop(['city'], axis=1)
        
        return X_merged, y, test_merged, test_ids
    
    return X_merged, y, None, None

def reduce_col_features(X, n_components=5):
    """
    Reduce cost of living features using PCA
    """
    # Select only cost of living columns
    col_cols = [col for col in X.columns if col.startswith('col_')]
    
    if len(col_cols) == 0:
        return X
    
    # Separate col features from categorical features
    col_features = X[col_cols].copy()
    non_col_features = X.drop(col_cols, axis=1).copy()
    
    # Handle missing values in col features
    col_features = col_features.fillna(col_features.median())
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(n_components, len(col_cols)))
    col_reduced = pca.fit_transform(col_features)
    
    # Create column names for PCA components
    pca_cols = [f'col_pca_{i+1}' for i in range(col_reduced.shape[1])]
    
    # Create DataFrame with PCA components
    col_reduced_df = pd.DataFrame(col_reduced, columns=pca_cols, index=X.index)
    
    # Combine with non-col features
    X_reduced = pd.concat([non_col_features.reset_index(drop=True), 
                          col_reduced_df.reset_index(drop=True)], axis=1)
    
    return X_reduced

def train_linear_regression_model(X_train, y_train, test_data=None):
    """
    Train a linear regression model with regularization to prevent overfitting
    """
    # Separate features by type
    categorical_features = ['country', 'state', 'role']
    numerical_features = [col for col in X_train.columns if col.startswith('col_')]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Create pipeline with preprocessing and Ridge regression (regularized)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))  # Using Ridge for regularization
    ])
    
    # Perform cross-validation to check for overfitting
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, 
        scoring='neg_mean_squared_error'
    )
    
    # Calculate RMSE from cross-validation
    cv_rmse = np.sqrt(-cv_scores)
    print(f"Cross-validation RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")
    
    # Train the model on full training data
    model.fit(X_train, y_train)
    
    # Make predictions if test data provided
    if test_data is not None:
        predictions = model.predict(test_data)
        return model, predictions
    
    return model, None

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('./data/q1/train.csv')
    col_df = pd.read_csv('./data/q1/cost_of_living.csv')
    test_df = pd.read_csv('./data/q1/test.csv')
    
    # Preprocess and merge data
    print("Preprocessing data...")
    X_train, y_train, X_test, test_ids = preprocess_and_merge_data(
        train_df, col_df, test_df
    )
    
    # Reduce cost of living features
    print("Reducing cost of living features...")
    X_train_reduced = reduce_col_features(X_train, n_components=5)
    X_test_reduced = reduce_col_features(X_test, n_components=5)
    
    # Split training data for validation
    print("Splitting data for validation...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_reduced, y_train, test_size=0.2, random_state=42
    )
    
    # Train model
    print("\nTraining model...")
    model, _ = train_linear_regression_model(X_train_split, y_train_split)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    rmpse_val = calculate_rmpse(y_val, y_val_pred)
    print(f"\nValidation RMPSE: {rmpse_val:.4f}")
    
    # Train final model on all training data
    print("\nTraining final model on all data...")
    final_model, test_predictions = train_linear_regression_model(
        X_train_reduced, y_train, X_test_reduced
    )
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': test_ids,
        'salary_average': test_predictions
    })
    
    # Save predictions
    submission.to_csv('salary_predictions.csv', index=False)
    print(f"\nPredictions saved to 'salary_predictions.csv'")
    print(f"Shape of predictions: {submission.shape}")
    
    return final_model, submission

if __name__ == "__main__":
    model, predictions = main()