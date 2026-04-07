import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

def load_and_preprocess_data():
    """Load and preprocess all datasets with enhanced feature engineering"""
    print("Loading datasets...")
    
    # Load data
    train_df = pd.read_csv('../data/q1/train.csv')
    test_df = pd.read_csv('../data/q1/test.csv')
    col_df = pd.read_csv('../data/q1/cost_of_living.csv')
    
    # Clean training data - remove missing targets
    train_df = train_df.dropna(subset=['salary_average'])
    
    # Process cost of living data
    print("Processing cost of living features...")
    
    # Fill missing values in cost columns with median
    cost_cols = [col for col in col_df.columns if col.startswith('col_')]
    for col in cost_cols:
        col_df[col] = col_df[col].fillna(col_df[col].median())
    
    # Create meaningful cost aggregates
    col_df['cost_basic'] = col_df[['col_1', 'col_2', 'col_3', 'col_14', 'col_15']].mean(axis=1)
    col_df['cost_housing'] = col_df[['col_16', 'col_17', 'col_18', 'col_19', 'col_20']].mean(axis=1)
    col_df['cost_transport'] = col_df[['col_4', 'col_5', 'col_6']].mean(axis=1)
    col_df['cost_food'] = col_df[['col_7', 'col_8', 'col_9', 'col_10']].mean(axis=1)
    col_df['cost_entertainment'] = col_df[['col_21', 'col_22', 'col_23', 'col_24']].mean(axis=1)
    col_df['cost_overall'] = col_df[['cost_basic', 'cost_housing', 'cost_transport', 'cost_food', 'cost_entertainment']].mean(axis=1)
    
    # Statistical cost features
    col_df['cost_std'] = col_df[cost_cols[:20]].std(axis=1)
    col_df['cost_max'] = col_df[cost_cols[:20]].max(axis=1)
    col_df['cost_min'] = col_df[cost_cols[:20]].min(axis=1)
    col_df['cost_range'] = col_df['cost_max'] - col_df['cost_min']
    
    return train_df, test_df, col_df

def create_enhanced_features(df, col_df, is_training=True, training_stats=None):
    """Create enhanced features with proper handling to avoid data leakage"""
    print(f"Creating enhanced features for {'training' if is_training else 'test'} data...")
    
    # Merge with cost of living data
    merged_df = pd.merge(df, col_df, on=['country', 'state', 'city'], how='left')
    
    # Define cost feature columns
    cost_features = ['cost_basic', 'cost_housing', 'cost_transport', 'cost_food', 
                    'cost_entertainment', 'cost_overall', 'cost_std', 'cost_max', 
                    'cost_min', 'cost_range']
    
    if is_training:
        # Calculate statistics from training data to avoid leakage
        training_stats = {}
        for feature in cost_features:
            training_stats[feature] = merged_df[feature].median()
        
        # Fill missing values with training medians
        for feature in cost_features:
            merged_df[feature] = merged_df[feature].fillna(training_stats[feature])
    else:
        # Use training statistics for test data
        for feature in cost_features:
            merged_df[feature] = merged_df[feature].fillna(training_stats[feature])
    
    # Role encoding features
    if is_training:
        role_counts = merged_df['role'].value_counts()
        role_freq = role_counts / len(merged_df)
        training_stats['role_freq'] = role_freq
        training_stats['role_median_freq'] = role_freq.median()
    else:
        role_freq = training_stats['role_freq']
    
    merged_df['role_freq'] = merged_df['role'].map(role_freq)
    merged_df['role_freq'] = merged_df['role_freq'].fillna(training_stats['role_median_freq'])
    
    # Create interaction features
    merged_df['cost_role_interaction'] = merged_df['cost_overall'] * merged_df['role_freq']
    merged_df['cost_housing_role'] = merged_df['cost_housing'] * merged_df['role_freq']
    merged_df['cost_transport_role'] = merged_df['cost_transport'] * merged_df['role_freq']
    
    # Log transformations for skewed features
    merged_df['log_cost_overall'] = np.log1p(merged_df['cost_overall'])
    merged_df['log_cost_housing'] = np.log1p(merged_df['cost_housing'])
    merged_df['sqrt_cost_basic'] = np.sqrt(merged_df['cost_basic'])
    
    # Location encoding (simple frequency encoding for countries and states)
    if is_training:
        country_freq = merged_df['country'].value_counts() / len(merged_df)
        state_freq = merged_df['state'].value_counts() / len(merged_df)
        training_stats['country_freq'] = country_freq
        training_stats['state_freq'] = state_freq
        training_stats['country_median_freq'] = country_freq.median()
        training_stats['state_median_freq'] = state_freq.median()
    
    merged_df['country_freq'] = merged_df['country'].map(training_stats['country_freq'])
    merged_df['country_freq'] = merged_df['country_freq'].fillna(training_stats['country_median_freq'])
    
    merged_df['state_freq'] = merged_df['state'].map(training_stats['state_freq'])
    merged_df['state_freq'] = merged_df['state_freq'].fillna(training_stats['state_median_freq'])
    
    # Define feature columns
    feature_columns = cost_features + [
        'role_freq', 'cost_role_interaction', 'cost_housing_role', 'cost_transport_role',
        'log_cost_overall', 'log_cost_housing', 'sqrt_cost_basic', 'country_freq', 'state_freq'
    ]
    
    # Handle any remaining NaN values
    for col in feature_columns:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    
    return merged_df[feature_columns], training_stats if is_training else None

def rmspe(y_true, y_pred):
    """Root Mean Squared Percentage Error"""
    mask = y_true != 0
    percentage_errors = ((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2
    return np.sqrt(np.mean(percentage_errors)) * 100

def train_teacher_models(X_train, y_train, X_val, y_val):
    """Train XGBoost and LightGBM teacher models"""
    print("\nTraining teacher models...")
    
    # XGBoost
    print("Training XGBoost...")
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # LightGBM
    print("Training LightGBM...")
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42,
        'verbosity': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    return xgb_model, lgb_model

def knowledge_distillation(X_train, y_train, X_val, y_val, xgb_model, lgb_model):
    """Implement knowledge distillation from teacher models to student linear model"""
    print("\nImplementing knowledge distillation...")
    
    # Get teacher predictions
    dtrain = xgb.DMatrix(X_train)
    dval = xgb.DMatrix(X_val)
    
    xgb_train_pred = xgb_model.predict(dtrain)
    xgb_val_pred = xgb_model.predict(dval)
    
    lgb_train_pred = lgb_model.predict(X_train)
    lgb_val_pred = lgb_model.predict(X_val)
    
    # Ensemble teacher predictions (average)
    teacher_train_pred = (xgb_train_pred + lgb_train_pred) / 2
    teacher_val_pred = (xgb_val_pred + lgb_val_pred) / 2
    
    # Create enhanced feature sets with teacher knowledge
    # Add teacher predictions as features
    X_train_enhanced = np.column_stack([X_train, teacher_train_pred, xgb_train_pred, lgb_train_pred])
    X_val_enhanced = np.column_stack([X_val, teacher_val_pred, xgb_val_pred, lgb_val_pred])
    
    # Feature importance guided transformations
    xgb_importance = xgb_model.get_score(importance_type='weight')
    lgb_importance = lgb_model.feature_importance(importance_type='split')
    
    # Create weighted features based on importance
    if len(lgb_importance) == X_train.shape[1]:
        importance_weights = lgb_importance / np.sum(lgb_importance)
        weighted_features_train = X_train * importance_weights
        weighted_features_val = X_val * importance_weights
        
        # Add weighted sum as new feature
        X_train_enhanced = np.column_stack([X_train_enhanced, np.sum(weighted_features_train, axis=1)])
        X_val_enhanced = np.column_stack([X_val_enhanced, np.sum(weighted_features_val, axis=1)])
    
    return X_train_enhanced, X_val_enhanced, teacher_train_pred, teacher_val_pred

def train_student_model(X_train_enhanced, y_train, X_val_enhanced, y_val, teacher_train_pred, teacher_val_pred):
    """Train student linear regression model with knowledge distillation"""
    print("\nTraining student model with knowledge distillation...")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enhanced)
    X_val_scaled = scaler.transform(X_val_enhanced)
    
    # Try different distillation approaches
    models = {}
    
    # 1. Direct linear regression on enhanced features
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models['linear_enhanced'] = (lr, scaler)
    
    # 2. Ridge regression with regularization
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    models['ridge_enhanced'] = (ridge, scaler)
    
    # 3. Combined loss: original target + teacher predictions
    # Weight between hard targets (true labels) and soft targets (teacher predictions)
    alpha = 0.7  # Weight for hard targets
    beta = 1 - alpha  # Weight for soft targets
    
    combined_targets = alpha * y_train + beta * teacher_train_pred
    
    lr_combined = LinearRegression()
    lr_combined.fit(X_train_scaled, combined_targets)
    models['linear_distilled'] = (lr_combined, scaler)
    
    ridge_combined = Ridge(alpha=0.5)
    ridge_combined.fit(X_train_scaled, combined_targets)
    models['ridge_distilled'] = (ridge_combined, scaler)
    
    # Evaluate all models
    best_model = None
    best_score = float('inf')
    best_name = None
    
    print("\nEvaluating student models:")
    for name, (model, model_scaler) in models.items():
        val_pred = model.predict(X_val_scaled)
        val_rmspe = rmspe(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"{name}: RMSPE={val_rmspe:.2f}%, RMSE={val_rmse:.2f}, R²={val_r2:.4f}")
        
        if val_rmspe < best_score:
            best_score = val_rmspe
            best_model = (model, model_scaler)
            best_name = name
    
    print(f"\nBest student model: {best_name} (RMSPE: {best_score:.2f}%)")
    
    return best_model, models

def main():
    """Main execution function"""
    print("=== Knowledge Distillation: Tree Models → Linear Regression ===")
    
    # Load and preprocess data
    train_df, test_df, col_df = load_and_preprocess_data()
    
    # Create enhanced features
    X_train_full, training_stats = create_enhanced_features(train_df, col_df, is_training=True)
    y_train_full = train_df['salary_average'].values
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"\nDataset shapes:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Features: {X_train.shape[1]}")
    
    # Train teacher models
    xgb_model, lgb_model = train_teacher_models(X_train, y_train, X_val, y_val)
    
    # Evaluate teacher models
    print("\nEvaluating teacher models:")
    dtrain = xgb.DMatrix(X_train)
    dval = xgb.DMatrix(X_val)
    
    xgb_val_pred = xgb_model.predict(dval)
    lgb_val_pred = lgb_model.predict(X_val)
    ensemble_val_pred = (xgb_val_pred + lgb_val_pred) / 2
    
    print(f"XGBoost RMSPE: {rmspe(y_val, xgb_val_pred):.2f}%")
    print(f"LightGBM RMSPE: {rmspe(y_val, lgb_val_pred):.2f}%")
    print(f"Ensemble RMSPE: {rmspe(y_val, ensemble_val_pred):.2f}%")
    
    # Knowledge distillation
    X_train_enhanced, X_val_enhanced, teacher_train_pred, teacher_val_pred = knowledge_distillation(
        X_train, y_train, X_val, y_val, xgb_model, lgb_model
    )
    
    # Train student model
    best_student, all_models = train_student_model(
        X_train_enhanced, y_train, X_val_enhanced, y_val, teacher_train_pred, teacher_val_pred
    )
    
    # Prepare test data and make final predictions
    print("\nPreparing test predictions...")
    X_test, _ = create_enhanced_features(test_df, col_df, is_training=False, training_stats=training_stats)
    
    # Get teacher predictions for test data
    dtest = xgb.DMatrix(X_test)
    xgb_test_pred = xgb_model.predict(dtest)
    lgb_test_pred = lgb_model.predict(X_test)
    teacher_test_pred = (xgb_test_pred + lgb_test_pred) / 2
    
    # Enhance test features
    X_test_enhanced = np.column_stack([X_test, teacher_test_pred, xgb_test_pred, lgb_test_pred])
    
    # Add weighted sum if we used it in training
    if X_test_enhanced.shape[1] == X_train_enhanced.shape[1] - 1:
        lgb_importance = lgb_model.feature_importance(importance_type='split')
        if len(lgb_importance) == X_test.shape[1]:
            importance_weights = lgb_importance / np.sum(lgb_importance)
            weighted_features_test = X_test * importance_weights
            X_test_enhanced = np.column_stack([X_test_enhanced, np.sum(weighted_features_test, axis=1)])
    
    # Scale test features
    student_model, scaler = best_student
    X_test_scaled = scaler.transform(X_test_enhanced)
    
    # Final predictions
    final_predictions = student_model.predict(X_test_scaled)
    final_predictions = np.maximum(final_predictions, 1000)  # Ensure reasonable minimum salary
    
    # Save predictions
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'salary_average': final_predictions
    })
    submission.to_csv('distilled_predictions.csv', index=False)
    
    # Load ground truth and evaluate
    try:
        ground_truth = pd.read_csv('./q1_solution.csv')
        merged_eval = pd.merge(submission, ground_truth, on='ID')
        
        final_rmspe = rmspe(merged_eval['salary_average_y'], merged_eval['salary_average_x'])
        final_rmse = np.sqrt(mean_squared_error(merged_eval['salary_average_y'], merged_eval['salary_average_x']))
        final_r2 = r2_score(merged_eval['salary_average_y'], merged_eval['salary_average_x'])
        
        print(f"\n=== FINAL TEST RESULTS ===")
        print(f"Test RMSPE: {final_rmspe:.2f}%")
        print(f"Test RMSE: {final_rmse:.2f}")
        print(f"Test R²: {final_r2:.4f}")
        
    except FileNotFoundError:
        print("\nGround truth file not found. Predictions saved to 'distilled_predictions.csv'")
    
    print(f"\nPredictions saved to 'distilled_predictions.csv'")
    print(f"Number of predictions: {len(submission)}")
    
    return xgb_model, lgb_model, best_student, submission

if __name__ == "__main__":
    xgb_model, lgb_model, student_model, predictions = main()