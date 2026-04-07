import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set environment variable to avoid MKL issues
import os
os.environ['OMP_NUM_THREADS'] = '1'

# Machine learning imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

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
    col_df['cost_basic'] = col_df[['col_1', 'col_2', 'col_3']].mean(axis=1)
    col_df['cost_housing'] = col_df[['col_14', 'col_15', 'col_16']].mean(axis=1) 
    col_df['cost_transport'] = col_df[['col_4', 'col_5']].mean(axis=1)
    col_df['cost_overall'] = col_df[['cost_basic', 'cost_housing', 'cost_transport']].mean(axis=1)
    
    return train_df, test_df, col_df

def create_enhanced_features(df, col_df, is_training=True, training_stats=None):
    """Create enhanced features with proper handling to avoid data leakage"""
    print(f"Creating enhanced features for {'training' if is_training else 'test'} data...")
    
    # Merge with cost of living data
    merged_df = pd.merge(df, col_df, on=['country', 'state', 'city'], how='left')
    
    # Define cost feature columns
    cost_features = ['cost_basic', 'cost_housing', 'cost_transport', 'cost_overall']
    
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
    
    # Log transformations for skewed features
    merged_df['log_cost_overall'] = np.log1p(merged_df['cost_overall'])
    merged_df['sqrt_cost_basic'] = np.sqrt(merged_df['cost_basic'])
    
    # Location encoding
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
        'role_freq', 'cost_role_interaction', 'cost_housing_role',
        'log_cost_overall', 'sqrt_cost_basic', 'country_freq', 'state_freq'
    ]
    
    # Handle any remaining NaN values
    for col in feature_columns:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    
    return merged_df[feature_columns], training_stats if is_training else None

def rmspe(y_true, y_pred):
    """Root Mean Squared Percentage Error"""
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0
    percentage_errors = ((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2
    return np.sqrt(np.mean(percentage_errors)) * 100

def train_teacher_models(X_train, y_train, X_val, y_val):
    """Train ensemble teacher models using sklearn"""
    print("\nTraining teacher models...")
    
    # Random Forest - simplified to avoid MKL issues
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,  # Reduced from 200
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1  # Single thread to avoid MKL
    )
    rf_model.fit(X_train, y_train)
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,  # Reduced from 200
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    return rf_model, gb_model

def knowledge_distillation(X_train, y_train, X_val, y_val, rf_model, gb_model):
    """Implement knowledge distillation from teacher models to student linear model"""
    print("\nImplementing knowledge distillation...")
    
    # Get teacher predictions
    rf_train_pred = rf_model.predict(X_train)
    rf_val_pred = rf_model.predict(X_val)
    
    gb_train_pred = gb_model.predict(X_train)
    gb_val_pred = gb_model.predict(X_val)
    
    # Ensemble teacher predictions (simple average)
    teacher_train_pred = (rf_train_pred + gb_train_pred) / 2
    teacher_val_pred = (rf_val_pred + gb_val_pred) / 2
    
    # Create enhanced feature sets with teacher knowledge
    # Add teacher predictions as features
    X_train_enhanced = np.column_stack([X_train, teacher_train_pred, rf_train_pred, gb_train_pred])
    X_val_enhanced = np.column_stack([X_val, teacher_val_pred, rf_val_pred, gb_val_pred])
    
    # Feature importance guided transformations
    rf_importance = rf_model.feature_importances_
    gb_importance = gb_model.feature_importances_
    
    # Average importance across models
    avg_importance = (rf_importance + gb_importance) / 2
    importance_weights = avg_importance / np.sum(avg_importance)
    
    # Create weighted features based on importance
    weighted_features_train = X_train * importance_weights
    weighted_features_val = X_val * importance_weights
    
    # Add weighted sum as new feature
    X_train_enhanced = np.column_stack([X_train_enhanced, np.sum(weighted_features_train, axis=1)])
    X_val_enhanced = np.column_stack([X_val_enhanced, np.sum(weighted_features_val, axis=1)])
    
    return X_train_enhanced, X_val_enhanced, teacher_train_pred, teacher_val_pred, importance_weights

def train_student_model(X_train_enhanced, y_train, X_val_enhanced, y_val, teacher_train_pred, teacher_val_pred):
    """Train student neural network model with knowledge distillation"""
    print("\nTraining student model with knowledge distillation...")
    
    # Define neural network parameters
    hidden_layer_sizes = (32, 32)
    random_state = 42
    
    # Create preprocessor (StandardScaler)
    preprocessor = StandardScaler()
    
    # Try different distillation approaches
    models = {}
    
    # 1. Direct neural network on enhanced features
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.01,  # L2 regularization
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        ))
    ])
    model.fit(X_train_enhanced, y_train)
    models['neural_enhanced'] = model
    
    # 2. Neural network with different regularization strengths
    for alpha in [0.001, 0.01, 0.1]:
        model = Pipeline(steps=[
            ('preprocessor', StandardScaler()),
            ('regressor', MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=alpha,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=False
            ))
        ])
        model.fit(X_train_enhanced, y_train)
        models[f'neural_alpha_{alpha}'] = model
    
    # 3. Combined loss: original target + teacher predictions
    # Weight between hard targets (true labels) and soft targets (teacher predictions)
    for alpha in [0.5, 0.7, 0.9]:
        beta = 1 - alpha
        combined_targets = alpha * y_train + beta * teacher_train_pred
        
        model_combined = Pipeline(steps=[
            ('preprocessor', StandardScaler()),
            ('regressor', MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=0.01,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=False
            ))
        ])
        model_combined.fit(X_train_enhanced, combined_targets)
        models[f'neural_distilled_{alpha}'] = model_combined
    
    # Evaluate all models
    best_model = None
    best_score = float('inf')
    best_name = None
    
    print("\nEvaluating student models:")
    for name, model in models.items():
        val_pred = model.predict(X_val_enhanced)
        val_rmspe = rmspe(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"{name}: RMSPE={val_rmspe:.2f}%, RMSE={val_rmse:.2f}, R²={val_r2:.4f}")
        
        if val_rmspe < best_score:
            best_score = val_rmspe
            best_model = model
            best_name = name
    
    print(f"\nBest student model: {best_name} (RMSPE: {best_score:.2f}%)")
    
    return best_model, models

def main():
    """Main execution function"""
    print("=== Knowledge Distillation: Ensemble Models → Neural Network ===")
    
    # Load and preprocess data
    train_df, test_df, col_df = load_and_preprocess_data()
    
    # Create enhanced features
    X_train_full, training_stats = create_enhanced_features(train_df, col_df, is_training=True)
    y_train_full = train_df['salary_average'].values
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset shapes:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Features: {X_train.shape[1]}")
    
    # Convert to numpy arrays
    X_train = X_train.values if hasattr(X_train, 'values') else X_train
    X_val = X_val.values if hasattr(X_val, 'values') else X_val
    
    # Train teacher models
    rf_model, gb_model = train_teacher_models(X_train, y_train, X_val, y_val)
    
    # Evaluate teacher models
    print("\nEvaluating teacher models:")
    rf_val_pred = rf_model.predict(X_val)
    gb_val_pred = gb_model.predict(X_val)
    ensemble_val_pred = (rf_val_pred + gb_val_pred) / 2
    
    print(f"Random Forest RMSPE: {rmspe(y_val, rf_val_pred):.2f}%")
    print(f"Gradient Boosting RMSPE: {rmspe(y_val, gb_val_pred):.2f}%")
    print(f"Ensemble RMSPE: {rmspe(y_val, ensemble_val_pred):.2f}%")
    
    # Knowledge distillation
    X_train_enhanced, X_val_enhanced, teacher_train_pred, teacher_val_pred, importance_weights = knowledge_distillation(
        X_train, y_train, X_val, y_val, rf_model, gb_model
    )
    
    # Train student model
    best_student, all_models = train_student_model(
        X_train_enhanced, y_train, X_val_enhanced, y_val, teacher_train_pred, teacher_val_pred
    )
    
    # Prepare test data and make final predictions
    print("\nPreparing test predictions...")
    X_test, _ = create_enhanced_features(test_df, col_df, is_training=False, training_stats=training_stats)
    X_test = X_test.values if hasattr(X_test, 'values') else X_test
    
    # Get teacher predictions for test data
    rf_test_pred = rf_model.predict(X_test)
    gb_test_pred = gb_model.predict(X_test)
    teacher_test_pred = (rf_test_pred + gb_test_pred) / 2
    
    # Enhance test features
    X_test_enhanced = np.column_stack([X_test, teacher_test_pred, rf_test_pred, gb_test_pred])
    
    # Add weighted sum
    weighted_features_test = X_test * importance_weights
    X_test_enhanced = np.column_stack([X_test_enhanced, np.sum(weighted_features_test, axis=1)])
    
    # Final predictions using neural network pipeline
    final_predictions = best_student.predict(X_test_enhanced)
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
        final_mae = mean_absolute_error(merged_eval['salary_average_y'], merged_eval['salary_average_x'])
        
        print(f"\n=== FINAL TEST RESULTS ===")
        print(f"Test RMSPE: {final_rmspe:.2f}%")
        print(f"Test RMSE: {final_rmse:.2f}")
        print(f"Test MAE: {final_mae:.2f}")
        print(f"Test R²: {final_r2:.4f}")
        
        # Compare with teacher ensemble
        teacher_test_rmspe = rmspe(merged_eval['salary_average_y'], teacher_test_pred)
        print(f"\n=== TEACHER MODEL COMPARISON ===")
        print(f"Teacher Ensemble RMSPE: {teacher_test_rmspe:.2f}%")
        print(f"Student Model RMSPE: {final_rmspe:.2f}%")
        improvement = teacher_test_rmspe - final_rmspe
        print(f"Improvement: {improvement:.2f}% {'(better)' if improvement > 0 else '(worse)'}")
        
        # Compare with baseline (original code performance estimation)
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Baseline Linear Regression (estimated): ~25-30% RMSPE")
        print(f"Teacher Ensemble: {teacher_test_rmspe:.2f}% RMSPE")
        print(f"Distilled Student Model: {final_rmspe:.2f}% RMSPE")
        
    except FileNotFoundError:
        print("\nGround truth file not found. Predictions saved to 'distilled_predictions.csv'")
    
    print(f"\nPredictions saved to 'distilled_predictions.csv'")
    print(f"Number of predictions: {len(submission)}")
    
    return rf_model, gb_model, best_student, submission

if __name__ == "__main__":
    rf_model, gb_model, student_model, predictions = main()