import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.stats import uniform, randint
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
    # Create copies
    train_processed = train_df.copy()
    col_processed = col_df.copy()
    
    # Drop ID and handle missing target
    train_processed = train_processed.drop(['ID'], axis=1)
    train_processed = train_processed.dropna(subset=['salary_average'])
    
    # Separate target
    y = train_processed['salary_average']
    X = train_processed.drop(['salary_average'], axis=1)
    
    # Aggregate cost of living data at state level to avoid city leakage
    # Use first 15 col features (reduces dimensionality while keeping important info)
    col_cols = [f'col_{i}' for i in range(1, 16) if f'col_{i}' in col_processed.columns]
    
    col_state_agg = col_processed.groupby(['country', 'state'])[col_cols].median().reset_index()
    
    # Fill missing values with global median for each column
    for col in col_cols:
        if col in col_state_agg.columns:
            col_state_agg[col] = col_state_agg[col].fillna(col_processed[col].median())
    
    # Merge with training data on country and state (NOT city)
    X_merged = pd.merge(
        X, 
        col_state_agg, 
        on=['country', 'state'], 
        how='left',
        suffixes=('', '_col')
    )
    
    # Drop city column to prevent leakage
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
        
        # Align columns with training data
        test_merged = test_merged.reindex(columns=X_merged.columns, fill_value=0)
        
        return X_merged, y, test_merged, test_ids
    
    return X_merged, y, None, None

def reduce_col_features_pca(X, n_components=5):
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

def create_neural_network_pipeline(numerical_features, categorical_features, 
                                   hidden_layer_sizes=(32, 32), random_state=42):
    """
    Create a neural network pipeline with preprocessing
    """
    # Preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Neural network model - 2 layer architecture (32-32)
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
    
    return model

def optimize_hyperparameters(X, y, numerical_features, categorical_features):
    """
    Optimize hyperparameters using random search
    """
    # Create base pipeline
    base_pipeline = create_neural_network_pipeline(
        numerical_features, categorical_features, random_state=42
    )
    
    # Define parameter grid for random search
    param_dist = {
        'regressor__hidden_layer_sizes': [(32, 32), (64, 32), (32, 16), (64, 64)],
        'regressor__alpha': uniform(0.0001, 0.1),  # Regularization
        'regressor__learning_rate_init': uniform(0.0001, 0.01),
        'regressor__batch_size': randint(32, 256)
    }
    
    # Randomized search with cross-validation
    random_search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("Optimizing hyperparameters...")
    random_search.fit(X, y)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {-random_search.best_score_:.2f}")
    
    return random_search.best_estimator_

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('../data/q1/train.csv')
    col_df = pd.read_csv('../data/q1/cost_of_living.csv')
    test_df = pd.read_csv('../data/q1/test.csv')
    
    # Preprocess and merge data
    print("Preprocessing data...")
    X_train, y_train, X_test, test_ids = preprocess_and_merge_data(
        train_df, col_df, test_df
    )
    
    # Reduce cost of living features with PCA
    print("Reducing cost of living features with PCA...")
    X_train_reduced = reduce_col_features_pca(X_train, n_components=5)
    X_test_reduced = reduce_col_features_pca(X_test, n_components=5)
    
    # Split for validation
    print("Splitting data for validation...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_reduced, y_train, test_size=0.2, random_state=42
    )
    
    # Identify feature types
    categorical_features = ['country', 'state', 'role']
    numerical_features = [col for col in X_train_reduced.columns 
                         if col not in categorical_features]
    
    # Create and train neural network
    print("\nTraining neural network...")
    
    # Option 1: Use optimized model
    # best_model = optimize_hyperparameters(
    #     X_train_split, y_train_split, numerical_features, categorical_features
    # )
    
    # Option 2: Use base model (faster)
    nn_model = create_neural_network_pipeline(
        numerical_features, categorical_features, hidden_layer_sizes=(32, 32)
    )
    
    # Train the model
    print("Fitting model...")
    nn_model.fit(X_train_split, y_train_split)
    
    # Evaluate on validation set
    y_val_pred = nn_model.predict(X_val)
    rmpse_val = calculate_rmpse(y_val, y_val_pred)
    print(f"\nValidation RMPSE: {rmpse_val:.4f}")
    
    # Cross-validation score
    print("\nPerforming cross-validation...")
    cv_scores = cross_val_score(
        nn_model, X_train_reduced, y_train,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    cv_rmse = np.sqrt(-cv_scores)
    print(f"CV RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = create_neural_network_pipeline(
        numerical_features, categorical_features, hidden_layer_sizes=(32, 32)
    )
    final_model.fit(X_train_reduced, y_train)
    
    # Make predictions on test set
    print("Making predictions on test set...")
    test_predictions = final_model.predict(X_test_reduced)
    
    # Ensure positive predictions (salaries can't be negative)
    test_predictions = np.maximum(test_predictions, 0)
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': test_ids,
        'salary_average': test_predictions
    })
    
    # Save predictions
    submission.to_csv('neural_network_predictions.csv', index=False)
    print(f"\nPredictions saved to 'neural_network_predictions.csv'")
    print(f"Number of predictions: {len(submission)}")
    print(f"Prediction range: {test_predictions.min():.2f} to {test_predictions.max():.2f}")
    
    # Print model architecture summary
    print("\n" + "="*50)
    print("NEURAL NETWORK ARCHITECTURE SUMMARY")
    print("="*50)
    print("Architecture: Input -> 32 neurons -> 32 neurons -> Output")
    print("Activation: ReLU for hidden layers, Linear for output")
    print("Regularization: L2 (alpha=0.01)")
    print("Optimizer: Adam with adaptive learning rate")
    print("Early stopping: Enabled")
    print("="*50)
    
    return final_model, submission

def analyze_feature_importance(model, X_train, numerical_features, categorical_features):
    """
    Analyze feature importance from the neural network
    """
    # Get the preprocessor and neural network from pipeline
    preprocessor = model.named_steps['preprocessor']
    nn = model.named_steps['regressor']
    
    # Transform the data
    X_transformed = preprocessor.transform(X_train)
    
    # For neural networks, we can look at the weights of the first layer
    if hasattr(nn, 'coefs_') and len(nn.coefs_) > 0:
        # Get absolute weights from input layer
        input_weights = np.abs(nn.coefs_[0])
        
        # Calculate feature importance as mean absolute weight per feature
        feature_importance = np.mean(input_weights, axis=1)
        
        # Get feature names after preprocessing
        # This is complex with ColumnTransformer, but we can get approximate names
        print("\nTop 10 most important features (approximate):")
        
        # Get numerical feature names
        num_feature_names = numerical_features
        
        # Estimate categorical feature count
        categorical_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        if hasattr(categorical_encoder, 'categories_'):
            cat_feature_count = sum(len(cats) for cats in categorical_encoder.categories_)
        else:
            cat_feature_count = 0
        
        # Print information
        print(f"Numerical features: {len(num_feature_names)}")
        print(f"Categorical features (after encoding): ~{cat_feature_count}")
        print(f"Total features after preprocessing: {X_transformed.shape[1]}")
    
    return None

if __name__ == "__main__":
    model, predictions = main()
    
    # Optional: Load data again for feature analysis
    train_df = pd.read_csv('../data/q1/train.csv')
    col_df = pd.read_csv('../data/q1/cost_of_living.csv')
    X_train, y_train, _, _ = preprocess_and_merge_data(train_df, col_df)
    X_train_reduced = reduce_col_features_pca(X_train, n_components=5)
    
    # Identify feature types
    categorical_features = ['country', 'state', 'role']
    numerical_features = [col for col in X_train_reduced.columns 
                         if col not in categorical_features]

    # Local Evaluation if solution exists
    try:
        solution = pd.read_csv('./q1_solution.csv') # OR '../data/q1/solution.csv'
        merged = pd.merge(predictions, solution, on='ID')
        
        def rmpse(y_true, y_pred):
            return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

        score = rmpse(merged['salary_average_y'], merged['salary_average_x'])
        print(f"\n=== FINAL SCORE ===\nRMSPE: {score:.4%}")
    except:
        pass
    # Analyze feature importance
    analyze_feature_importance(model, X_train_reduced, numerical_features, categorical_features)