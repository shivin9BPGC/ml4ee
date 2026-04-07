import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import warnings
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. ADVANCED FEATURE ENGINEERING
# ==========================================

def get_macro_features(df):
    """Enriches with GDP, Tech Scores, and Inflation data."""
    
    # 2024 GDP PPP Estimates (approx)
    gdp_map = {
        'United States Of America': 76000, 'Switzerland': 83000, 'Ireland': 120000,
        'Norway': 78000, 'United Arab Emirates': 87000, 'Denmark': 74000,
        'Netherlands': 69000, 'Germany': 63000, 'Sweden': 60000,
        'Australia': 62000, 'Belgium': 61000, 'Canada': 58000,
        'United Kingdom Of Great Britain And Northern Ireland': 55000,
        'France': 54000, 'Japan': 46000, 'Italy': 50000,
        'Korea (Republic Of)': 53000, 'Spain': 45000, 'New Zealand': 48000,
        'Czech Republic': 45000, 'Poland': 43000, 'Portugal': 40000,
        'Saudi Arabia': 55000, 'Hungary': 39000, 'Turkey': 37000,
        'Russian Federation': 35000, 'Malaysia': 33000, 'Panama': 32000,
        'China': 23000, 'Thailand': 20000, 'Mexico': 22000,
        'Brazil': 18000, 'South Africa': 15000, 'Colombia': 17000,
        'Indonesia': 15000, 'India': 9000, 'Philippines': 10000,
        'Pakistan': 6000, 'Nigeria': 5000, 'Argentina': 26000,
        'Taiwan, Province Of China': 65000, 'Israel': 52000,
        'Singapore': 127000, 'Austria': 64000, 'Finland': 58000
    }
    
    # Tech Hub Score (1-10)
    tech_score_map = {
        'United States Of America': 10, 'Israel': 9.5, 'Switzerland': 9.5,
        'United Kingdom Of Great Britain And Northern Ireland': 8.5,
        'Germany': 8.5, 'Canada': 8.0, 'India': 6.0, 'China': 7.0,
        'Russian Federation': 6.5, 'Poland': 6.0, 'Ukraine': 5.5,
        'Singapore': 9.0, 'Sweden': 8.0, 'Netherlands': 8.0
    }

    df['gdp_ppp'] = df['country'].map(gdp_map).fillna(35000)
    df['tech_score'] = df['country'].map(tech_score_map).fillna(5.0)
    df['wealth_index'] = df['gdp_ppp'] * df['tech_score']
    
    return df

def get_department(role):
    """Groups roles into departments."""
    if role in ['accountant', 'budget-finance-analyst', 'treasury-analyst', 'payroll-specialist']:
        return 'Finance'
    elif role in ['procurement-specialist', 'supply-chain-specialist']:
        return 'Operations'
    elif role == 'specialist-human-resources':
        return 'HR'
    elif role == 'marketing-specialist':
        return 'Marketing'
    elif role == 'automation-analyst':
        return 'Tech'
    return 'Other'

# ==========================================
# 2. PREPROCESSING
# ==========================================

def preprocess_data():
    print("Loading Data...")
    train = pd.read_csv('../data/q1/train.csv')
    test = pd.read_csv('../data/q1/test.csv')
    col_df = pd.read_csv('../data/q1/cost_of_living.csv')

    train = train.dropna(subset=['salary_average'])

    # --- 1. Cost of Living Clustering ---
    # Create broad "Cost Clusters" (e.g., Expensive Western vs Cheap Asian)
    # This helps even if we don't know the specific city.
    cost_cols = [c for c in col_df.columns if c.startswith('col_')]
    
    # Fill missing CoL data for clustering
    imputer = SimpleImputer(strategy='median')
    col_matrix = imputer.fit_transform(col_df[cost_cols])
    
    # Cluster cities into 8 types
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    col_df['city_cluster'] = kmeans.fit_predict(col_matrix)
    
    # Also calculate a simple scalar index
    col_df['col_index'] = np.median(col_matrix, axis=1)

    # Aggregate to State/Country level (The Fix for Cold Start)
    state_agg = col_df.groupby(['country', 'state'])[['col_index', 'city_cluster']].median().reset_index()
    country_agg = col_df.groupby(['country'])[['col_index', 'city_cluster']].median().reset_index()

    # --- 2. Process Datasets ---
    processed_dfs = []
    
    # Calculate role freq on TRAIN only
    role_counts = train['role'].value_counts().to_dict()

    for df in [train, test]:
        # Merge CoL data (State priority, then Country)
        df = pd.merge(df, state_agg, on=['country', 'state'], how='left', suffixes=('', '_state'))
        df = pd.merge(df, country_agg, on=['country'], how='left', suffixes=('', '_country'))
        
        # Coalesce: Take State value, if null take Country value, if null take Global median
        df['col_index'] = df['col_index'].fillna(df['col_index_country']).fillna(col_df['col_index'].median())
        df['city_cluster'] = df['city_cluster'].fillna(df['city_cluster_country']).fillna(4).astype(int)
        
        # Cleanup merge columns
        cols_to_drop = [c for c in df.columns if '_country' in c or '_state' in c]
        df.drop(columns=cols_to_drop, inplace=True)
        
        # Features
        df = get_macro_features(df)
        df['department'] = df['role'].apply(get_department)
        df['role_freq'] = df['role'].map(role_counts).fillna(10)
        
        processed_dfs.append(df)

    return processed_dfs[0], processed_dfs[1]

# ==========================================
# 3. OOF STACKING (The Magic)
# ==========================================

def train_hybrid_ensemble(train_df, test_df, col_df):
    
    y = train_df['salary_average'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # --- INPUT A: DENSE FEATURES (For Trees) ---
    # (Use your existing preprocess logic to get X_dense)
    # X_dense_train, X_dense_test = ...
    
    # --- INPUT B: SPARSE FEATURES (For NN) ---
    # We must join col_df to train/test first
    # (Assuming simple merge on country/state for col data)
    
    # Define NN Architecture (Exact match to 19.56% result)
    nn_pipeline = Pipeline([
        ('preprocessor', get_sparse_pipeline()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(32, 32),
                             activation='relu',
                             alpha=0.01,
                             learning_rate='adaptive',
                             random_state=42,
                             max_iter=500,
                             early_stopping=True))
    ])
    
    # --- TRAINING LOOP ---
    print("Training Hybrid Ensemble...")
    
    oof_preds_xgb = np.zeros(len(train_df))
    oof_preds_nn = np.zeros(len(train_df))
    
    test_preds_xgb = np.zeros(len(test_df))
    test_preds_nn = np.zeros(len(test_df))

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, y)):
        # Split Data
        X_tr_dense, X_val_dense = X_dense_train[train_idx], X_dense_train[val_idx]
        df_tr_raw, df_val_raw = train_df.iloc[train_idx], train_df.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # 1. Train Trees (on Dense Data)
        xgb_model = xgb.XGBRegressor(...)
        xgb_model.fit(X_tr_dense, y_tr)
        oof_preds_xgb[val_idx] = xgb_model.predict(X_val_dense)
        test_preds_xgb += xgb_model.predict(X_dense_test) / 5

        # 2. Train NN (on Raw/Sparse Data)
        nn_pipeline.fit(df_tr_raw, y_tr)
        oof_preds_nn[val_idx] = nn_pipeline.predict(df_val_raw)
        test_preds_nn += nn_pipeline.predict(test_df) / 5
        
    # --- META LEARNER ---
    X_stack = np.column_stack([oof_preds_xgb, oof_preds_nn])
    X_stack_test = np.column_stack([test_preds_xgb, test_preds_nn])
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_stack, y)
    
    final_preds = meta_model.predict(X_stack_test)
    
    print(f"Weights: XGB={meta_model.coef_[0]:.2f}, NN={meta_model.coef_[1]:.2f}")
    return final_preds

# def train_stacking_ensemble(train_df, test_df):
    
#     # 1. Prepare Features
#     # Numeric features for Scaling/PCA
#     num_cols = ['gdp_ppp', 'tech_score', 'wealth_index', 'col_index', 'role_freq']
#     # Cost columns for PCA (assuming they are merged back or we use col_index)
#     # Note: If you want to use the PCA logic from the script, we need the raw col_ columns.
#     # For this implementation, we will use the features we engineered which are already robust.
    
#     cat_cols = ['role', 'department', 'country', 'state'] # High cardinality for NN
    
#     # 2. Define The Neural Network Pipeline (The "New Teacher")
#     # It needs its own preprocessor because NNs require Scaling, Trees don't.
#     nn_preprocessor = ColumnTransformer([
#         ('num', StandardScaler(), num_cols),
#         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
#     ])
    
#     nn_model = Pipeline([
#         ('preprocessor', nn_preprocessor),
#         ('pca', PCA(n_components=30)), # Optional: Reduce categorical dimensionality
#         ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), 
#                              activation='relu',
#                              alpha=0.01, # L2 Regularization
#                              random_state=42, 
#                              max_iter=500))
#     ])

#     # 3. Prepare Tree Data (Raw features are fine)
#     encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#     X_cat_train = encoder.fit_transform(train_df[['role', 'department']]) # Less cardinality for Trees
#     X_cat_test = encoder.transform(test_df[['role', 'department']])
    
#     X_num_train = train_df[num_cols].values
#     X_num_test = test_df[num_cols].values
    
#     X = np.hstack([X_num_train, X_cat_train])
#     X_test = np.hstack([X_num_test, X_cat_test])
#     y = train_df['salary_average'].values

#     # 4. Out-of-Fold Stacking
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
#     # Storage for OOF Predictions
#     oof_xgb = np.zeros(X.shape[0])
#     oof_lgb = np.zeros(X.shape[0])
#     oof_nn  = np.zeros(X.shape[0]) # New NN OOF
    
#     test_pred_xgb = np.zeros(X_test.shape[0])
#     test_pred_lgb = np.zeros(X_test.shape[0])
#     test_pred_nn  = np.zeros(X_test.shape[0]) # New NN Test Preds

#     print("Training 3-Teacher Ensemble (XGB + LGBM + NN)...")

#     for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#         # Data Splitting
#         X_tr, X_val = X[train_idx], X[val_idx]
#         y_tr, y_val = y[train_idx], y[val_idx]
        
#         # DF splitting for NN (needs DataFrame input for Pipeline)
#         df_tr, df_val = train_df.iloc[train_idx], train_df.iloc[val_idx]

#         # --- Teacher 1: XGBoost ---
#         xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, n_jobs=-1, random_state=42)
#         xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
#         oof_xgb[val_idx] = xgb_model.predict(X_val)
#         test_pred_xgb += xgb_model.predict(X_test) / 5
        
#         # --- Teacher 2: LightGBM ---
#         lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, n_jobs=-1, verbose=-1, random_state=42)
#         lgb_model.fit(X_tr, y_tr)
#         oof_lgb[val_idx] = lgb_model.predict(X_val)
#         test_pred_lgb += lgb_model.predict(X_test) / 5

#         # --- Teacher 3: Neural Network (The New Star) ---
#         nn_model.fit(df_tr, y_tr)
#         oof_nn[val_idx] = nn_model.predict(df_val)
#         test_pred_nn += nn_model.predict(test_df) / 5
        
#         print(f"  Fold {fold+1} complete.")

#     # 5. Student Layer (Meta-Learner)
#     print("Training Student (Ridge)...")
    
#     # Stack all 3 predictions
#     X_stack_train = np.column_stack([oof_xgb, oof_lgb, oof_nn])
#     X_stack_test = np.column_stack([test_pred_xgb, test_pred_lgb, test_pred_nn])
    
#     # We can also add original features to the stack for "Residual Learning"
#     # But usually, just the predictions are enough for a strong student.
    
#     student = Ridge(alpha=10.0)
#     student.fit(X_stack_train, y)
    
#     final_preds = student.predict(X_stack_test)
#     final_preds = np.maximum(final_preds, 5000)
    
#     # Check Weights (Interpretation)
#     print(f"\nStudent Weights -> XGB: {student.coef_[0]:.2f}, LGB: {student.coef_[1]:.2f}, NN: {student.coef_[2]:.2f}")
    
#     return final_preds

# ==========================================
# 4. EXECUTION
# ==========================================

train_df, test_df = preprocess_data()
predictions = train_stacking_ensemble(train_df, test_df)

# Save
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'salary_average': predictions
})
submission.to_csv('final_oof_submission.csv', index=False)
print("Saved to 'final_oof_submission.csv'")

# Local Evaluation if solution exists
try:
    solution = pd.read_csv('./q1_solution.csv') # OR '../data/q1/solution.csv'
    merged = pd.merge(submission, solution, on='ID')
    
    def rmpse(y_true, y_pred):
        return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

    score = rmpse(merged['salary_average_y'], merged['salary_average_x'])
    print(f"\n=== FINAL SCORE ===\nRMSPE: {score:.4%}")
except:
    pass