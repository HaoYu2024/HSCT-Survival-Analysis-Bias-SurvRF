
# Re-import necessary libraries after execution state reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from lifelines.utils import concordance_index

# Reload the dataset
train_file_path = "train.csv"
train_data = pd.read_csv(train_file_path)

# Identify categorical and numerical features
numerical_features = train_data.select_dtypes(include=['number']).columns.tolist()
categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()

# Remove target variables from numerical features
numerical_features = [col for col in numerical_features if col not in ["efs", "efs_time", "ID"]]

# Encode categorical variables using Label Encoding
encoded_data = train_data.copy()
for col in categorical_features:
    encoded_data[col] = LabelEncoder().fit_transform(encoded_data[col].astype(str))

# Handle missing values using median for numerical and mode for categorical
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

encoded_data[numerical_features] = num_imputer.fit_transform(encoded_data[numerical_features])
encoded_data[categorical_features] = cat_imputer.fit_transform(encoded_data[categorical_features])

# Define features and target
X = encoded_data.drop(columns=["efs", "efs_time", "ID"])
y = encoded_data["efs_time"]  # Use time-to-event as the target

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define multiple XGBoost models with different characteristics
def create_model_ensemble():
    models = {
        'deep_trees': xgb.XGBRegressor(
            objective="survival:cox",
            n_estimators=100,
            max_depth=6,        # Deeper trees for complex patterns
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.6,
            min_child_weight=2,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            tree_method="hist"
        ),
        'shallow_trees': xgb.XGBRegressor(
            objective="survival:cox",
            n_estimators=100,
            max_depth=2,        # Shallow trees for robust patterns
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=43,
            tree_method="hist"
        ),
        'regularized': xgb.XGBRegressor(
            objective="survival:cox",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=4,
            reg_alpha=2.0,      # Heavy regularization
            reg_lambda=2.0,
            random_state=44,
            tree_method="hist"
        ),
        'high_samples': xgb.XGBRegressor(
            objective="survival:cox",
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.95,     # Use more samples
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_alpha=0.3,
            reg_lambda=0.3,
            random_state=45,
            tree_method="hist"
        )
    }
    return models

# Train all models and compute individual C-indices
def train_ensemble(models, X_train, y_train, X_val, y_val, event_indicators):
    trained_models = {}
    c_indices = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Get predictions and calculate C-index
        val_pred = model.predict(X_val)
        c_index = concordance_index(y_val, -val_pred, event_indicators)
        
        trained_models[name] = model
        c_indices[name] = c_index
        print(f"{name} C-index: {c_index:.4f}")
    
    return trained_models, c_indices

# Blend predictions using different strategies
def blend_predictions(trained_models, X, weights=None):
    all_preds = []
    for name, model in trained_models.items():
        preds = model.predict(X)
        all_preds.append(preds)
    
    if weights is None:
        # Simple average
        return np.mean(all_preds, axis=0)
    else:
        # Weighted average
        return np.average(all_preds, axis=0, weights=weights)

# Main execution
def run_ensemble_model():
    # Create and train models
    models = create_model_ensemble()
    trained_models, c_indices = train_ensemble(
        models, 
        X_train, 
        y_train, 
        X_val, 
        y_val,
        encoded_data.loc[y_val.index, "efs"]
    )
    
    # Try different blending strategies
    # 1. Simple average
    simple_blend_pred = blend_predictions(trained_models, X_val)
    simple_blend_cindex = concordance_index(
        y_val, 
        -simple_blend_pred, 
        encoded_data.loc[y_val.index, "efs"]
    )
    
    # 2. Weighted average based on individual performance
    weights = np.array([v for v in c_indices.values()])
    weights = weights / np.sum(weights)  # Normalize
    weighted_blend_pred = blend_predictions(trained_models, X_val, weights)
    weighted_blend_cindex = concordance_index(
        y_val, 
        -weighted_blend_pred, 
        encoded_data.loc[y_val.index, "efs"]
    )
    
    # Print results
    print("\nEnsemble Results:")
    print(f"Individual model C-indices: {c_indices}")
    print(f"Simple average blend C-index: {simple_blend_cindex:.4f}")
    print(f"Weighted average blend C-index: {weighted_blend_cindex:.4f}")
    
    return trained_models, c_indices, simple_blend_cindex, weighted_blend_cindex

# Run the ensemble
trained_models, c_indices, simple_blend_cindex, weighted_blend_cindex = run_ensemble_model()

# Visualize results
plt.figure(figsize=(10, 6))
c_indices_list = list(c_indices.values()) + [simple_blend_cindex, weighted_blend_cindex]
names = list(c_indices.keys()) + ['Simple Blend', 'Weighted Blend']

plt.bar(range(len(c_indices_list)), c_indices_list)
plt.xticks(range(len(c_indices_list)), names, rotation=45)
plt.ylabel('C-index')
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.show()