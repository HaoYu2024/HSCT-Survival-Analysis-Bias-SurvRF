from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sksurv.metrics import concordance_index_censored
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.base import clone


import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored
import gc

import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored
import gc

def load_feature_importance():
    """Load the previously calculated feature importance scores"""
    return pd.read_csv('feature_importance.csv', index_col=0)

def select_top_features(importance_df, threshold=0.4):
    """Select features above a certain concordance index threshold"""
    return importance_df[importance_df['0'] > threshold].index.tolist()

def preprocess_features(data, features):
    """Preprocess the features with proper type handling"""
    # Identify numeric and categorical columns
    numeric_features = ['age_at_hct']
    categorical_features = [f for f in features if f not in numeric_features + ['efs']]
    
    # Initialize imputers
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    # Handle numeric features
    if numeric_features:
        data[numeric_features] = num_imputer.fit_transform(data[numeric_features])
        data[numeric_features] = data[numeric_features].astype(np.float32)
    
    # Handle categorical features
    if categorical_features:
        # Convert to string first to handle any numeric categories
        for cat_col in categorical_features:
            data[cat_col] = data[cat_col].astype(str)
        
        # Impute missing values
        data[categorical_features] = cat_imputer.fit_transform(data[categorical_features])
        
        # Encode categorical variables
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        data[categorical_features] = encoder.fit_transform(data[categorical_features])
        data[categorical_features] = data[categorical_features].astype(np.float32)
    
    return data

def prepare_data_for_prediction(data_path, important_features):
    """Prepare the data using only the important features"""
    # Add 'efs' and 'age_at_hct' as they're needed
    features_to_load = important_features + ['efs', 'age_at_hct']
    features_to_load = list(set(features_to_load))  # Remove duplicates
    
    # Load data
    train_data = pd.read_csv(data_path, usecols=features_to_load)
    
    # Preprocess features
    train_data = preprocess_features(train_data, features_to_load)
    
    # Create the survival target
    y = np.array(
        [(bool(event), float(time)) 
         for event, time in zip(train_data["efs"], train_data["age_at_hct"])],
        dtype=[("event", "bool"), ("time", "float32")]
    )
    
    # Prepare features (exclude target variables)
    X = train_data.drop(columns=["efs", "age_at_hct"])
    
    # Ensure all features are float32
    X = X.astype(np.float32)
    
    # Verify data is not empty
    if X.empty:
        raise ValueError("No features available for training")
    
    return X, y

def train_optimized_model(X, y):
    """Train a RandomSurvivalForest model with optimized parameters"""
    if X.empty:
        raise ValueError("Empty feature set provided for training")
        
    model = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=20,
        min_samples_leaf=50,
        max_features="sqrt",
        random_state=42,
        n_jobs=1
    )
    
    print(f"Training with features: {X.columns.tolist()}")
    print(f"Number of samples: {len(X)}")
    
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """Evaluate the model's performance"""
    predictions = model.predict(X)
    c_index = concordance_index_censored(y["event"], y["time"], predictions)[0]
    return c_index

def plot_survival_curves(model, X, num_samples=5):
    """Plot survival curves for a few samples"""
    plt.figure(figsize=(10, 6))
    
    for i in range(min(num_samples, len(X))):
        surv_funcs = model.predict_survival_function(X.iloc[[i]])
        for surv_func in surv_funcs:
            plt.step(surv_func.x, surv_func.y, where="post", 
                    label=f"Sample {i+1}")
    
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.grid(True)
    plt.legend()
    plt.title("Survival Curves for Selected Samples")
    plt.savefig('survival_curves.png')
    plt.close()

def main():
    try:
        # Load feature importance scores
        print("Loading feature importance...")
        importance_df = load_feature_importance()
        
        # Select important features
        important_features = select_top_features(importance_df, threshold=0.4)
        print(f"Selected features: {important_features}")
        
        if not important_features:
            raise ValueError("No features selected based on importance threshold")
        
        # Prepare data
        print("Preparing data...")
        X, y = prepare_data_for_prediction("train.csv", important_features)
        
        print(f"Data shape: {X.shape}")
        print(f"Feature types:\n{X.dtypes}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        print("Training model...")
        model = train_optimized_model(X_train, y_train)
        
        # Evaluate model
        train_cindex = evaluate_model(model, X_train, y_train)
        test_cindex = evaluate_model(model, X_test, y_test)
        print(f"Training C-index: {train_cindex:.3f}")
        print(f"Testing C-index: {test_cindex:.3f}")
        
        # Plot survival curves
        print("Plotting survival curves...")
        plot_survival_curves(model, X_test)
        
        print("Done!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        raise
    finally:
        gc.collect()

if __name__ == "__main__":
    main()
'''
def load_and_preprocess_data(file_path, chunk_size=5000):
    # Define dtypes for memory efficiency
    dtype_dict = {
        'efs': 'float32',
        'age_at_hct': 'float32',
        'graft_type': str,
        'prod_type': str,
        'hla_match_drb1_high': str,
        'hla_high_res_10': str,
        'hla_high_res_8': str
    }
    
    selected_features = list(dtype_dict.keys())
    
    # Read data in one go since we're handling types explicitly
    train_data = pd.read_csv(file_path, 
                           usecols=selected_features,
                           dtype=dtype_dict)
    
    return train_data

def preprocess_data(train_data):
    # Identify numerical and categorical columns
    numeric_columns = ['efs', 'age_at_hct']
    categorical_columns = [col for col in train_data.columns if col not in numeric_columns]
    
    # Handle missing values for numeric columns
    num_imputer = SimpleImputer(strategy='median')
    train_data[numeric_columns] = num_imputer.fit_transform(train_data[numeric_columns])
    
    # Handle missing values for categorical columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    train_data[categorical_columns] = cat_imputer.fit_transform(train_data[categorical_columns])
    
    # Encode categorical variables
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_data[categorical_columns] = encoder.fit_transform(train_data[categorical_columns])
    
    # Convert all features to float32 after encoding
    for col in train_data.columns:
        train_data[col] = train_data[col].astype('float32')
    
    return train_data

def batch_predict(model, X, batch_size=1000):
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype='float32')
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = X[i:end_idx]
        predictions[i:end_idx] = model.predict(batch)
        gc.collect()
    
    return predictions

def compute_feature_importance_batched(X, y, feature_names, batch_size=1000):
    feature_importance = {}
    base_rsf = RandomSurvivalForest(
        n_estimators=10,
        min_samples_split=20,
        min_samples_leaf=50,
        max_features="sqrt",
        random_state=42,
        n_jobs=1
    )
    
    for feature in feature_names:
        print(f"Processing feature: {feature}")
        rsf = clone(base_rsf)
        X_subset = X[[feature]].astype('float32')
        
        # Fit model
        rsf.fit(X_subset, y)
        
        # Get predictions in batches
        predictions = batch_predict(rsf, X_subset, batch_size)
        
        # Calculate concordance index
        c_index = concordance_index_censored(y["event"], y["time"], predictions)[0]
        feature_importance[feature] = c_index
        
        # Clean up
        del X_subset, rsf, predictions
        gc.collect()
    
    return pd.Series(feature_importance)

def create_survival_target(data):
    return np.array(
        [(bool(event), float(time)) for event, time in zip(data["efs"], data["age_at_hct"])],
        dtype=[("event", "bool"), ("time", "float32")]
    )

def plot_importance(importance_scores):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importance_scores.index, y=importance_scores.values, palette="magma")
    plt.xticks(rotation=45)
    plt.ylabel("Concordance Index Score")
    plt.title("Feature Importance (Random Survival Forest - Concordance Index)")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    try:
        print("Loading data...")
        train_data = load_and_preprocess_data("train.csv")
        
        print("Preprocessing data...")
        processed_data = preprocess_data(train_data)
        
        print("Preparing features and target...")
        X = processed_data.drop(columns=["efs"])
        y = create_survival_target(processed_data)
        
        # Clean up memory
        del processed_data, train_data
        gc.collect()
        
        print("Computing feature importance...")
        feature_importance = compute_feature_importance_batched(X, y, X.columns, batch_size=1000)
        
        print("Plotting results...")
        plot_importance(feature_importance)
        
        # Save results
        feature_importance.to_csv('feature_importance.csv')
        
        print("Done!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()


### lagacy code 
actural code run on kaggle :
!pip install scikit-survival
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sksurv.metrics import concordance_index_censored
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets
train_path = "/kaggle/input/survive/train.csv"
test_path = "/kaggle/input/survive/test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Select important features
selected_features = ["efs", "age_at_hct", "graft_type", "prod_type", "hla_match_drb1_high", "hla_high_res_10", "hla_high_res_8"]
train_data = train_data[selected_features]

# Handle missing values
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

numerical_features = train_data.select_dtypes(include=['number']).columns.tolist()
categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()

train_data[numerical_features] = num_imputer.fit_transform(train_data[numerical_features])
train_data[categorical_features] = cat_imputer.fit_transform(train_data[categorical_features])

# Encode categorical features
ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_data[categorical_features] = ordinal_encoder.fit_transform(train_data[categorical_features].astype(str))

# Prepare survival dataset
X = train_data.drop(columns=["efs"])
y = np.array([(bool(event), time) for event, time in zip(train_data["efs"], train_data["age_at_hct"])],
             dtype=[("event", "bool"), ("time", "float32")])

# Train Random Survival Forest
rsf = RandomSurvivalForest(
    n_estimators=100,      # More trees since Kaggle has more RAM
    min_samples_split=20,  
    min_samples_leaf=50,   
    max_features="sqrt",  
    random_state=42,
    n_jobs=-1              # Use all available CPU cores
)
rsf.fit(X, y)

# Compute feature importance using Concordance Index
feature_importance = {}

for feature in X.columns:
    X_subset = X[[feature]]
    rsf.fit(X_subset, y)
    predictions = rsf.predict(X_subset)
    c_index = concordance_index_censored(y["event"], y["time"], predictions)[0]
    feature_importance[feature] = c_index

feature_importance = pd.Series(feature_importance).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance.index, y=feature_importance.values, palette="magma")
plt.xticks(rotation=45)
plt.ylabel("Concordance Index Score")
plt.title("Feature Importance (Random Survival Forest - Concordance Index)")
plt.show()


# Apply same preprocessing to test data
test_data[numerical_features] = num_imputer.transform(test_data[numerical_features])
test_data[categorical_features] = cat_imputer.transform(test_data[categorical_features])
test_data[categorical_features] = ordinal_encoder.transform(test_data[categorical_features].astype(str))

# Select the same features as training data
X_test = test_data[selected_features]

# Make Predictions
test_data["prediction"] = rsf.predict(X_test)


# Ensure test_data has an "ID" column
if "ID" not in test_data.columns:
    test_data["ID"] = range(len(test_data))

# Save predictions
submission = test_data[["ID", "prediction"]]
submission.to_csv("/kaggle/working/survival_predictions.csv", index=False)

print("Predictions saved: survival_predictions.csv")


'''