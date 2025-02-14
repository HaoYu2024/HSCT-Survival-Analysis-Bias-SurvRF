from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Select only the important features to reduce memory usage
selected_features = ["efs", "age_at_hct", "graft_type", "prod_type", 
                     "hla_match_drb1_high", "hla_high_res_10", "hla_high_res_8"]

# Load the training dataset
train_data = pd.read_csv("train.csv")

# Keep only the selected features
train_data = train_data[selected_features]

# Handle missing values using median for numerical and mode for categorical
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Identify numerical and categorical features
numerical_features = train_data.select_dtypes(include=['number']).columns.tolist()
categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()

# Impute missing values
train_data[numerical_features] = num_imputer.fit_transform(train_data[numerical_features])
train_data[categorical_features] = cat_imputer.fit_transform(train_data[categorical_features])

# Encode categorical features using Ordinal Encoding
ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_data[categorical_features] = ordinal_encoder.fit_transform(train_data[categorical_features].astype(str))

# Prepare survival dataset
X = train_data.drop(columns=["efs"])
y = np.array(
    [(bool(event), time) for event, time in zip(train_data["efs"], train_data["age_at_hct"])],
    dtype=[("event", "bool"), ("time", "float32")]
)


# Train Random Survival Forest with selected features
rsf = RandomSurvivalForest(
    n_estimators=20,         # Reduce number of trees
    min_samples_split=20,    # Reduce depth of trees
    min_samples_leaf=50,     # Prevent overfitting
    max_features="sqrt",     # Limit number of features per tree
    random_state=42,
    n_jobs=-1                # Use multiple CPU cores
)

# Fit RSF
rsf.fit(X, y)
from sksurv.metrics import concordance_index_censored

# Compute feature importance based on C-index
feature_importance = {}

for feature in X.columns:
    X_subset = X[[feature]]  # Use one feature at a time
    rsf.fit(X_subset, y)     # Refit model
    predictions = rsf.predict(X_subset)  # Get risk scores

    # Compute Concordance Index for this feature
    c_index = concordance_index_censored(y["event"], y["time"], predictions)[0]
    feature_importance[feature] = c_index

# Convert to pandas Series and sort
feature_importance = pd.Series(feature_importance).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance.index, y=feature_importance.values, palette="magma")
plt.xticks(rotation=45)
plt.ylabel("Concordance Index Score")
plt.title("Feature Importance (Random Survival Forest - Concordance Index)")
plt.show()
'''
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