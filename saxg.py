# Re-import necessary libraries after execution state reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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

xgb_model = xgb.XGBRegressor(
    objective="survival:cox",
    n_estimators=200,      # Kept low based on your observation
    learning_rate=1.5,     # Increased since we have fewer trees
    max_depth=2,           # Reduced to focus on strongest signals
    min_child_weight=2,    # Reduced to allow more splits given shallow depth
    subsample=0.8,         # Keep high for stability with fewer trees
    colsample_bytree=0.5,  # Reduced to focus on strongest features per tree
    reg_alpha=0.5,         # Light regularization since we have fewer trees
    reg_lambda=0.5,        # Light regularization
    random_state=42,
    tree_method="hist"
)

'''
c:0.59
xgb_model = xgb.XGBRegressor(
    objective="survival:cox",
    n_estimators=1000,  # Increased for complex relationships
    learning_rate=0.01,  # Reduced to prevent overfitting
    max_depth=3,        # Reduced to prevent overfitting
    min_child_weight=3, # Added to handle noise
    subsample=0.7,      # Reduced to prevent overfitting
    colsample_bytree=0.6, # Reduced due to low feature correlation
    reg_alpha=0.1,      # L1 regularization added
    reg_lambda=1.0,     # L2 regularization added
    random_state=42,
    tree_method="hist"
)
'''

# Train the model
xgb_model.fit(X_train, y_train,eval_set=[(X_val, y_val)], verbose=True)

# Feature Importance Analysis
feature_importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance.index[:20], y=feature_importance.values[:20], palette="magma")
plt.xticks(rotation=90)
plt.ylabel("Feature Importance Score")
plt.title("Top 20 Feature Importance (XGBoost - Cox Survival Model)")
plt.show()




# Load test dataset
test_file_path = "test.csv"
test_data = pd.read_csv(test_file_path)

# Encode categorical variables in the test set using the same approach
for col in categorical_features:
    if col in test_data.columns:
        test_data[col] = LabelEncoder().fit_transform(test_data[col].astype(str))

# Handle missing values using the same strategy
test_data[numerical_features] = num_imputer.transform(test_data[numerical_features])
test_data[categorical_features] = cat_imputer.transform(test_data[categorical_features])

# Extract features for prediction
X_test = test_data.drop(columns=["ID"])

# Predict survival risk scores on test set
test_predictions = xgb_model.predict(X_test)

# Prepare submission file
submission = pd.DataFrame({"ID": test_data["ID"], "prediction": test_predictions})

# Save to CSV
submission_file_path = "xgb_predictions.csv"
submission.to_csv(submission_file_path, index=False)


from sklearn.metrics import r2_score

# Predict on validation set
y_val_pred = xgb_model.predict(X_val)

# Calculate R² score
r2 = r2_score(y_val, y_val_pred)

# Display R² score
print(f"R² Score on Validation Set: {r2:.4f}")


from lifelines.utils import concordance_index

# Calculate C-index on validation set
def calculate_c_index(y_true, y_pred, durations):
    """
    Calculate concordance index for survival predictions
    
    Parameters:
    y_true: True event indicators (0 or 1)
    y_pred: Predicted risk scores
    durations: Observed times
    
    Returns:
    float: concordance index
    """
    try:
        c_index = concordance_index(durations, -y_pred, y_true)
        return c_index
    except Exception as e:
        print(f"Error calculating C-index: {e}")
        return None

# Get the true event indicators from the validation set
y_val_events = encoded_data.loc[y_val.index, "efs"]

# Calculate C-index
c_index = calculate_c_index(y_val_events, y_val_pred, y_val)

print(f"\nConcordance Index (C-index) on Validation Set: {c_index:.4f}")

# Visualize C-index alongside R² score
plt.figure(figsize=(10, 5))
metrics = {'R² Score': r2, 'C-index': c_index}
colors = ['#2ecc71', '#3498db']

plt.bar(metrics.keys(), metrics.values(), color=colors)
plt.ylim(0, 1)
plt.title('Model Performance Metrics')
plt.ylabel('Score')

# Add value labels on top of each bar
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')


###
