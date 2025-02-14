#take about 10 minites to run 


from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer




import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


# Load the training dataset
train_file_path = "train.csv"
train_data = pd.read_csv(train_file_path)
# Prepare data for Random Survival Forest Feature Importance Analysis

# Select numerical and categorical features
numerical_features = train_data.select_dtypes(include=['number']).columns.tolist()
categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()

# Encode categorical variables
encoded_data = train_data.copy()
for col in categorical_features:
    encoded_data[col] = LabelEncoder().fit_transform(encoded_data[col].astype(str))

# Handle missing values using median for numerical and mode for categorical
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

encoded_data[numerical_features] = num_imputer.fit_transform(encoded_data[numerical_features])
encoded_data[categorical_features] = cat_imputer.fit_transform(encoded_data[categorical_features])

# Prepare survival dataset
X = encoded_data.drop(columns=["efs", "efs_time", "ID"])
y = np.array([(event, time) for event, time in zip(encoded_data["efs"], encoded_data["efs_time"])],
             dtype=[("event", "bool"), ("time", "float")])

# Train Random Survival Forest
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=42)
rsf.fit(X, y)

# Extract feature importance
feature_importance = pd.Series(rsf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance.index[:20], y=feature_importance.values[:20], palette="magma")
plt.xticks(rotation=90)
plt.ylabel("Feature Importance Score")
plt.title("Top 20 Feature Importance (Random Survival Forest)")
plt.show()


