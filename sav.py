
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

from lifelines import WeibullFitter, LogLogisticFitter, LogNormalFitter, KaplanMeierFitter
from lifelines.utils import median_survival_times

# Load the training dataset
train_file_path = "train.csv"
train_data = pd.read_csv(train_file_path)

# Display the first few rows to understand its structure
train_data.head()

'''
# Check missing values
missing_values = train_data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)


# Visualize missing values
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values, palette="viridis")
plt.xticks(rotation=90)
plt.ylabel("Count of Missing Values")
plt.title("Missing Values Per Feature")
plt.show()

# Convert efs_time to numerical format
train_data["efs_time"] = pd.to_numeric(train_data["efs_time"], errors="coerce")

# Plot Kaplan-Meier Survival Curve
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))
kmf.fit(train_data["efs_time"], event_observed=train_data["efs"])
kmf.plot_survival_function()
plt.title("Kaplan-Meier Survival Curve")
plt.xlabel("Time to Event-Free Survival (efs_time)")
plt.ylabel("Survival Probability")
plt.show()

# Encode categorical variables numerically for PCA and t-SNE visualization
encoded_data = train_data.copy()
for col in encoded_data.select_dtypes(include=["object"]).columns:
    encoded_data[col], _ = pd.factorize(encoded_data[col])

# Normalize numerical features
numerical_features = encoded_data.select_dtypes(include=[np.number]).columns
encoded_data[numerical_features] = (encoded_data[numerical_features] - encoded_data[numerical_features].mean()) / encoded_data[numerical_features].std()

# PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(encoded_data[numerical_features].fillna(0))
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=train_data["efs"], cmap="coolwarm", alpha=0.6)
plt.colorbar(label="Event-Free Survival (efs)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Visualization of Feature Space")
plt.show()

# t-SNE for clustering
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(encoded_data[numerical_features].fillna(0))
plt.figure(figsize=(10, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=train_data["efs"], cmap="coolwarm", alpha=0.6)
plt.colorbar(label="Event-Free Survival (efs)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Feature Space")
plt.show()



# Fit Weibull Model
weibull_fitter = WeibullFitter()
weibull_fitter.fit(train_data["efs_time"], event_observed=train_data["efs"])

# Fit Log-Logistic Model
loglogistic_fitter = LogLogisticFitter()
loglogistic_fitter.fit(train_data["efs_time"], event_observed=train_data["efs"])

# Fit Log-Normal Model
lognormal_fitter = LogNormalFitter()
lognormal_fitter.fit(train_data["efs_time"], event_observed=train_data["efs"])

# Plot survival functions
plt.figure(figsize=(10, 6))
weibull_fitter.plot_survival_function(label="Weibull")
loglogistic_fitter.plot_survival_function(label="Log-Logistic")
lognormal_fitter.plot_survival_function(label="Log-Normal")
plt.title("Comparison of Parametric Survival Models")
plt.xlabel("Time to Event-Free Survival (efs_time)")
plt.ylabel("Survival Probability")
plt.legend()
plt.show()

# Kaplan-Meier survival curves for group-level trends (race_group)
kmf = KaplanMeierFitter()
plt.figure(figsize=(12, 6))

for race in train_data["race_group"].dropna().unique():
    race_subset = train_data[train_data["race_group"] == race]
    kmf.fit(race_subset["efs_time"], event_observed=race_subset["efs"], label=str(race))
    kmf.plot_survival_function()

plt.title("Kaplan-Meier Survival Curves by Race Group")
plt.xlabel("Time to Event-Free Survival (efs_time)")
plt.ylabel("Survival Probability")
plt.legend(title="Race Group")
plt.show()

'''


from sklearn.preprocessing import LabelEncoder

# Select early failures (patients with events within 20 months)
early_failure_data = train_data[train_data["efs_time"] <= 20].copy()

# Identify categorical columns
categorical_features = early_failure_data.select_dtypes(include=['object']).columns

# Encode categorical features using Label Encoding
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    early_failure_data[col] = le.fit_transform(early_failure_data[col].astype(str))
    label_encoders[col] = le  # Store encoders if needed later

# Drop any remaining non-numeric columns (if necessary)
early_failure_data = early_failure_data.select_dtypes(include=["number"])

# Compute feature correlations with early failures (efs)
feature_correlations = early_failure_data.corr()["efs"].sort_values(ascending=False)

# Display top 10 correlated features
print(feature_correlations.head(10))
'''
### **Analysis of Early Failure Correlations**  

The **top correlated features** with early failures (`efs = 1`) provide valuable insights into risk factors affecting survival outcomes in the first **20 months**. Here’s a breakdown of what we can learn:

---

### **🚀 Key Takeaways from Correlation Results**  

#### **1️⃣ Age at Transplant (`age_at_hct` → 0.1438)**
- **Higher age at transplant is correlated with early failure.**  
- This makes sense since **older patients** typically have more comorbidities, slower recovery, and increased risks post-transplant.
- **Next Step:** Consider splitting the population into **age groups** and analyzing survival curves separately.

#### **2️⃣ Graft Type (`graft_type` → 0.1432) & Product Type (`prod_type` → 0.1426)**
- This suggests that **certain stem cell sources (bone marrow, peripheral blood, cord blood) may have a significant impact on early survival.**  
- Peripheral blood stem cells (PBSC) vs. bone marrow vs. cord blood may have different immune rejection risks.
- **Next Step:** Check **which specific types of grafts and products** are driving this correlation.

#### **3️⃣ HLA Matching (Several HLA-related features)**
- **Multiple HLA-matching features (DRB1, NMDP, high-res 10/8/6) show moderate correlation (0.07–0.09).**  
- This confirms that **better donor-recipient matching is protective**, while mismatches might be increasing early transplant failure risks.
- However, the correlation strength is **lower than age and graft type**, suggesting other factors (like conditioning regimen and comorbidities) may play a larger role.

#### **4️⃣ Duplicate Feature Issue?**
- `hla_match_drb1_low` appears **twice** in the correlation output, meaning there might be **redundant or highly related features** in the dataset.  
- **Next Step:** Perform **feature selection** or **principal component analysis (PCA)** to remove multicollinearity.

---

### **🔬 What This Means for Modeling**
✔ **Age, graft type, and product type should be prioritized in feature selection.**  
✔ **HLA-matching features can be used, but careful selection is needed to avoid redundancy.**  
✔ **Stratifying by patient age and graft type might improve model performance.**  

---

### **🛠 Next Steps & Suggested Visualizations**
Would you like me to:
1. **Visualize Kaplan-Meier curves** stratified by **age groups & graft type**?
2. **Check survival distributions across different HLA match levels**?
3. **Perform PCA to remove redundant HLA features** and identify the most important ones?


'''