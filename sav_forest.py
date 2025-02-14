import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load dataset
train_data = pd.read_csv("train.csv")

### --- Random Forest Feature Importance (Ignoring Censoring) --- ###

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
y = encoded_data["efs"]  # Binary classification target

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest Classifier (ignoring censoring for now)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Extract feature importance
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(14, 8))
sns.barplot(x=feature_importance.index[:20], y=feature_importance.values[:20], palette="magma")
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.3)  
plt.ylabel("Feature Importance Score")
plt.title("Top 20 Feature Importance (Random Forest - Ignoring Censoring)")
plt.show()

### --- Detect Censoring Using Probability Distributions --- ###

# Plot distribution of efs_time
plt.figure(figsize=(10, 6))
sns.histplot(train_data["efs_time"], bins=50, kde=True, color="royalblue", label="efs_time")
plt.xlabel("Event-Free Survival Time (efs_time)")
plt.ylabel("Frequency")
plt.title("Distribution of Event-Free Survival Time")
plt.legend()
plt.show()

# Compute empirical cumulative distribution function (ECDF)
sorted_times = np.sort(train_data["efs_time"])
ecdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)

plt.figure(figsize=(10, 6))
plt.plot(sorted_times, ecdf, marker=".", linestyle="none", color="darkred")
plt.xlabel("Event-Free Survival Time (efs_time)")
plt.ylabel("ECDF")
plt.title("Empirical CDF of Event-Free Survival Time")
plt.show()

# Check probability distributions for censoring
observed_events = train_data["efs"] == 1  # Patients with event (not censored)
censored_events = train_data["efs"] == 0  # Patients censored

plt.figure(figsize=(10, 6))
sns.kdeplot(train_data.loc[observed_events, "efs_time"], label="Observed Events", color="red")
sns.kdeplot(train_data.loc[censored_events, "efs_time"], label="Censored Events", color="blue", linestyle="dashed")
plt.xlabel("Event-Free Survival Time (efs_time)")
plt.ylabel("Density")
plt.title("Kernel Density Estimate of Observed vs. Censored Data")
plt.legend()
plt.show()

# Perform statistical test: Are censored and observed event distributions different?
ks_stat, p_value = stats.ks_2samp(train_data.loc[observed_events, "efs_time"],
                                  train_data.loc[censored_events, "efs_time"])

print(f"Kolmogorov-Smirnov Test Statistic: {ks_stat:.4f}, p-value: {p_value:.4e}")

#Kolmogorov-Smirnov Test Statistic: 0.9520, p-value: 0.0000e+00


'''
### **Kolmogorov-Smirnov (KS) Test Statistic Explained**
The **Kolmogorov-Smirnov (KS) test** is a non-parametric test used to compare a sample distribution with a reference distribution (one-sample KS test) or to compare two independent samples (two-sample KS test). It measures how different two cumulative distribution functions (CDFs) are.

The **test statistic (D)** represents the maximum absolute difference between the empirical cumulative distribution function (ECDF) of the sample and the CDF of the reference distribution.

Mathematically, for a one-sample KS test:
\[
D = \sup_x | F_n(x) - F(x) |
\]
where:
- \( F_n(x) \) is the ECDF of the sample,
- \( F(x) \) is the CDF of the reference distribution,
- \( \sup \) denotes the supremum (the maximum deviation at any point).

For a two-sample KS test:
\[
D = \sup_x | F_1(x) - F_2(x) |
\]
where \( F_1(x) \) and \( F_2(x) \) are the empirical CDFs of two different samples.

### **Interpreting the Results**
Your reported values:

- **KS Test Statistic (D) = 0.9520**  
  - This is a very high test statistic, meaning the maximum difference between the two distributions is 0.952 (close to 1).  
  - A value close to 1 suggests that the sample distribution is **very different** from the reference distribution.
  - A value close to 0 would mean the distributions are almost identical.

- **p-value = 0.0000 (practically 0)**  
  - The p-value tells us the probability of observing such an extreme \( D \) value under the null hypothesis.
  - A p-value of essentially **zero** means there is extremely strong evidence to reject the null hypothesis.

### **Implications**
1. **If this is a one-sample KS test:**  
   - The null hypothesis (\(H_0\)) assumes that the sample follows the given theoretical distribution.
   - Since the p-value is 0, we **reject \(H_0\)**, meaning the sample does **not** follow the reference distribution.
   - The sample comes from a **significantly different distribution**.

2. **If this is a two-sample KS test:**  
   - The null hypothesis (\(H_0\)) assumes that both samples come from the same underlying distribution.
   - With \( p \approx 0 \), we **reject \(H_0\)**, meaning the two distributions are **significantly different**.

### **Possible Causes of High KS Statistic**
- The sample may have a **different mean, variance, skewness, or shape** compared to the reference distribution.
- There could be **outliers** or **heavy-tailed behavior** in the sample that the reference distribution does not capture.
- The reference distribution may not be appropriate for modeling the sample.

### **Next Steps**
- **If you are testing against a normal distribution:** Consider using a **QQ plot** to visually inspect deviations.
- **If comparing two samples:** Examine histograms or kernel density plots to see how they differ.
- **If rejecting normality in a goodness-of-fit test:** Try alternative distributions (e.g., log-normal, exponential, etc.).
- **If performing a two-sample test:** Consider the **Mann-Whitney U test** or **Chi-Square test** for further insights.

Would you like me to generate some plots to help visualize the distributions?

'''
