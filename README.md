Here’s a **markdown summary** of our current goal and analysis so far:

---

# **Survival Analysis for Hematopoietic Stem Cell Transplantation (HSCT)**

## **🔹 Goal**
Our objective is to build an **accurate and fair survival prediction model** for patients undergoing **Hematopoietic Stem Cell Transplantation (HSCT)**.  
We aim to optimize predictions using **Bias-SurvRF** while ensuring **fairness across racial groups** through the **Stratified Concordance Index**.

## **🔹 Data Overview**
- **Dataset:** Synthetic dataset generated via **SurvivalGAN**, trained on real CIBMTR data.
- **Key Variables:**
  - `efs`: Event-Free Survival (target variable).
  - `efs_time`: Time to event (or censoring).
  - **59 demographic & clinical features** including age, donor-recipient matching, disease risk, and transplant type.

## **🔹 Our Analysis & Findings**
### **1️⃣ Data Exploration & Visualization**
- **Kaplan-Meier Survival Curve:**  
  - Most transplant failures occur **within the first 20 months**.
  - Long-term survival stabilizes **beyond 50 months**.
- **PCA & t-SNE Analysis:**  
  - Some clustering patterns exist but **linear separation is weak**, suggesting that **non-linear models** like **Random Survival Forests (RSF)** may be useful.

### **2️⃣ Censoring Analysis**
- **Censored vs. Observed Events:**
  - **Censored patients tend to survive longer** than those with observed failures.
  - The **Kolmogorov-Smirnov (KS) test** confirms that their distributions **differ significantly** (`KS stat = 0.95, p = 0.0`).

### **3️⃣ Feature Importance (Random Forest - Ignoring Censoring)**
- **Top Features:**
  - `age_at_hct` (Patient Age at Transplant)
  - `donor_age` (Donor’s Age)
  - `conditioning_intensity` (Pre-transplant chemotherapy/radiation)
  - `dri_score` (Disease Risk Index)
  - `hla_match_drb1_high` (HLA Matching)

- **Key Takeaways:**
  - **Older patients & certain transplant procedures are more at risk.**
  - **HLA matching plays a role but is not the most dominant predictor.**
  - **Potential redundancy in HLA features (e.g., DRB1 appears twice).**

### **4️⃣ Early Failure Analysis (First 20 Months)**
- **Most correlated features with early failures:**
  - **`age_at_hct` (0.1438)** → Older patients face higher early failure risk.
  - **`graft_type` & `prod_type` (~0.14)** → Certain transplant sources impact survival.
  - **HLA matching (0.07–0.09)** → Lower but still important.
- **Next Steps:**
  - **Stratify survival curves** by age and graft type.
  - **Check redundancy in HLA-related variables.**
  - **Explore potential feature engineering improvements.**

---

## **🔹 Next Steps**
1. **📊 Kaplan-Meier Curves:** Compare **survival trends by age groups & transplant type**.
2. **📈 Parametric Survival Models (Weibull, Log-Logistic, etc.):** Evaluate if they better capture early failures.
3. **🧩 Feature Engineering:** Address redundancy & optimize variable selection.

---

