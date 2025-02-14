import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import xgboost
import lightgbm as lgb
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index

print("Using XGBoost version", xgboost.__version__)
print("Using LightGBM version", lgb.__version__)

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("test.csv")
print("Test shape:", test.shape)

train = pd.read_csv("train.csv")
print("Train shape:", train.shape)
train.head()

plt.hist(train.loc[train.efs==1,"efs_time"], bins=100, label="efs=1, Did Not Survive")
plt.hist(train.loc[train.efs==0,"efs_time"], bins=100, label="efs=0, Maybe Survived")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to death, or time observed alive.")
plt.legend()
plt.show()

# Transform targets
train["y"] = train.efs_time.values
mx = train.loc[train.efs==1,"efs_time"].max()
mn = train.loc[train.efs==0,"efs_time"].min()
train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
train.y = train.y.rank()
train.loc[train.efs==0,"y"] += len(train)//2
train.y = train.y / train.y.max()

plt.hist(train.loc[train.efs==1,"y"], bins=100, label="efs=1, Did Not Survive")
plt.hist(train.loc[train.efs==0,"y"], bins=100, label="efs=0, Maybe Survived")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()

RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")

# Handle categorical features
CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"There are {len(CATS)} CATEGORICAL FEATURES: {CATS}")

# Training with k-fold
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    # Convert categorical columns for XGBoost
    for cat in CATS:
        x_train[cat] = x_train[cat].astype('category')
        x_valid[cat] = x_valid[cat].astype('category')
        x_test[cat] = x_test[cat].astype('category')
    
    # XGBoost
    print("Training XGBoost...")
    model_xgb = XGBRegressor(
        device="cpu",
        max_depth=3,  
        colsample_bytree=0.5, 
        subsample=0.8, 
        n_estimators=10_000,  
        learning_rate=0.1, 
        eval_metric="mae",
        objective='reg:logistic',
        enable_categorical=True,
        min_child_weight=5
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=100 
    )
    
    # LightGBM
    print("Training LightGBM...")
    train_data_lgb = lgb.Dataset(x_train, label=y_train, categorical_feature=CATS)
    valid_data_lgb = lgb.Dataset(x_valid, label=y_valid, categorical_feature=CATS)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'device': 'cpu',
        'num_threads': -1  # Use all available CPU cores
    }
    
    model_lgb = lgb.train(
        params,
        train_data_lgb,
        valid_sets=[valid_data_lgb],
        num_boost_round=10000,
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    oof_lgb[test_index] = model_lgb.predict(x_valid)
    
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)
    pred_lgb += model_lgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS
pred_lgb /= FOLDS

# Calculate R2 scores
train_preds_xgb = -oof_xgb
train_preds_lgb = -oof_lgb
train_preds_ensemble = -(oof_xgb + oof_lgb)/2
train_actual = train['y']

r2_xgb = r2_score(train_actual, train_preds_xgb)
r2_lgb = r2_score(train_actual, train_preds_lgb)
r2_ensemble = r2_score(train_actual, train_preds_ensemble)

print(f"\nTraining R2 Scores:")
print(f"XGBoost: {r2_xgb:.4f}")
print(f"LightGBM: {r2_lgb:.4f}")
print(f"Ensemble: {r2_ensemble:.4f}")

# Compute concordance index (C-index) scores
print("\nConcordance Index Scores:")
# XGBoost score
c_index_xgb = concordance_index(train['efs_time'], oof_xgb, 1-train['efs'])  # Note: flipped efs and removed negative
print(f"XGBoost C-index = {c_index_xgb:.4f}")

# LightGBM score
c_index_lgb = concordance_index(train['efs_time'], oof_lgb, 1-train['efs'])
print(f"LightGBM C-index = {c_index_lgb:.4f}")

# Ensemble score
c_index_ensemble = concordance_index(train['efs_time'], (oof_xgb + oof_lgb)/2, 1-train['efs'])
print(f"Ensemble C-index = {c_index_ensemble:.4f}")

# Create submission
try:
    sub = pd.read_csv("sample_submission.csv")
except FileNotFoundError:
    sub = test[["ID"]].copy()
    sub["prediction"] = 0

sub.prediction = -(pred_xgb + pred_lgb)/2  # Average of both models
sub.to_csv("submission.csv",index=False)
print("\nSub shape:",sub.shape)
sub.head()



'''
The **Concordance Index (C-index)** is a measure of the predictive accuracy of a risk model, commonly used in survival analysis to evaluate the performance of models like the **Cox proportional hazards model**.

### **Definition**
The C-index quantifies how well a model ranks survival times. It is essentially the probability that, for a randomly selected pair of observations, the model correctly predicts which individual will experience an event (e.g., death, failure, relapse) first.

### **Formula**
The C-index is calculated as:

\[
C = \frac{\sum I(\hat{t}_i > \hat{t}_j)}{\text{total number of comparable pairs}}
\]

where:
- \(\hat{t}_i, \hat{t}_j\) are the predicted risk scores (or survival times) for individuals \(i\) and \(j\),
- \(I(\hat{t}_i > \hat{t}_j)\) is an indicator function that equals 1 if the predicted ordering matches the actual ordering, and 0 otherwise.

### **Interpretation**
- \( C = 0.5 \): The model performs no better than random chance.
- \( C = 1.0 \): The model perfectly ranks survival times.
- \( C < 0.5 \): The model is worse than random (misclassification).

### **Use Cases**
- **Survival Analysis:** Used to evaluate the performance of Cox models and other survival models.
- **Credit Scoring:** Helps assess the accuracy of risk prediction models.
- **Medical Research:** Determines the effectiveness of prognostic models in predicting patient survival.

Would you like a Python implementation example?

'''








'''
#https://www.kaggle.com/code/cdeotte/xgboost-catboost-baseline-cv-668-lb-668/notebook
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)

from catboost import CatBoostRegressor, CatBoostClassifier
import catboost
print("Using CatBoost version",catboost.__version__)

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("train.csv")
print("Train shape:",train.shape)
train.head()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Did Not Survive")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Survived")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to death, or time observed alive.")
plt.legend()
plt.show()


#Both targets efs and efs_time provide useful information. We will tranform these two targets into a single target to train our model with.
train["y"] = train.efs_time.values
mx = train.loc[train.efs==1,"efs_time"].max()
mn = train.loc[train.efs==0,"efs_time"].min()
train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
train.y = train.y.rank()
train.loc[train.efs==0,"y"] += len(train)//2
train.y = train.y / train.y.max()

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Did Not Survive")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Survived")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()

RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")

CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.5, 
        subsample=0.8, 
        n_estimators=10_000,  
        learning_rate=0.1, 
        eval_metric="mae",
        early_stopping_rounds=25,
        objective='reg:logistic',
        enable_categorical=True,
        min_child_weight=5
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=100 
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = -oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost
print("Using CatBoost version",catboost.__version__)


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_cat = CatBoostRegressor(
        task_type="GPU",  
    )
    model_cat.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=CATS,
              verbose=100)

    # INFER OOF
    oof_cat[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = -oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost =",m)




sub = pd.read_csv("sample_submission.csv")
sub.prediction = -pred_xgb -pred_cat
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()
'''
