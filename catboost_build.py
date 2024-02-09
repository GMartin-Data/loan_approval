import time

from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
from rich import print
from scipy.stats import randint as sp_randint
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Meant to encode time data with cycles (days of week, month...)"""
    def __init__(self, column_name, cycle_length):
        self.column_name = column_name
        self.cycle_length = cycle_length

    def fit(self, X, y=None):
        # No fitting needed, implemented for compatibility with sklearn's API
        return self

    def transform(self, X, y=None):
        # Apply cyclical encoding directly without needing to fit
        X = X.copy()
        values = X[self.column_name]
        # Create the cyclical features
        X[f'{self.column_name}_sin'] = np.sin(2 * np.pi * values / self.cycle_length)
        X[f'{self.column_name}_cos'] = np.cos(2 * np.pi * values / self.cycle_length)
        # Drop the original column
        X.drop(columns=[self.column_name], inplace=True)
        return X

    def get_feature_names_out(self, input_features=None):
        # Generate names for the output features
        return np.array(
          [f'{self.column_name}_sin', f'{self.column_name}_cos'], dtype=object
        )

# LOAD DATASET AND DROP USELESS COLUMNS
df = (pd.read_csv("data/cleaned_dataset.csv", index_col=0)
      .drop(columns=['LoanNr_ChkDgt', 'Name'])
)

# SEPARATE FEATURES AND TARGET
X = df.copy()
y = X.pop("MIS_Status")

# HOLD-OUT - Stratify with y by default
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.05,
                                                    stratify=y,
                                                    random_state=42)

# PREPROCESSING
# Splitting columns
num_cols = ["Term", "NoEmp", "CreateJob", "RetainedJob",
            "GrAppv", "SBA_Appv"]

cyc_cols = ["ApprovalMonth", "ApprovalDoW"]

nom_cols = ["Bank", "BankState", "City", "Franchise", "LowDoc", "NAICS",
            "NewExist", "Recession", "RevLineCr", "SameState", "State", "UrbanRural"]

# Instanciating Transformers
std_scl = StandardScaler()
cyc_dow = CyclicalEncoder("ApprovalDoW", 7)
cyc_mth = CyclicalEncoder("ApprovalMonth", 12)

# Assembling Column Transformer
preproc = ColumnTransformer(
    transformers = [
        ("num", std_scl, num_cols),
        ("cyc_mth", cyc_mth, ["ApprovalMonth"]),
        ("cyc_dow", cyc_dow, ["ApprovalDoW"]),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
)
# preproc.set_output(transform="pandas")  # Debug

# PREPROCESSED OUTPUT
X_train_tr = preproc.fit_transform(X_train)
nom_indexes = [idx for idx, col in enumerate(X_train_tr.dtypes)
               if col == "object"]

# ADDING ESTIMATOR
train_pool = Pool(data=X_train_tr,
                  label=y_train, 
                  cat_features=nom_indexes,
                  feature_names=X_train_tr.columns.to_list())

cb = CatBoostClassifier(cat_features=nom_cols)

model = make_pipeline(preproc, cb)

# RAW TRAINING AND SCORE
tr_start = time.time()
model.fit(X_train, y_train)
tr_end = time.time()
tr_duration = tr_end - tr_start
print("Raw Train Duration: {tr_duration}")
y_pred = model.predict(X_test)
f1_raw = f1_score(y_test, y_pred, average="macro")
print(f"F1 macro on raw model: {f1_raw:.4f}")

# RANDOMIZED SEARCH
param_distributions = {
    'iterations': [100, 200, 400, 800],
    'depth': [1, 2, 4],
    'learning_rate': loguniform(0.01, 1.0),
    'random_strength': loguniform(1e-9, 10),
    'l2_leaf_reg': [2, 4, 8, 16],
    'bagging_temperature': uniform(0, 1),
    'border_count': sp_randint(1, 255)
}

train_pool = Pool(data=X_train_tr,
                  label=y_train, 
                  cat_features=nom_indexes,
                  feature_names=X_train_tr.columns.to_list())

cb = CatBoostClassifier(cat_features=nom_cols,
                        eval_metric="TotalF1")

rs_start = time.time()
search_results = cb.randomized_search(
    param_distributions,
    X=X_train_tr,
    y=y_train,
    cv=5,
    n_iter=20,
    partition_random_seed=42,
    verbose=True
)
rs_end = time.time()
rs_duration = rs_end - rs_start
print("RandomizedSearch Train Duration: {tr_duration}")




