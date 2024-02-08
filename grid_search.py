import time

import category_encoders as ce
import numpy as np
import pandas as pd
from rich import print
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


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


# Loading and droping useless columns
df = (pd
      .read_csv("data/cleaned_dataset.csv", index_col=0)
      .drop(columns=['LoanNr_ChkDgt', 'Name'])
)

# Separating Feature and Target
X = df.copy()
y = X.pop("MIS_Status")
      
# Hold-Out (stratify with y by default)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.05,
                                                    stratify=y,
                                                    random_state=42)

# Encoding Target
lbl_enc = LabelEncoder()
y_train = lbl_enc.fit_transform(y_train)
y_test = lbl_enc.transform(y_test)

# Preprocessing
# Splitting Columns
num_cols = ["Term", "NoEmp", "CreateJob", "RetainedJob",
            "GrAppv", "SBA_Appv"]
bin_cols = df.select_dtypes("bool").columns
nom_cols = ["UrbanRural", "RevLineCr", "LowDoc"]
cyc_cols = ["ApprovalMonth", "ApprovalDoW"]
tgt_cols = ["State", "BankState", "NAICS", "Bank", "City"]

std_scl = StandardScaler()
ohe_bin = OneHotEncoder(drop="if_binary",
                        sparse_output=True,
                        handle_unknown="ignore")
ohe_nom = OneHotEncoder(sparse_output=True,  # No drop="first" with a nonlinear model
                        handle_unknown="ignore")
cyc_dow = CyclicalEncoder("ApprovalDoW", 7)
cyc_mth = CyclicalEncoder("ApprovalMonth", 12)
tgt_enc = ce.TargetEncoder(cols=tgt_cols)

preproc = ColumnTransformer(
    transformers = [
        ("num", std_scl, num_cols),
        ("bin", ohe_bin, bin_cols),
        ("nom", ohe_nom, nom_cols),
        ("cyc_mth", cyc_mth, ["ApprovalMonth"]),
        ("cyc_dow", cyc_dow, ["ApprovalDoW"]),
        ("tgt_enc", tgt_enc, tgt_cols),
    ],
    verbose_feature_names_out=False
)

# Feature Selection
rfc = RandomForestClassifier(random_state=0)
sfm = SelectFromModel(rfc, threshold=0.01)
model = make_pipeline(preproc, sfm, rfc)

# Training and Score
tr_start = time.time()
model.fit(X_train, y_train)
tr_end = time.time()
tr_duration = tr_end - tr_start
print("Raw Train Duration: {tr_duration}")
y_pred = model.predict(X_test)
f1_raw = f1_score(y_test, y_pred, average="macro")
print(f"F1 macro on raw model: {f1_raw:.4f}")

param_dist = {
    "randomforestclassifier__max_features": ['sqrt', 'log2', None] + list(np.linspace(0.1, 1.0, 10)),
    "randomforestclassifier__min_samples_leaf": sp_randint(1, 20),
    "randomforestclassifier__bootstrap": [True, False],
    "randomforestclassifier__n_estimators": sp_randint(100, 2_000)
}

random_search = RandomizedSearchCV(model,
                                   param_distributions=param_dist,
                                   n_iter=100,
                                   cv=5,
                                   scoring='f1_macro',
                                   verbose=1,
                                   n_jobs=-1,
                                   random_state=42)
rs_start = time.time()
random_search.fit(X_train, y_train)
rs_end = time.time()
rs_duration = rs_end - rs_start
print("RandomizedSearch Train Duration: {tr_duration}")

y_pred = random_search.best_estimator_.predict(X_test)
f1_tuned = f1_score(y_test, y_pred, average="macro")
print(f"F1 macro on tuned model: {f1_tuned:.4f}")
print(random_search.best_params_)
