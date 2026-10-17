import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
DATA_PATH = os.path.join(DATA_DIR, "german_credit.csv")


def _coerce_target(y):
    y = y.copy()
    if y.dtype == object or str(y.dtype).startswith("category"):
        normalized = y.astype(str).str.strip().str.lower()
        return normalized.map(lambda v: 1 if v in ("good", "1", "1.0") else 0)
    return y.map(lambda v: 1 if str(v).strip() in ("1", "1.0", "good") else 0)


def _load_dataframe():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        from sklearn.datasets import fetch_openml

        os.makedirs(DATA_DIR, exist_ok=True)
        bunch = fetch_openml(data_id=31, as_frame=True, parser="auto")
        df = bunch.frame
        df.to_csv(DATA_PATH, index=False)
        return df


def load_and_preprocess_data():
    df = _load_dataframe()

    target_col = "class" if "class" in df.columns else df.columns[-1]

    y = _coerce_target(df[target_col])
    X = df.drop(columns=[target_col])

    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
