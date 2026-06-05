import pandas as pd

def detect_task_type(y, threshold=5):
    if pd.api.types.is_numeric_dtype(y):
        return "Regression" if y.nunique() > threshold else "Classification"
    return "Classification"

def preprocess_target(y):
    task_type = detect_task_type(y)
    if task_type == "Classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return y_encoded, task_type, le.classes_
    else:
        y_encoded = pd.to_numeric(y, errors="coerce")
        return y_encoded, task_type, None