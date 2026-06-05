import numpy as np
import pandas as pd
from sklearn.datasets import load_wine,load_breast_cancer,load_diabetes,load_iris,fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer



def get_selected_dataframe(source, uploaded_file=None, demo_choice=None):
    if source == "User Dataset" and uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        label = "User Dataset"
    elif source == "Preload Dataset" and demo_choice is not None and demo_choice != "None":
        if demo_choice == "Iris (C)":
            raw_data = load_iris()
        elif demo_choice == "Breast Cancer (C)":
            raw_data = load_breast_cancer()
        elif demo_choice == "Diabetes (R)":
            raw_data = load_diabetes()
        elif demo_choice == "California Housing (R)":
            raw_data = fetch_california_housing()
        else:
            raw_data = load_wine()

        data = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
        data["target"] = raw_data.target
        data = data[[*raw_data.feature_names, "target"]]
        label = demo_choice
    else:
        data = None
        label = None

    return data, label




def preprocess_with_column_names(df, cardinality_threshold=10):
    
    target_col = df.columns[-1]
    X_raw = df.iloc[:, :-1]
    y = df[target_col]


    num_cols = X_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in num_cols.copy():
        if X_raw[col].nunique() <= cardinality_threshold:
            num_cols.remove(col)
            cat_cols.append(col)

    column_types = {
        "numeric": num_cols,
        "categorical": cat_cols,
        "target": target_col
    }

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])


    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X_raw)
    feature_names_out = preprocessor.get_feature_names_out()

    def get_transformed_feature_names(preprocessor):
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
        return feature_names


    return X_processed, y, feature_names_out, column_types, preprocessor