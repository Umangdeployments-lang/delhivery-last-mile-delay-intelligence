"""Package the final trained delay model and related artifacts."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments


def run_package_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Read config
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    max_depth_val = config["max_depth"]
    max_depth = None if str(max_depth_val).lower() in ["unlimited", "none", "nan"] else int(max_depth_val)
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    # 2. Load modelling table and recreate Part C split
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # 3. Fit the final pipeline on training rows
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=selector_k)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )),
    ])
    final_pipeline.fit(X_train, y_train)

    # 4. Save artifacts in models/
    # 4a. Model pickle
    joblib.dump(final_pipeline, "models/delay_model.pkl")
    print("Saved models/delay_model.pkl")

    # 4b. Threshold CSV
    th_df = pd.DataFrame([{"threshold": threshold}])
    th_df.to_csv("models/delay_threshold.csv", index=False)
    print("Saved models/delay_threshold.csv")

    # 4c. Feature order CSV
    feat_df = pd.DataFrame({"feature": FEATURES})
    feat_df.to_csv("models/feature_order.csv", index=False)
    print("Saved models/feature_order.csv")

    # 4d. Impute values CSV from raw shipments
    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)
    impute_df = pd.DataFrame([
        {"column": "distance_km", "median": round(float(medians["distance_km"]), 4)},
        {"column": "parcel_weight_kg", "median": round(float(medians["parcel_weight_kg"]), 4)},
        {"column": "declared_value_inr", "median": round(float(medians["declared_value_inr"]), 4)},
    ])
    impute_df.to_csv("models/impute_values.csv", index=False)
    print("Saved models/impute_values.csv")

    # 5. Reload check
    reloaded_pipeline = joblib.load("models/delay_model.pkl")
    probs_in_memory = final_pipeline.predict_proba(X_test)[:, 1]
    probs_reloaded = reloaded_pipeline.predict_proba(X_test)[:, 1]

    diff = float(np.abs(probs_in_memory - probs_reloaded).max())
    reload_check_df = pd.DataFrame([
        {"metric": "test_rows_scored", "value": str(len(X_test))},
        {"metric": "reload_max_abs_probability_diff", "value": str(round(diff, 4)) if diff > 0 else "0.0"},
        {"metric": "threshold", "value": str(threshold)},
    ])
    reload_check_df.to_csv("outputs/part_d_reload_check.csv", index=False)
    print("Saved outputs/part_d_reload_check.csv")


if __name__ == "__main__":
    run_package_model()

