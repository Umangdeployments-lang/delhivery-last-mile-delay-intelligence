"""Scoring script for new shipments.

Turns a batch of new shipments into actionable operations decisions.
Combines Part C's delay classification pipeline with Part D's hub segments.
"""
import os
import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features


def score_new_shipments():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data and models
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    hub_segments = pd.read_csv("outputs/hub_segments.csv")

    model = joblib.load("models/delay_model.pkl")
    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"]))

    # 2. Preprocess new shipments
    # Keep original file order
    clean_df, _ = clean_shipments(new_raw, medians=medians)
    features_df = build_features(clean_df, hubs)

    # 3. Predict delay probabilities
    X_new = features_df[feature_order]
    probs = model.predict_proba(X_new)[:, 1]
    features_df["delay_probability"] = np.round(probs, 4)
    features_df["flagged"] = (features_df["delay_probability"] >= threshold).astype(int)

    # 4. Join hub segment
    features_df = features_df.merge(
        hub_segments[["hub_id", "segment"]], on="hub_id", how="left"
    )

    # 5. Determine action
    def get_action(row):
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                return "Escalate to hub manager"
            return "Proactive customer call"
        return "No action"

    features_df["action"] = features_df.apply(get_action, axis=1)

    # 6. Arrange output columns in original file order
    output_cols = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
    actions_df = features_df[output_cols].copy()

    output_path = "outputs/new_shipment_actions.csv"
    actions_df.to_csv(output_path, index=False)
    print(f"Saved {output_path} with {len(actions_df)} records.")
    return actions_df


if __name__ == "__main__":
    score_new_shipments()

