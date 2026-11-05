"""Part A: Data Foundation and Delay Evidence."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES


def run_part_a():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load raw data
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    raw_rows = len(raw)

    # Dedup check
    dedup = raw.drop_duplicates().copy()
    duplicate_rows_removed = raw_rows - len(dedup)
    rows_after_dedup = len(dedup)

    # Missing counts before any filling (measured after removing duplicates)
    # missing_parcel_weight_before does not include sentinel rows (>100)
    missing_distance_km_before = int(dedup["distance_km"].isna().sum())
    sentinel_weight_rows = int((dedup["parcel_weight_kg"] > 100).sum())
    missing_parcel_weight_before = int(dedup["parcel_weight_kg"].isna().sum())
    missing_declared_value_before = int(dedup["declared_value"].isna().sum())
    missing_payment_mode_before = int(dedup["payment_mode"].isna().sum())

    service_type_spellings_before = int(dedup["service_type"].nunique())

    # 2. Run clean_shipments
    clean, medians = clean_shipments(raw)
    service_type_spellings_after = int(clean["service_type"].nunique())

    # 3. Run build_features
    df = build_features(clean, hubs)
    rows_after_hub_merge = len(df)

    # Add target
    df["is_delayed"] = (df["delivery_hours"] > df["promised_hours"]).astype(int)

    # Save data/modelling_table.csv
    df.to_csv("data/modelling_table.csv", index=False)
    print(f"Saved data/modelling_table.csv with {len(df)} rows and {len(df.columns)} columns.")

    # 4. Compute metrics for part_a_checks.csv
    nulls_in_features_and_target = int(df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())
    delayed_count = int(df["is_delayed"].sum())
    delay_rate = float(df["is_delayed"].mean())

    q1 = float(df["delivery_hours"].quantile(0.25))
    q3 = float(df["delivery_hours"].quantile(0.75))
    iqr = q3 - q1
    lb = float(q1 - 1.5 * iqr)
    ub = float(q3 + 1.5 * iqr)
    hours_outlier_count = int(((df["delivery_hours"] < lb) | (df["delivery_hours"] > ub)).sum())

    declared_value_mean_inr = float(df["declared_value_inr"].mean())
    declared_value_median_inr = float(df["declared_value_inr"].median())
    log_declared_value_mean = float(df["log_declared_value"].mean())
    log_declared_value_median = float(df["log_declared_value"].median())

    checks_data = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_distance_km_before),
        ("missing_parcel_weight_before", missing_parcel_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_declared_value_before),
        ("missing_payment_mode_before", missing_payment_mode_before),
        ("service_type_spellings_before", service_type_spellings_before),
        ("service_type_spellings_after", service_type_spellings_after),
        ("median_distance_km", round(medians["distance_km"], 4)),
        ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 4)),
        ("median_declared_value_inr", round(medians["declared_value_inr"], 4)),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", round(delay_rate, 4)),
        ("hours_q1", round(q1, 4)),
        ("hours_q3", round(q3, 4)),
        ("hours_lower_bound", round(lb, 4)),
        ("hours_upper_bound", round(ub, 4)),
        ("hours_outlier_count", hours_outlier_count),
        ("declared_value_mean_inr", round(declared_value_mean_inr, 4)),
        ("declared_value_median_inr", round(declared_value_median_inr, 4)),
        ("log_declared_value_mean", round(log_declared_value_mean, 4)),
        ("log_declared_value_median", round(log_declared_value_median, 4)),
    ]

    def fmt_val(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        elif isinstance(v, (float, np.floating)):
            r = round(float(v), 4)
            return str(int(r)) if r.is_integer() else str(r)
        return str(v)

    checks_df = pd.DataFrame([
        {"metric": m, "value": fmt_val(v)} for m, v in checks_data
    ])
    checks_df.to_csv("outputs/part_a_checks.csv", index=False)
    print("Saved outputs/part_a_checks.csv")

    # 5. Delay evidence table: outputs/part_a_delay_by_group.csv
    # distance_band: cut into 0-200, 201-400, 401-600, 601+ (edges 0, 200, 400, 600, 2000, right edge inclusive)
    labels = ["0-200", "201-400", "401-600", "601+"]
    df["distance_band"] = pd.cut(df["distance_km"], bins=[0, 200, 400, 600, 2000], labels=labels, right=True)

    group_rows = []
    # service_type (Economy, Express, Standard)
    for st in ["Economy", "Express", "Standard"]:
        sub = df[df["service_type"] == st]
        group_rows.append({
            "dimension": "service_type",
            "group": st,
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    # hub_tier (1, 2, 3)
    for ht in [1, 2, 3]:
        sub = df[df["hub_tier"] == ht]
        group_rows.append({
            "dimension": "hub_tier",
            "group": str(ht),
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    # distance_band
    for db in labels:
        sub = df[df["distance_band"] == db]
        group_rows.append({
            "dimension": "distance_band",
            "group": db,
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    delay_group_df = pd.DataFrame(group_rows)
    delay_group_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)
    print("Saved outputs/part_a_delay_by_group.csv")

    # 6. Correlations: outputs/part_a_correlations.csv
    # Correlation matrix of numeric columns of modelling table, row labels kept as first column
    num_cols = df.select_dtypes(include=["number"]).columns
    corr_df = df[num_cols].corr().round(4)
    corr_df.to_csv("outputs/part_a_correlations.csv", index=True, index_label="feature")
    print("Saved outputs/part_a_correlations.csv")


if __name__ == "__main__":
    run_part_a()
