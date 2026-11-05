"""Shared constants and preprocessing functions for the Delay Intelligence pipeline."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw, medians=None):
    """Clean raw shipment DataFrame and return (clean_df, medians).
    
    Steps:
    1. Drop exact duplicate rows.
    2. Strip whitespace and convert service_type to title case (Express, Standard, Economy).
    3. Fill missing payment_mode with 'Unknown'.
    4. Clean declared_value (remove rupee sign and commas), cast to float as declared_value_inr, drop text declared_value.
    5. Convert dispatch_time to datetime.
    6. Values of parcel_weight_kg > 100 are scanner-fault sentinels -> treat as missing (NaN).
    7. Fill missing distance_km, parcel_weight_kg, declared_value_inr with medians.
       If medians is None, compute them from non-missing values after steps 1-6.
    """
    df = raw.drop_duplicates().copy()

    # Step 2: service_type strip & title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # Step 3: payment_mode missing -> "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Step 4: declared_value clean and convert to numeric
    val_series = df["declared_value"].astype(str).str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False).str.strip()
    df["declared_value_inr"] = pd.to_numeric(val_series, errors="coerce")
    df = df.drop(columns=["declared_value"])

    # Step 5: dispatch_time to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # Step 6: parcel_weight_kg sentinel values > 100 to NaN
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # Step 7: compute or apply medians
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }
    else:
        medians = {
            "distance_km": float(medians["distance_km"]),
            "parcel_weight_kg": float(medians["parcel_weight_kg"]),
            "declared_value_inr": float(medians["declared_value_inr"]),
        }

    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians["declared_value_inr"])

    return df, medians


def build_features(clean, hubs):
    """Attach hub information and engineer model features without using delivery_hours.
    
    Returns DataFrame with:
    - hub columns from hubs.csv (hub_city, region, hub_tier, hub_daily_capacity)
    - promised_hours (int)
    - is_cod (int 0/1)
    - is_weekend (int 0/1)
    - dispatch_hour (int)
    - log_declared_value (float natural log)
    """
    df = clean.copy()
    hub_cols = [c for c in ["hub_city", "region", "hub_tier", "hub_daily_capacity"] if c in hubs.columns]
    df = df.merge(hubs[["hub_id"] + hub_cols], on="hub_id", how="left")

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"].astype(float))

    return df
