"""Part D: Hub Segments and Profiling."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform


def run_part_d_segments():
    os.makedirs("outputs", exist_ok=True)

    # Load data
    df = pd.read_csv("data/modelling_table.csv")
    hubs_df = pd.read_csv("data/hubs.csv")

    # Task 1: Hub profile
    profile = df.groupby("hub_id").agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    ).reset_index().sort_values("hub_id").reset_index(drop=True)

    # Save outputs/hub_profile.csv (24 rows, sorted by ascending hub_id)
    profile_out = profile.copy()
    profile_out["mean_delivery_hours"] = profile_out["mean_delivery_hours"].round(4)
    profile_out["delay_rate"] = profile_out["delay_rate"].round(4)
    profile_out["mean_distance_km"] = profile_out["mean_distance_km"].round(4)
    profile_out["mean_n_stops"] = profile_out["mean_n_stops"].round(4)
    profile_out.to_csv("outputs/hub_profile.csv", index=False)
    print("Saved outputs/hub_profile.csv")

    # Task 2 & 3: Scaling and KMeans segmentation
    cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profile[cols])

    # Task 3: Separation evidence for K in 2, 3, 4, 5
    dists = squareform(pdist(X_scaled, metric="euclidean"))
    n_hubs = len(profile)

    separation_rows = []
    for k in [2, 3, 4, 5]:
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
        k_labels = km_k.fit_predict(X_scaled)

        # Cluster sizes descending joined by hyphen
        counts = pd.Series(k_labels).value_counts().sort_values(ascending=False)
        sizes_str = "-".join(counts.astype(str))

        within_dists = []
        across_dists = []
        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                d = dists[i, j]
                if k_labels[i] == k_labels[j]:
                    within_dists.append(d)
                else:
                    across_dists.append(d)

        ratio = np.mean(across_dists) / np.mean(within_dists)
        separation_rows.append({
            "K": k,
            "across_over_within": round(float(ratio), 4),
            "cluster_sizes": sizes_str,
        })

    separation_df = pd.DataFrame(separation_rows)
    separation_df.to_csv("outputs/part_d_separation.csv", index=False)
    print("Saved outputs/part_d_separation.csv")

    # Task 4: Segment names and tier check for K = 3
    km_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = km_3.fit_predict(X_scaled)
    profile["cluster"] = clusters

    # Order clusters by mean delay_rate: Standard (lowest), Watch (middle), Escalate (highest)
    cluster_delay_means = profile.groupby("cluster")["delay_rate"].mean().sort_values()
    name_map = {
        cluster_delay_means.index[0]: "Standard",
        cluster_delay_means.index[1]: "Watch",
        cluster_delay_means.index[2]: "Escalate",
    }
    profile["segment"] = profile["cluster"].map(name_map)

    # Attach hub_city, hub_tier
    profile = profile.merge(hubs_df[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")

    # outputs/hub_segments.csv (hub_id, segment, hub_city, hub_tier)
    hub_segments_df = profile[["hub_id", "segment", "hub_city", "hub_tier"]].sort_values("hub_id")
    hub_segments_df.to_csv("outputs/hub_segments.csv", index=False)
    print("Saved outputs/hub_segments.csv")

    # outputs/segment_profile.csv (segment, hubs, mean_delivery_hours, delay_rate, mean_distance_km, mean_n_stops)
    seg_profile = profile.groupby("segment").agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    ).reindex(["Standard", "Watch", "Escalate"]).reset_index()

    seg_profile["mean_delivery_hours"] = seg_profile["mean_delivery_hours"].round(4)
    seg_profile["delay_rate"] = seg_profile["delay_rate"].round(4)
    seg_profile["mean_distance_km"] = seg_profile["mean_distance_km"].round(4)
    seg_profile["mean_n_stops"] = seg_profile["mean_n_stops"].round(4)
    seg_profile.to_csv("outputs/segment_profile.csv", index=False)
    print("Saved outputs/segment_profile.csv")

    # outputs/tier_vs_segment_mismatch.csv
    # Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    tier_expected = {1: "Standard", 2: "Watch", 3: "Escalate"}
    profile["expected_segment"] = profile["hub_tier"].map(tier_expected)
    mismatches = profile[profile["segment"] != profile["expected_segment"]]
    mismatch_df = mismatches[["hub_id", "segment", "hub_city", "hub_tier"]].sort_values("hub_id")
    mismatch_df.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
    print("Saved outputs/tier_vs_segment_mismatch.csv")


if __name__ == "__main__":
    run_part_d_segments()
