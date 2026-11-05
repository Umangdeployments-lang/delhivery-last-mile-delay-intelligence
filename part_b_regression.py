"""Part B: Delivery-Hours Regression."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES


def run_part_b():
    os.makedirs("outputs", exist_ok=True)

    # Load modelling table
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    # Task 1: Train, Val, Test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Task 2: Scaling without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = float(np.abs(X_train_scaled.mean(axis=0)).max())
    test_scaled_max_abs_mean = float(np.abs(X_test_scaled.mean(axis=0)).max())

    checks_df = pd.DataFrame([
        {"metric": "train_rows", "value": str(len(X_train))},
        {"metric": "val_rows", "value": str(len(X_val))},
        {"metric": "test_rows", "value": str(len(X_test))},
        {"metric": "train_scaled_max_abs_mean", "value": f"{train_scaled_max_abs_mean:.4e}"},
        {"metric": "test_scaled_max_abs_mean", "value": str(round(test_scaled_max_abs_mean, 4))},
    ])
    checks_df.to_csv("outputs/part_b_checks.csv", index=False)
    print("Saved outputs/part_b_checks.csv")

    # Task 3: Baseline
    y_val_baseline = np.full_like(y_val, y_train.mean())
    base_mae = mean_absolute_error(y_val, y_val_baseline)
    base_mse = mean_squared_error(y_val, y_val_baseline)
    base_rmse = np.sqrt(base_mse)
    base_r2 = r2_score(y_val, y_val_baseline)

    # Task 4: Tree-depth diagnostic
    depth_candidates = [1, 2, 3, 4, 6, 8, 12, None]
    tree_rows = []
    best_dt_depth = None
    best_dt_rmse = float("inf")

    for d in depth_candidates:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        tr_pred = dt.predict(X_train)
        val_pred = dt.predict(X_val)

        tr_rmse = np.sqrt(mean_squared_error(y_train, tr_pred))
        v_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        d_str = "unlimited" if d is None else str(d)
        tree_rows.append({
            "max_depth": d_str,
            "train_rmse": round(tr_rmse, 4),
            "val_rmse": round(v_rmse, 4),
        })

        if v_rmse < best_dt_rmse:
            best_dt_rmse = v_rmse
            best_dt_depth = d

    tree_depth_df = pd.DataFrame(tree_rows)
    tree_depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)
    print(f"Saved outputs/part_b_tree_depth.csv (best depth: {best_dt_depth})")

    # Task 5: Four candidate models
    dt_chosen = DecisionTreeRegressor(max_depth=best_dt_depth, random_state=42)
    models = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": dt_chosen,
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metrics_rows = [
        {
            "model": "mean_baseline",
            "split": "val",
            "MAE": round(base_mae, 4),
            "MSE": round(base_mse, 4),
            "RMSE": round(base_rmse, 4),
            "R2": round(base_r2, 4),
        }
    ]

    val_rmses = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        val_rmses[name] = rmse

        metrics_rows.append({
            "model": name,
            "split": "val",
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        })

    # Task 6: Learning-rate experiment
    lr_rows = []
    for lr in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        preds = gb.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        lr_rows.append({
            "learning_rate": lr,
            "val_rmse": round(rmse, 4),
            "val_r2": round(r2, 4),
        })
    lr_df = pd.DataFrame(lr_rows)
    lr_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    print("Saved outputs/part_b_gb_learning_rate.csv")

    # Task 7: Cross-validated comparison and final model
    # Rank candidates by validation RMSE
    ranked_models = sorted(val_rmses.items(), key=lambda x: x[1])
    top_two = [ranked_models[0][0], ranked_models[1][0]]
    print(f"Top two models by validation RMSE: {top_two}")

    # 5-fold CV on 1020 training-plus-validation rows
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_rows = []
    for name in top_two:
        scores = cross_val_score(models[name], X_train_val, y_train_val, cv=kf)
        cv_rows.append({
            "model": name,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv("outputs/part_b_cv.csv", index=False)
    print("Saved outputs/part_b_cv.csv")

    # Final model is lowest validation RMSE
    final_model_name = ranked_models[0][0]
    final_model = models[final_model_name]
    # Score on test rows (already fitted on training rows)
    test_preds = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_preds)

    metrics_rows.append({
        "model": final_model_name,
        "split": "test",
        "MAE": round(test_mae, 4),
        "MSE": round(test_mse, 4),
        "RMSE": round(test_rmse, 4),
        "R2": round(test_r2, 4),
    })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)
    print("Saved outputs/part_b_metrics.csv")


if __name__ == "__main__":
    run_part_b()
