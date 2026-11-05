"""Part C: Delay Classifier and Alert Threshold."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS


def run_part_c():
    os.makedirs("outputs", exist_ok=True)

    # Load data
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # Task 1: Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"Train size: {len(X_train)} (delayed: {y_train.sum()})")
    print(f"Val size: {len(X_val)} (delayed: {y_val.sum()})")
    print(f"Test size: {len(X_test)} (delayed: {y_test.sum()})")

    # Task 2: Baseline and initial models scored on val with threshold 0.5
    models_rows = []

    # 1. Blind baseline (predict 0 for all)
    blind_preds = np.zeros(len(y_val), dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_val, blind_preds).ravel()
    models_rows.append({
        "model": "blind_all_on_time",
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": round(accuracy_score(y_val, blind_preds), 4),
        "precision": 0.0,
        "recall": round(recall_score(y_val, blind_preds, zero_division=0), 4),
        "f1": round(f1_score(y_val, blind_preds, zero_division=0), 4),
    })

    # 2. Logistic Regression Pipeline
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, lr_preds).ravel()
    models_rows.append({
        "model": "logistic_regression",
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": round(accuracy_score(y_val, lr_preds), 4),
        "precision": round(precision_score(y_val, lr_preds, zero_division=0), 4),
        "recall": round(recall_score(y_val, lr_preds, zero_division=0), 4),
        "f1": round(f1_score(y_val, lr_preds, zero_division=0), 4),
    })

    # 3. Decision Tree depth search
    dt_candidates = [2, 3, 4, 5, 6, 8, 10]
    best_dt_d = None
    best_dt_f1 = -1.0
    best_dt_metrics = None

    for d in dt_candidates:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        dt_preds = dt.predict(X_val)
        val_f1 = f1_score(y_val, dt_preds, zero_division=0)
        if val_f1 > best_dt_f1:
            best_dt_f1 = val_f1
            best_dt_d = d
            tn, fp, fn, tp = confusion_matrix(y_val, dt_preds).ravel()
            best_dt_metrics = {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "accuracy": round(accuracy_score(y_val, dt_preds), 4),
                "precision": round(precision_score(y_val, dt_preds, zero_division=0), 4),
                "recall": round(recall_score(y_val, dt_preds, zero_division=0), 4),
                "f1": round(val_f1, 4),
            }

    models_rows.append({
        "model": f"decision_tree_depth_{best_dt_d}",
        **best_dt_metrics
    })

    # 4. Default Random Forest
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    rf_preds = rf_def.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, rf_preds).ravel()
    models_rows.append({
        "model": "random_forest_default",
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": round(accuracy_score(y_val, rf_preds), 4),
        "precision": round(precision_score(y_val, rf_preds, zero_division=0), 4),
        "recall": round(recall_score(y_val, rf_preds, zero_division=0), 4),
        "f1": round(f1_score(y_val, rf_preds, zero_division=0), 4),
    })

    models_df = pd.DataFrame(models_rows)
    models_df.to_csv("outputs/part_c_models.csv", index=False)
    print("Saved outputs/part_c_models.csv")

    # Task 3: Signal check & SelectK
    # fit default forest on all 17 features
    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)

    feature_scores = pd.DataFrame({
        "feature": FEATURES,
        "f_score": selector_all.scores_,
        "rf_importance": rf_def.feature_importances_,
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
    })
    feature_scores["f_rank"] = feature_scores["f_score"].rank(ascending=False, method="min").astype(int)
    feature_scores = feature_scores.sort_values("f_rank").reset_index(drop=True)
    # Arrange columns: feature, f_score, f_rank, rf_importance, is_noise
    feature_scores = feature_scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    feature_scores["f_score"] = feature_scores["f_score"].round(4)
    feature_scores["rf_importance"] = feature_scores["rf_importance"].round(4)
    feature_scores.to_csv("outputs/part_c_feature_scores.csv", index=False)
    print("Saved outputs/part_c_feature_scores.csv")

    # SelectK experiments
    k_values = [5, 7, 10, 12, 17]
    selectk_rows = []
    for k in k_values:
        pipe_k = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        pipe_k.fit(X_train, y_train)
        k_preds = pipe_k.predict(X_val)
        val_f1_k = f1_score(y_val, k_preds, zero_division=0)
        selectk_rows.append({
            "k": k,
            "val_f1": round(val_f1_k, 4),
        })
    selectk_df = pd.DataFrame(selectk_rows)
    selectk_df.to_csv("outputs/part_c_selectk.csv", index=False)
    print("Saved outputs/part_c_selectk.csv")

    # Task 4: Tuning with k=7
    selector_7 = SelectKBest(f_classif, k=7)
    X_train_7 = selector_7.fit_transform(X_train, y_train)
    X_val_7 = selector_7.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }

    # GridSearchCV (27 candidates x 5 cv_folds = 135 total fits)
    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    gs.fit(X_train_7, y_train)
    gs_best_model = gs.best_estimator_
    gs_val_preds = gs_best_model.predict(X_val_7)
    gs_val_f1 = f1_score(y_val, gs_val_preds, zero_division=0)

    # RandomizedSearchCV (9 candidates x 5 cv_folds = 45 total fits)
    rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42)
    rs.fit(X_train_7, y_train)
    rs_best_model = rs.best_estimator_
    rs_val_preds = rs_best_model.predict(X_val_7)
    rs_val_f1 = f1_score(y_val, rs_val_preds, zero_division=0)

    # Default forest on 7 features
    rf_default_7 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_default_7.fit(X_train_7, y_train)
    def7_val_preds = rf_default_7.predict(X_val_7)
    def7_val_f1 = f1_score(y_val, def7_val_preds, zero_division=0)

    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(gs.best_params_),
            "best_cv_accuracy": round(float(gs.best_score_), 4),
            "val_f1": round(gs_val_f1, 4),
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rs.best_params_),
            "best_cv_accuracy": round(float(rs.best_score_), 4),
            "val_f1": round(rs_val_f1, 4),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": "",
            "val_f1": round(def7_val_f1, 4),
        },
    ]
    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)
    print("Saved outputs/part_c_tuning.csv")

    # Task 5: Choosing final configuration
    # Ties -> default_forest_same_7_features, then grid_search, then randomized_search
    # Highest val_f1
    # Check candidates
    tuning_candidates = [
        ("default_forest_same_7_features", def7_val_f1, {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}),
        ("grid_search", gs_val_f1, gs.best_params_),
        ("randomized_search", rs_val_f1, rs.best_params_),
    ]
    # sort by val_f1 descending, maintaining tie-breaker order
    winning_candidate = sorted(tuning_candidates, key=lambda x: x[1], reverse=True)[0]
    chosen_from = winning_candidate[0]
    chosen_params = winning_candidate[2]
    print(f"Final chosen configuration from: {chosen_from} with params: {chosen_params}")

    # Build final model Pipeline
    if chosen_from == "default_forest_same_7_features":
        final_model = Pipeline([
            ("selector", SelectKBest(f_classif, k=7)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
    else:
        final_model = Pipeline([
            ("selector", SelectKBest(f_classif, k=7)),
            ("model", RandomForestClassifier(random_state=42, **chosen_params)),
        ])
    final_model.fit(X_train, y_train)

    # Task 6: Alert threshold
    val_probs = final_model.predict_proba(X_val)[:, 1]
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    th_rows = []
    chosen_threshold = 0.1

    for t in threshold_list:
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1_val = f1_score(y_val, preds, zero_division=0)

        th_rows.append({
            "threshold": t,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1_val, 4),
        })

        if rec >= 0.80:
            chosen_threshold = t

    th_df = pd.DataFrame(th_rows)
    th_df.to_csv("outputs/part_c_thresholds.csv", index=False)
    print(f"Saved outputs/part_c_thresholds.csv (chosen threshold: {chosen_threshold})")

    # Task 7: One test evaluation
    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)

    final_test_df = pd.DataFrame([{
        "threshold": chosen_threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": round(test_acc, 4),
        "precision": round(test_prec, 4),
        "recall": round(test_rec, 4),
        "f1": round(test_f1, 4),
    }])
    final_test_df.to_csv("outputs/part_c_final_test.csv", index=False)
    print("Saved outputs/part_c_final_test.csv")

    # Config CSV
    max_d_val = "unlimited" if chosen_params.get("max_depth") is None else str(chosen_params.get("max_depth"))
    final_config_df = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": chosen_params.get("n_estimators", 100),
        "max_depth": max_d_val,
        "min_samples_leaf": chosen_params.get("min_samples_leaf", 1),
        "threshold": chosen_threshold,
        "chosen_from": chosen_from,
    }])
    final_config_df.to_csv("outputs/part_c_final_config.csv", index=False)
    print("Saved outputs/part_c_final_config.csv")


if __name__ == "__main__":
    run_part_c()
