#!/usr/bin/env python3
"""
Direct fingerprint-based modeling workflow for GAC removal datasets.

Supported tasks
---------------
regression:
    Fits an XGBoost regressor for logBV10 using molecular fingerprints plus
    BV10 non-structural variables such as C0, pH, DOC, UV254, mp_ratio, BET,
    pzc, EBCT, PD, and CD when these columns are available.

classification:
    Fits an XGBoost classifier for binary removal efficiency classes using
    molecular fingerprints plus DOC only.

The workflow uses grouped nested 5 x 2 cross-validation and writes TSV outputs.
"""

import argparse
import os
import re
import warnings
from typing import Dict, List

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


REGRESSION_NON_STRUCTURAL_DEFAULTS = [
    "C0",
    "pH",
    "DOC",
    "UV254",
    "mp_ratio",
    "BET",
    "pzc",
    "EBCT",
    "PD",
    "CD",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def parse_number_or_mean(value):
    if pd.isna(value):
        return np.nan

    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    if len(numbers) == 0:
        return np.nan

    numbers = [float(x) for x in numbers]
    return float(np.mean(numbers))


def split_comma_separated(value: str | None) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def convert_xgb_params(params: Dict) -> Dict:
    params = params.copy()
    for key in ["n_estimators", "max_depth"]:
        if key in params:
            params[key] = int(params[key])
    return params


def safe_r2(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def resolve_doc_column(df: pd.DataFrame, requested_doc_col: str | None) -> str:
    candidates = []
    if requested_doc_col:
        candidates.append(requested_doc_col)
    candidates.extend(["DOC", "DOC (mg/L)", "DOC_mean"])

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError("Could not find a DOC column. Tried DOC, DOC (mg/L), and DOC_mean.")


def prepare_doc_column(df: pd.DataFrame, doc_col: str, parsed_doc_col: str = "DOC_mean") -> str:
    numeric = pd.to_numeric(df[doc_col], errors="coerce")
    if numeric.notna().sum() == df[doc_col].notna().sum():
        df[parsed_doc_col] = numeric
    else:
        df[parsed_doc_col] = df[doc_col].apply(parse_number_or_mean)
    return parsed_doc_col


def load_data(
    data_path: str,
    task: str,
    target_col: str,
    group_col: str,
    non_structural_cols: List[str],
    doc_col: str | None,
    fingerprint_prefix: str,
):
    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    if group_col not in df.columns:
        raise ValueError(f"Group column not found: {group_col}")

    if task == "classification":
        raw_doc_col = resolve_doc_column(df, doc_col)
        doc_model_col = prepare_doc_column(df, raw_doc_col)
        non_structural_cols = [doc_model_col]
    else:
        doc_model_col = None
        resolved_non_structural = []
        for col in non_structural_cols:
            if col == "DOC":
                raw_doc_col = resolve_doc_column(df, doc_col)
                doc_model_col = prepare_doc_column(df, raw_doc_col)
                resolved_non_structural.append(doc_model_col)
            elif col in df.columns:
                resolved_non_structural.append(col)
        non_structural_cols = resolved_non_structural

    if len(non_structural_cols) == 0:
        raise ValueError("No requested non-structural variables were found in the input file.")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[group_col] = df[group_col].astype(str)

    if task == "classification":
        df = df[df[target_col].notna()].copy()
        df[target_col] = df[target_col].astype(int)
    else:
        df = df[df[target_col].notna()].copy()

    for col in non_structural_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fingerprint_cols_all = [col for col in df.columns if col.startswith(fingerprint_prefix)]
    if len(fingerprint_cols_all) == 0:
        raise ValueError(f"No fingerprint columns starting with {fingerprint_prefix!r} were found.")

    for col in fingerprint_cols_all:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    fingerprint_cols = [col for col in fingerprint_cols_all if df[col].sum() != 0]
    if len(fingerprint_cols) == 0:
        raise ValueError("All fingerprint columns are zero after filtering.")

    model_cols = non_structural_cols + fingerprint_cols
    keep_cols = [group_col, target_col] + model_cols
    df = df[keep_cols].copy().dropna(subset=[group_col, target_col]).reset_index(drop=True)

    y = df[target_col].values
    groups = df[group_col].astype(str).values

    return df, y, groups, model_cols, non_structural_cols, fingerprint_cols


# -----------------------------------------------------------------------------
# Models and tuning
# -----------------------------------------------------------------------------

def suggest_regression_params(trial: optuna.Trial) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 25, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 3),
        "min_child_weight": trial.suggest_float("min_child_weight", 3.0, 40.0),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 200.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 20.0),
    }


def suggest_classification_params(trial: optuna.Trial) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def make_regressor(params: Dict, seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            **convert_xgb_params(params),
        )),
    ])


def make_classifier(params: Dict, seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            **convert_xgb_params(params),
        )),
    ])


def make_inner_cv(task: str, y_train, groups_train, n_splits: int, seed: int):
    actual_splits = min(n_splits, len(np.unique(groups_train)))
    if actual_splits < 2:
        raise ValueError("At least two unique groups are required for inner cross-validation.")

    if task == "classification":
        return StratifiedGroupKFold(
            n_splits=actual_splits,
            shuffle=True,
            random_state=seed,
        )

    return GroupKFold(n_splits=actual_splits)


def tune_direct_model(X_train, y_train, groups_train, task: str, args, seed: int):
    def objective(trial):
        if task == "classification":
            params = suggest_classification_params(trial)
            cv = make_inner_cv(task, y_train, groups_train, args.inner_splits, seed)
            scores = []

            for tr_idx, va_idx in cv.split(X_train, y_train, groups_train):
                y_tr = y_train[tr_idx]
                y_va = y_train[va_idx]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                    continue

                model = make_classifier(params, seed)
                model.fit(X_train.iloc[tr_idx], y_tr)
                pred = model.predict(X_train.iloc[va_idx])
                scores.append(balanced_accuracy_score(y_va, pred))

            return float(np.mean(scores)) if len(scores) > 0 else 0.0

        params = suggest_regression_params(trial)
        cv = make_inner_cv(task, y_train, groups_train, args.inner_splits, seed)
        scores = []

        for tr_idx, va_idx in cv.split(X_train, y_train, groups_train):
            y_tr = y_train[tr_idx]
            y_va = y_train[va_idx]

            model = make_regressor(params, seed)
            model.fit(X_train.iloc[tr_idx], y_tr)
            pred = model.predict(X_train.iloc[va_idx])
            scores.append(root_mean_squared_error(y_va, pred))

        return float(np.mean(scores))

    direction = "maximize" if task == "classification" else "minimize"
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=args.n_optuna_trials, show_progress_bar=False)

    return study.best_params, study.best_value, study.trials_dataframe()


# -----------------------------------------------------------------------------
# Cross-validation and evaluation
# -----------------------------------------------------------------------------

def make_outer_splits(task: str, y, groups, repeats: int, n_splits: int, seed: int):
    outer_splits = []

    if task == "classification":
        for repeat in range(repeats):
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed + repeat,
            )

            for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups), start=1):
                if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                    raise ValueError(f"Outer repeat {repeat + 1}, fold {fold} lacks both classes.")
                outer_splits.append({
                    "repeat": repeat + 1,
                    "fold": fold,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                })
        return outer_splits

    unique_groups = np.array(sorted(np.unique(groups)))
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)
        fold_group_sets = np.array_split(shuffled_groups, n_splits)

        for fold, test_groups in enumerate(fold_group_sets, start=1):
            test_idx = np.where(np.isin(groups, test_groups))[0]
            train_idx = np.where(~np.isin(groups, test_groups))[0]
            outer_splits.append({
                "repeat": repeat + 1,
                "fold": fold,
                "train_idx": train_idx,
                "test_idx": test_idx,
            })

    return outer_splits


def regression_metrics(y_true, y_pred):
    return {
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": safe_r2(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    metrics["roc_auc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    return metrics


def run_workflow(args):
    if args.task == "regression":
        default_data_path = "data/BV10_regression_set.tsv"
        target_col = args.target_col or "logBV10"
        group_col = args.group_col or "SMILES"
        non_structural_cols = split_comma_separated(args.non_structural_cols)
        if len(non_structural_cols) == 0:
            non_structural_cols = REGRESSION_NON_STRUCTURAL_DEFAULTS.copy()
    else:
        default_data_path = "data/RE_classification_set.tsv"
        target_col = args.target_col or "binary_class"
        group_col = args.group_col or "Compound"
        non_structural_cols = []

    data_path = args.data_path or default_data_path
    ensure_dir(args.output_dir)

    df, y, groups, model_cols, non_structural_cols, fingerprint_cols = load_data(
        data_path=data_path,
        task=args.task,
        target_col=target_col,
        group_col=group_col,
        non_structural_cols=non_structural_cols,
        doc_col=args.doc_col,
        fingerprint_prefix=args.fingerprint_prefix,
    )

    print(f"Rows: {len(df)}")
    print(f"Unique groups: {df[group_col].nunique()}")
    print(f"Non-structural variables: {len(non_structural_cols)}")
    print(non_structural_cols)
    print(f"Fingerprint variables: {len(fingerprint_cols)}")

    X = df[model_cols].copy()
    outer_splits = make_outer_splits(
        task=args.task,
        y=y,
        groups=groups,
        repeats=args.outer_repeats,
        n_splits=args.outer_splits,
        seed=args.random_seed,
    )

    metric_records = []
    prediction_records = []
    tuning_records = []
    best_param_records = []

    for split in outer_splits:
        repeat = split["repeat"]
        fold = split["fold"]
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        seed = args.random_seed + repeat * 100 + fold

        print(f"\nOuter repeat {repeat}, fold {fold}")

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        params, inner_score, trials = tune_direct_model(
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            task=args.task,
            args=args,
            seed=seed,
        )

        trials["repeat"] = repeat
        trials["fold"] = fold
        trials["approach"] = "Direct_FP_plus_nonstructural" if args.task == "regression" else "Direct_FP_plus_DOC"
        tuning_records.append(trials)

        if args.task == "classification":
            model = make_classifier(params, seed)
            model.fit(X_train, y_train)
            train_prob = model.predict_proba(X_train)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]
            train_pred = (train_prob >= 0.5).astype(int)
            test_pred = (test_prob >= 0.5).astype(int)

            train_metrics = classification_metrics(y_train, train_pred, train_prob)
            test_metrics = classification_metrics(y_test, test_pred, test_prob)

            metric_records.append({
                "repeat": repeat,
                "fold": fold,
                "cv_iteration": f"{repeat}_{fold}",
                "approach": "Direct_FP_plus_DOC",
                "n_train_rows": len(train_idx),
                "n_test_rows": len(test_idx),
                "n_train_groups": len(np.unique(groups_train)),
                "n_test_groups": len(np.unique(groups_test)),
                "n_nonstructural_features": len(non_structural_cols),
                "n_fingerprint_features": len(fingerprint_cols),
                "inner_balanced_accuracy": inner_score,
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_f1": train_metrics["f1"],
                "train_roc_auc": train_metrics["roc_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
            })

            for local_i, idx in enumerate(test_idx):
                prediction_records.append({
                    "repeat": repeat,
                    "fold": fold,
                    "cv_iteration": f"{repeat}_{fold}",
                    group_col: df.loc[idx, group_col],
                    "y_true": int(y_test[local_i]),
                    "y_pred": int(test_pred[local_i]),
                    "prob_class_1": float(test_prob[local_i]),
                    "correct": int(y_test[local_i] == test_pred[local_i]),
                })
        else:
            model = make_regressor(params, seed)
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_metrics = regression_metrics(y_train, train_pred)
            test_metrics = regression_metrics(y_test, test_pred)

            metric_records.append({
                "repeat": repeat,
                "fold": fold,
                "cv_iteration": f"{repeat}_{fold}",
                "approach": "Direct_FP_plus_nonstructural",
                "n_train_rows": len(train_idx),
                "n_test_rows": len(test_idx),
                "n_train_groups": len(np.unique(groups_train)),
                "n_test_groups": len(np.unique(groups_test)),
                "n_nonstructural_features": len(non_structural_cols),
                "n_fingerprint_features": len(fingerprint_cols),
                "inner_RMSE": inner_score,
                "train_RMSE": train_metrics["RMSE"],
                "train_MAE": train_metrics["MAE"],
                "train_R2": train_metrics["R2"],
                "test_RMSE": test_metrics["RMSE"],
                "test_MAE": test_metrics["MAE"],
                "test_R2": test_metrics["R2"],
            })

            for local_i, idx in enumerate(test_idx):
                prediction_records.append({
                    "repeat": repeat,
                    "fold": fold,
                    "cv_iteration": f"{repeat}_{fold}",
                    group_col: df.loc[idx, group_col],
                    "y_true": float(y_test[local_i]),
                    "y_pred": float(test_pred[local_i]),
                    "absolute_error": float(abs(y_test[local_i] - test_pred[local_i])),
                    "signed_error": float(test_pred[local_i] - y_test[local_i]),
                })

        best_param_records.append({
            "repeat": repeat,
            "fold": fold,
            "best_inner_score": inner_score,
            "best_params": str(params),
        })

    metrics = pd.DataFrame(metric_records)
    predictions = pd.DataFrame(prediction_records)
    best_params = pd.DataFrame(best_param_records)
    tuning = pd.concat(tuning_records, ignore_index=True) if tuning_records else pd.DataFrame()

    metrics.to_csv(os.path.join(args.output_dir, f"{args.task}_direct_metrics.tsv"), sep="\t", index=False)
    predictions.to_csv(os.path.join(args.output_dir, f"{args.task}_direct_predictions.tsv"), sep="\t", index=False)
    best_params.to_csv(os.path.join(args.output_dir, f"{args.task}_direct_best_params.tsv"), sep="\t", index=False)
    tuning.to_csv(os.path.join(args.output_dir, f"{args.task}_direct_optuna_trials.tsv"), sep="\t", index=False)

    if args.task == "classification":
        summary = metrics.groupby("approach", as_index=False).agg(
            mean_accuracy=("test_accuracy", "mean"),
            std_accuracy=("test_accuracy", "std"),
            mean_balanced_accuracy=("test_balanced_accuracy", "mean"),
            std_balanced_accuracy=("test_balanced_accuracy", "std"),
            mean_f1=("test_f1", "mean"),
            std_f1=("test_f1", "std"),
            mean_roc_auc=("test_roc_auc", "mean"),
            std_roc_auc=("test_roc_auc", "std"),
            median_balanced_accuracy=("test_balanced_accuracy", "median"),
            median_roc_auc=("test_roc_auc", "median"),
        )
    else:
        summary = metrics.groupby("approach", as_index=False).agg(
            mean_RMSE=("test_RMSE", "mean"),
            std_RMSE=("test_RMSE", "std"),
            median_RMSE=("test_RMSE", "median"),
            mean_MAE=("test_MAE", "mean"),
            std_MAE=("test_MAE", "std"),
            median_MAE=("test_MAE", "median"),
            mean_R2=("test_R2", "mean"),
            std_R2=("test_R2", "std"),
            median_R2=("test_R2", "median"),
        ).sort_values("mean_RMSE")

    summary.to_csv(os.path.join(args.output_dir, f"{args.task}_direct_summary.tsv"), sep="\t", index=False)

    print("\nSummary:")
    print(summary)
    print(f"\nDone. Outputs saved to: {args.output_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="Run direct fingerprint-based GAC modeling.")
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default="outputs/direct_models")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--doc-col", default=None)
    parser.add_argument("--non-structural-cols", default=None)
    parser.add_argument("--fingerprint-prefix", default="absoluteIndex_")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--n-optuna-trials", type=int, default=75)
    return parser


if __name__ == "__main__":
    run_workflow(build_parser().parse_args())
