#!/usr/bin/env python3
"""
Generic residualization workflow for fingerprint-based GAC removal modeling.

Two tasks are supported:
    1. regression:      non-structural variables -> continuous target
                        fingerprints -> residuals
    2. classification:  non-structural variables -> class probability
                        fingerprints -> probability residuals

The workflow fits a direct model and a residualized two-stage model using
nested grouped 5 x 2 cross-validation. All output files are written as TSV files.
"""

import argparse
import os
import re
import warnings
from typing import Dict, Iterable, List, Tuple

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


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def parse_number_or_mean(value):
    if pd.isna(value):
        return np.nan

    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    if len(numbers) == 0:
        return np.nan

    numbers = [float(x) for x in numbers]
    return float(np.mean(numbers))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def split_comma_separated(value: str) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def convert_xgb_params(params: Dict) -> Dict:
    params = params.copy()
    for key in ["n_estimators", "max_depth"]:
        if key in params:
            params[key] = int(params[key])
    return params


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_modeling_data(
    data_path: str,
    task: str,
    target_col: str,
    group_col: str,
    non_structural_cols: Iterable[str],
    doc_col: str | None = None,
    parsed_doc_col: str = "DOC_mean",
    fingerprint_prefix: str = "absoluteIndex_",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if group_col not in df.columns:
        raise ValueError(f"Group column not found: {group_col}")
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    non_structural_cols = list(non_structural_cols)

    if doc_col is not None:
        if doc_col not in df.columns:
            raise ValueError(f"DOC column not found: {doc_col}")
        df[parsed_doc_col] = df[doc_col].apply(parse_number_or_mean)
        non_structural_cols = [parsed_doc_col if c == doc_col else c for c in non_structural_cols]

    non_structural_cols = [c for c in non_structural_cols if c in df.columns]
    if len(non_structural_cols) == 0:
        raise ValueError("No non-structural columns were found in the input data.")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[group_col] = df[group_col].astype(str)

    if task == "classification":
        df = df[df[target_col].notna()].copy()
        df[target_col] = df[target_col].astype(int)
    else:
        df = df[df[target_col].notna()].copy()

    for col in non_structural_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fingerprint_cols_all = [c for c in df.columns if c.startswith(fingerprint_prefix)]
    if len(fingerprint_cols_all) == 0:
        raise ValueError(f"No fingerprint columns starting with {fingerprint_prefix!r} were found.")

    for col in fingerprint_cols_all:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    fingerprint_cols = [c for c in fingerprint_cols_all if df[c].sum() != 0]

    required_cols = [group_col, target_col] + non_structural_cols + fingerprint_cols
    df = df[required_cols].copy()
    df = df.dropna(subset=[group_col, target_col]).reset_index(drop=True)

    y = df[target_col].values
    groups = df[group_col].astype(str).values

    return df, y, groups, non_structural_cols, fingerprint_cols


# -----------------------------------------------------------------------------
# Model definitions and hyperparameter spaces
# -----------------------------------------------------------------------------

def suggest_xgb_regression_params(trial: optuna.Trial) -> Dict:
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


def suggest_xgb_classification_params(trial: optuna.Trial) -> Dict:
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


def make_regressor(params: Dict, seed: int, impute_strategy: str) -> Pipeline:
    imputer = SimpleImputer(
        strategy="constant",
        fill_value=0,
    ) if impute_strategy == "zero" else SimpleImputer(strategy="median")

    return Pipeline([
        ("imputer", imputer),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            **convert_xgb_params(params),
        )),
    ])


def make_classifier(params: Dict, seed: int, impute_strategy: str) -> Pipeline:
    imputer = SimpleImputer(
        strategy="constant",
        fill_value=0,
    ) if impute_strategy == "zero" else SimpleImputer(strategy="median")

    return Pipeline([
        ("imputer", imputer),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            **convert_xgb_params(params),
        )),
    ])


# -----------------------------------------------------------------------------
# Cross-validation helpers
# -----------------------------------------------------------------------------

def make_outer_splits(
    task: str,
    y: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    splits: int,
    seed: int,
) -> List[Dict]:
    output = []

    if task == "classification":
        for repeat in range(repeats):
            cv = StratifiedGroupKFold(
                n_splits=splits,
                shuffle=True,
                random_state=seed + repeat,
            )
            for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups), start=1):
                if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                    raise ValueError(f"Outer repeat {repeat + 1}, fold {fold} lacks both classes.")
                output.append({"repeat": repeat + 1, "fold": fold, "train_idx": train_idx, "test_idx": test_idx})
        return output

    unique_groups = np.array(sorted(np.unique(groups)))
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)
        fold_group_sets = np.array_split(shuffled_groups, splits)

        for fold, test_groups in enumerate(fold_group_sets, start=1):
            test_idx = np.where(np.isin(groups, test_groups))[0]
            train_idx = np.where(~np.isin(groups, test_groups))[0]
            output.append({"repeat": repeat + 1, "fold": fold, "train_idx": train_idx, "test_idx": test_idx})

    return output


def make_inner_cv(task: str, y_train: np.ndarray, groups_train: np.ndarray, inner_splits: int, seed: int):
    n_splits = min(inner_splits, len(np.unique(groups_train)))
    if n_splits < 2:
        raise ValueError("At least two unique groups are required for inner CV.")

    if task == "classification":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    return GroupKFold(n_splits=n_splits)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": safe_r2(y_true, y_pred),
    }


def get_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else np.nan
    return out


# -----------------------------------------------------------------------------
# Tuning functions
# -----------------------------------------------------------------------------

def tune_direct_model(
    task: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    model_cols: List[str],
    inner_splits: int,
    n_trials: int,
    seed: int,
) -> Tuple[Dict, float, pd.DataFrame]:
    def objective(trial):
        params = suggest_xgb_classification_params(trial) if task == "classification" else suggest_xgb_regression_params(trial)
        cv = make_inner_cv(task, y_train, groups_train, inner_splits, seed)
        scores = []

        for tr_idx, va_idx in cv.split(X_train, y_train, groups_train):
            y_tr = y_train[tr_idx]
            y_va = y_train[va_idx]

            if task == "classification" and (len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2):
                continue

            X_tr = X_train.iloc[tr_idx][model_cols]
            X_va = X_train.iloc[va_idx][model_cols]

            if task == "classification":
                model = make_classifier(params, seed, impute_strategy="median")
                model.fit(X_tr, y_tr)
                pred = model.predict(X_va)
                scores.append(balanced_accuracy_score(y_va, pred))
            else:
                model = make_regressor(params, seed, impute_strategy="median")
                model.fit(X_tr, y_tr)
                pred = model.predict(X_va)
                scores.append(root_mean_squared_error(y_va, pred))

        if len(scores) == 0:
            return 0.0 if task == "classification" else np.inf
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize" if task == "classification" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value, study.trials_dataframe()


def tune_stage1_model(
    task: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    non_structural_cols: List[str],
    inner_splits: int,
    n_trials: int,
    seed: int,
) -> Tuple[Dict, float, pd.DataFrame]:
    return tune_direct_model(
        task=task,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        model_cols=non_structural_cols,
        inner_splits=inner_splits,
        n_trials=n_trials,
        seed=seed,
    )


def make_cross_fitted_stage1_predictions(
    task: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    non_structural_cols: List[str],
    stage1_params: Dict,
    inner_splits: int,
    seed: int,
) -> np.ndarray:
    cv = make_inner_cv(task, y_train, groups_train, inner_splits, seed)
    oof_pred = np.zeros(len(y_train))

    for tr_idx, va_idx in cv.split(X_train, y_train, groups_train):
        y_tr = y_train[tr_idx]

        model = make_classifier(stage1_params, seed, "median") if task == "classification" else make_regressor(stage1_params, seed, "median")
        model.fit(X_train.iloc[tr_idx][non_structural_cols], y_tr)

        if task == "classification":
            oof_pred[va_idx] = model.predict_proba(X_train.iloc[va_idx][non_structural_cols])[:, 1]
        else:
            oof_pred[va_idx] = model.predict(X_train.iloc[va_idx][non_structural_cols])

    return oof_pred


def tune_stage2_residual_model(
    task: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    non_structural_cols: List[str],
    fingerprint_cols: List[str],
    stage1_params: Dict,
    inner_splits: int,
    n_trials: int,
    seed: int,
) -> Tuple[Dict, float, pd.DataFrame]:
    def objective(trial):
        params = suggest_xgb_regression_params(trial)
        cv = make_inner_cv(task, y_train, groups_train, inner_splits, seed)
        scores = []

        for tr_idx, va_idx in cv.split(X_train, y_train, groups_train):
            y_tr = y_train[tr_idx]
            y_va = y_train[va_idx]
            groups_tr = groups_train[tr_idx]

            if task == "classification" and (len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2):
                continue

            X_tr = X_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]

            stage1_oof_tr = make_cross_fitted_stage1_predictions(
                task=task,
                X_train=X_tr,
                y_train=y_tr,
                groups_train=groups_tr,
                non_structural_cols=non_structural_cols,
                stage1_params=stage1_params,
                inner_splits=inner_splits,
                seed=seed,
            )
            residual_tr = y_tr.astype(float) - stage1_oof_tr

            stage1_model = make_classifier(stage1_params, seed, "median") if task == "classification" else make_regressor(stage1_params, seed, "median")
            stage1_model.fit(X_tr[non_structural_cols], y_tr)

            if task == "classification":
                stage1_va = stage1_model.predict_proba(X_va[non_structural_cols])[:, 1]
            else:
                stage1_va = stage1_model.predict(X_va[non_structural_cols])

            residual_model = make_regressor(params, seed, impute_strategy="zero")
            residual_model.fit(X_tr[fingerprint_cols], residual_tr)
            residual_va = residual_model.predict(X_va[fingerprint_cols])

            final_va = stage1_va + residual_va

            if task == "classification":
                final_prob_va = np.clip(final_va, 0.0, 1.0)
                final_pred_va = (final_prob_va >= 0.5).astype(int)
                scores.append(balanced_accuracy_score(y_va, final_pred_va))
            else:
                scores.append(root_mean_squared_error(y_va, final_va))

        if len(scores) == 0:
            return 0.0 if task == "classification" else np.inf
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize" if task == "classification" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value, study.trials_dataframe()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def run_workflow(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)

    non_structural_cols = split_comma_separated(args.non_structural_cols)
    df, y, groups, non_structural_cols, fingerprint_cols = load_modeling_data(
        data_path=args.data_path,
        task=args.task,
        target_col=args.target_col,
        group_col=args.group_col,
        non_structural_cols=non_structural_cols,
        doc_col=args.doc_col,
        parsed_doc_col=args.parsed_doc_col,
        fingerprint_prefix=args.fingerprint_prefix,
    )

    print(f"Rows: {len(df)}")
    print(f"Unique groups: {df[args.group_col].nunique()}")
    print(f"Non-structural features: {len(non_structural_cols)}")
    print(f"Fingerprint features: {len(fingerprint_cols)}")
    if args.task == "classification":
        print("Class counts:")
        print(df[args.target_col].value_counts().sort_index())

    outer_splits = make_outer_splits(
        task=args.task,
        y=y,
        groups=groups,
        repeats=args.outer_repeats,
        splits=args.outer_splits,
        seed=args.random_seed,
    )

    model_cols = non_structural_cols + fingerprint_cols
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

        X_train = df.iloc[train_idx].copy()
        X_test = df.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        print("Tuning direct model")
        direct_params, direct_inner_score, direct_trials = tune_direct_model(
            task=args.task,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            model_cols=model_cols,
            inner_splits=args.inner_splits,
            n_trials=args.n_optuna_trials,
            seed=seed,
        )
        direct_trials["repeat"] = repeat
        direct_trials["fold"] = fold
        direct_trials["approach"] = "direct_nonstructural_plus_fingerprints"
        tuning_records.append(direct_trials)

        print("Tuning residualized stage 1")
        stage1_params, stage1_inner_score, stage1_trials = tune_stage1_model(
            task=args.task,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            non_structural_cols=non_structural_cols,
            inner_splits=args.inner_splits,
            n_trials=args.n_optuna_trials,
            seed=seed,
        )
        stage1_trials["repeat"] = repeat
        stage1_trials["fold"] = fold
        stage1_trials["approach"] = "residualized_stage1_nonstructural"
        tuning_records.append(stage1_trials)

        print("Generating cross-fitted residuals")
        stage1_oof_train = make_cross_fitted_stage1_predictions(
            task=args.task,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            non_structural_cols=non_structural_cols,
            stage1_params=stage1_params,
            inner_splits=args.inner_splits,
            seed=seed,
        )
        residual_train = y_train.astype(float) - stage1_oof_train

        print("Tuning residualized stage 2")
        stage2_params, stage2_inner_score, stage2_trials = tune_stage2_residual_model(
            task=args.task,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            non_structural_cols=non_structural_cols,
            fingerprint_cols=fingerprint_cols,
            stage1_params=stage1_params,
            inner_splits=args.inner_splits,
            n_trials=args.n_optuna_trials,
            seed=seed,
        )
        stage2_trials["repeat"] = repeat
        stage2_trials["fold"] = fold
        stage2_trials["approach"] = "residualized_stage2_fingerprint_residual"
        tuning_records.append(stage2_trials)

        if args.task == "classification":
            direct_model = make_classifier(direct_params, seed, "median")
            stage1_model = make_classifier(stage1_params, seed, "median")
        else:
            direct_model = make_regressor(direct_params, seed, "median")
            stage1_model = make_regressor(stage1_params, seed, "median")

        direct_model.fit(X_train[model_cols], y_train)
        stage1_model.fit(X_train[non_structural_cols], y_train)

        stage2_model = make_regressor(stage2_params, seed, "zero")
        stage2_model.fit(X_train[fingerprint_cols], residual_train)

        if args.task == "classification":
            direct_train_prob = direct_model.predict_proba(X_train[model_cols])[:, 1]
            direct_test_prob = direct_model.predict_proba(X_test[model_cols])[:, 1]
            direct_train_pred = (direct_train_prob >= 0.5).astype(int)
            direct_test_pred = (direct_test_prob >= 0.5).astype(int)

            stage1_train = stage1_model.predict_proba(X_train[non_structural_cols])[:, 1]
            stage1_test = stage1_model.predict_proba(X_test[non_structural_cols])[:, 1]
            stage2_train = stage2_model.predict(X_train[fingerprint_cols])
            stage2_test = stage2_model.predict(X_test[fingerprint_cols])

            residualized_train_prob = np.clip(stage1_train + stage2_train, 0.0, 1.0)
            residualized_test_prob = np.clip(stage1_test + stage2_test, 0.0, 1.0)
            residualized_train_pred = (residualized_train_prob >= 0.5).astype(int)
            residualized_test_pred = (residualized_test_prob >= 0.5).astype(int)

            results = {
                "direct_nonstructural_plus_fingerprints": {
                    "train_pred": direct_train_pred,
                    "test_pred": direct_test_pred,
                    "train_prob": direct_train_prob,
                    "test_prob": direct_test_prob,
                    "inner_score": direct_inner_score,
                },
                "residualized_nonstructural_then_fingerprints": {
                    "train_pred": residualized_train_pred,
                    "test_pred": residualized_test_pred,
                    "train_prob": residualized_train_prob,
                    "test_prob": residualized_test_prob,
                    "inner_score": stage2_inner_score,
                },
            }
        else:
            direct_train_pred = direct_model.predict(X_train[model_cols])
            direct_test_pred = direct_model.predict(X_test[model_cols])

            stage1_train = stage1_model.predict(X_train[non_structural_cols])
            stage1_test = stage1_model.predict(X_test[non_structural_cols])
            stage2_train = stage2_model.predict(X_train[fingerprint_cols])
            stage2_test = stage2_model.predict(X_test[fingerprint_cols])

            residualized_train_pred = stage1_train + stage2_train
            residualized_test_pred = stage1_test + stage2_test

            results = {
                "direct_nonstructural_plus_fingerprints": {
                    "train_pred": direct_train_pred,
                    "test_pred": direct_test_pred,
                    "inner_score": direct_inner_score,
                },
                "residualized_nonstructural_then_fingerprints": {
                    "train_pred": residualized_train_pred,
                    "test_pred": residualized_test_pred,
                    "inner_score": stage2_inner_score,
                },
            }

        for approach, result in results.items():
            base_record = {
                "repeat": repeat,
                "fold": fold,
                "cv_iteration": f"{repeat}_{fold}",
                "task": args.task,
                "approach": approach,
                "n_train_rows": len(train_idx),
                "n_test_rows": len(test_idx),
                "n_train_groups": len(np.unique(groups_train)),
                "n_test_groups": len(np.unique(groups_test)),
                "n_nonstructural_features": len(non_structural_cols),
                "n_fingerprint_features": len(fingerprint_cols),
                "inner_score": result["inner_score"],
                "stage1_inner_score": stage1_inner_score if approach.startswith("residualized") else np.nan,
                "stage2_inner_score": stage2_inner_score if approach.startswith("residualized") else np.nan,
            }

            if args.task == "classification":
                train_metrics = get_classification_metrics(y_train, result["train_pred"], result["train_prob"])
                test_metrics = get_classification_metrics(y_test, result["test_pred"], result["test_prob"])
                metric_record = {
                    **base_record,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }

                for local_i, idx in enumerate(test_idx):
                    prob = float(result["test_prob"][local_i])
                    prediction_records.append({
                        "repeat": repeat,
                        "fold": fold,
                        "cv_iteration": f"{repeat}_{fold}",
                        "task": args.task,
                        "approach": approach,
                        "group": df.loc[idx, args.group_col],
                        "observed": int(y_test[local_i]),
                        "predicted": int(result["test_pred"][local_i]),
                        "probability_class_1": prob,
                        "prediction_confidence": max(prob, 1.0 - prob),
                        "correct": int(y_test[local_i] == result["test_pred"][local_i]),
                    })
            else:
                train_metrics = get_regression_metrics(y_train, result["train_pred"])
                test_metrics = get_regression_metrics(y_test, result["test_pred"])
                metric_record = {
                    **base_record,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }

                for local_i, idx in enumerate(test_idx):
                    pred = float(result["test_pred"][local_i])
                    obs = float(y_test[local_i])
                    prediction_records.append({
                        "repeat": repeat,
                        "fold": fold,
                        "cv_iteration": f"{repeat}_{fold}",
                        "task": args.task,
                        "approach": approach,
                        "group": df.loc[idx, args.group_col],
                        "observed": obs,
                        "predicted": pred,
                        "absolute_error": abs(obs - pred),
                        "signed_error": pred - obs,
                    })

            metric_records.append(metric_record)

        best_param_records.append({
            "repeat": repeat,
            "fold": fold,
            "task": args.task,
            "direct_params": str(direct_params),
            "stage1_nonstructural_params": str(stage1_params),
            "stage2_fingerprint_residual_params": str(stage2_params),
            "direct_inner_score": direct_inner_score,
            "stage1_inner_score": stage1_inner_score,
            "stage2_inner_score": stage2_inner_score,
        })

    metrics = pd.DataFrame(metric_records)
    predictions = pd.DataFrame(prediction_records)
    best_params = pd.DataFrame(best_param_records)
    tuning_all = pd.concat(tuning_records, ignore_index=True)

    metrics.to_csv(os.path.join(args.output_dir, f"{args.task}_metrics.tsv"), sep="\t", index=False)
    predictions.to_csv(os.path.join(args.output_dir, f"{args.task}_predictions.tsv"), sep="\t", index=False)
    best_params.to_csv(os.path.join(args.output_dir, f"{args.task}_best_params.tsv"), sep="\t", index=False)
    tuning_all.to_csv(os.path.join(args.output_dir, f"{args.task}_optuna_trials.tsv"), sep="\t", index=False)

    summary = summarize_metrics(metrics, args.task)
    summary.to_csv(os.path.join(args.output_dir, f"{args.task}_summary.tsv"), sep="\t", index=False)

    print("\nSummary:")
    print(summary)
    print(f"\nDone. Outputs saved to: {args.output_dir}")


def summarize_metrics(metrics: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "classification":
        return metrics.groupby("approach", as_index=False).agg(
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

    return metrics.groupby("approach", as_index=False).agg(
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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run direct and residualized fingerprint models.")
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="residualized_model_outputs")
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--group-col", required=True)
    parser.add_argument("--non-structural-cols", required=True, help="Comma-separated column names.")
    parser.add_argument("--doc-col", default=None, help="Optional raw DOC column to parse into DOC_mean.")
    parser.add_argument("--parsed-doc-col", default="DOC_mean")
    parser.add_argument("--fingerprint-prefix", default="absoluteIndex_")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--n-optuna-trials", type=int, default=75)
    return parser


if __name__ == "__main__":
    run_workflow(build_parser().parse_args())
