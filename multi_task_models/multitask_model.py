#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    classification_path: Path = Path("data/RE_classification_set.tsv")
    regression_path: Path = Path("data/BV10_regression_set.tsv")
    output_dir: Path = Path("results/multitask_model")
    random_state: int = 88
    n_splits: int = 5
    batch_size: int = 16
    epochs: int = 800
    patience: int = 150
    learning_rate: float = 2e-4
    weight_decay: float = 5e-3
    class_loss_weight: float = 1.0
    reg_loss_weight: float = 0.5
    checkpoint_smoothing_alpha: float = 0.25
    top_k_class: int = 50
    top_k_reg: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_smiles(value: object) -> object:
    if pd.isna(value):
        return np.nan
    return str(value).strip()


def parse_numeric_value(value: object) -> float:
    if pd.isna(value):
        return np.nan

    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    if not numbers:
        return np.nan

    values = [float(number) for number in numbers]
    return float(np.mean(values))


def find_first_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Could not find required {label} column. Tried: {candidates}")


def find_fingerprint_columns(df_class: pd.DataFrame, df_reg: pd.DataFrame) -> list[str]:
    prefixes = ("absoluteIndex_", "fingerprint_", "fp_")
    class_columns = [column for column in df_class.columns if column.startswith(prefixes)]
    reg_columns = [column for column in df_reg.columns if column.startswith(prefixes)]
    columns = sorted(set(class_columns) | set(reg_columns))

    if not columns:
        raise ValueError("No fingerprint columns were found.")

    return columns


def copy_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_index = int(np.argmax(tpr - fpr))
    return float(thresholds[best_index])


class TaskCheckpoint:
    def __init__(self, task_names: list[str], smoothing_alpha: float) -> None:
        self.smoothing_alpha = smoothing_alpha
        self.best_losses = {task: np.inf for task in task_names}
        self.smoothed_losses = {task: None for task in task_names}
        self.best_states = {task: None for task in task_names}

    def update(self, model: nn.Module, task_losses: dict[str, float]) -> None:
        for task, loss in task_losses.items():
            if loss is None or np.isnan(loss):
                continue

            if self.smoothed_losses[task] is None:
                self.smoothed_losses[task] = loss
            else:
                self.smoothed_losses[task] = (
                    self.smoothing_alpha * loss
                    + (1.0 - self.smoothing_alpha) * self.smoothed_losses[task]
                )

            if self.smoothed_losses[task] < self.best_losses[task]:
                self.best_losses[task] = self.smoothed_losses[task]
                self.best_states[task] = copy_state_dict(model)

    def get_best_state(self, task: str) -> dict[str, torch.Tensor] | None:
        return self.best_states[task]


def build_multitask_dataset(config: Config):
    df_class = pd.read_csv(config.classification_path, sep="\t", low_memory=False)
    df_reg = pd.read_csv(config.regression_path, sep="\t", low_memory=False)

    smiles_class = find_first_column(df_class, ["SMILES", "smiles"], "SMILES")
    smiles_reg = find_first_column(df_reg, ["SMILES", "smiles"], "SMILES")
    doc_class = find_first_column(df_class, ["DOC", "DOC (mg/L)", "doc", "doc_mg_l"], "DOC")
    doc_reg = find_first_column(df_reg, ["DOC", "DOC (mg/L)", "doc", "doc_mg_l"], "DOC")
    class_target = find_first_column(
        df_class,
        ["binary_class", "class", "RE_class", "removal_class"],
        "classification target",
    )
    reg_target = find_first_column(df_reg, ["logBV10", "log_bv10"], "regression target")

    df_class = df_class.copy()
    df_reg = df_reg.copy()
    df_class["merge_smiles"] = df_class[smiles_class].apply(normalize_smiles)
    df_reg["merge_smiles"] = df_reg[smiles_reg].apply(normalize_smiles)
    df_class["DOC_mean"] = df_class[doc_class].apply(parse_numeric_value)
    df_reg["DOC_mean"] = df_reg[doc_reg].apply(parse_numeric_value)

    fingerprint_cols = find_fingerprint_columns(df_class, df_reg)
    feature_cols = ["DOC_mean"] + fingerprint_cols

    class_rows = df_class[
        df_class["DOC_mean"].notna()
        & df_class[class_target].notna()
        & df_class["merge_smiles"].notna()
    ].copy()

    reg_rows = df_reg[
        df_reg["DOC_mean"].notna()
        & df_reg[reg_target].notna()
        & df_reg["merge_smiles"].notna()
    ].copy()

    for column in fingerprint_cols:
        if column not in class_rows.columns:
            class_rows[column] = np.nan
        if column not in reg_rows.columns:
            reg_rows[column] = np.nan

    class_rows["binary_class"] = pd.to_numeric(class_rows[class_target], errors="coerce")
    class_rows["logBV10"] = np.nan
    class_rows["source_dataset"] = "classification"

    reg_rows["binary_class"] = np.nan
    reg_rows["logBV10"] = pd.to_numeric(reg_rows[reg_target], errors="coerce")
    reg_rows["source_dataset"] = "regression"

    keep_cols = ["merge_smiles", "source_dataset", "binary_class", "logBV10"] + feature_cols
    df = pd.concat([class_rows[keep_cols], reg_rows[keep_cols]], ignore_index=True)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_class = df["binary_class"].astype(float).to_numpy()
    y_reg = df["logBV10"].astype(float).to_numpy()
    mask_class = df["binary_class"].notna().astype(float).to_numpy()
    mask_reg = df["logBV10"].notna().astype(float).to_numpy()
    groups = df["merge_smiles"].astype(str).to_numpy()

    return df, X, y_class, y_reg, mask_class, mask_reg, groups


class MaskedMultiTaskDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y_class: np.ndarray,
        y_reg: np.ndarray,
        mask_class: np.ndarray,
        mask_reg: np.ndarray,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(np.nan_to_num(y_class, nan=0.0), dtype=torch.float32)
        self.y_reg = torch.tensor(np.nan_to_num(y_reg, nan=0.0), dtype=torch.float32)
        self.mask_class = torch.tensor(mask_class, dtype=torch.float32)
        self.mask_reg = torch.tensor(mask_reg, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return (
            self.X[index],
            self.y_class[index],
            self.y_reg[index],
            self.mask_class[index],
            self.mask_reg[index],
        )


class MultiTaskGACNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(n_features, 24),
            nn.LayerNorm(24),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        self.class_head = nn.Sequential(
            nn.Linear(24, 24),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(24, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(24, 16),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(16, 1),
        )

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(X)
        class_logit = self.class_head(shared).squeeze(1)
        reg_pred = self.reg_head(shared).squeeze(1)
        return class_logit, reg_pred


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    return (loss * mask).sum() / mask.sum()


def masked_huber_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)

    loss = nn.functional.smooth_l1_loss(
        preds,
        targets,
        reduction="none",
        beta=0.5,
    )
    return (loss * mask).sum() / mask.sum()


def select_features_inside_fold(
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    y_class_train: np.ndarray,
    y_reg_train: np.ndarray,
    mask_class_train: np.ndarray,
    mask_reg_train: np.ndarray,
    top_k_class: int,
    top_k_reg: int,
    random_state: int,
):
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    feature_names = np.array(X_train_raw.columns)
    selected = {"DOC_mean"}

    class_available = mask_class_train == 1
    if class_available.sum() >= 5 and len(np.unique(y_class_train[class_available])) == 2:
        mi_class = mutual_info_classif(
            X_train_imp[class_available],
            y_class_train[class_available].astype(int),
            random_state=random_state,
        )
        selected.update(feature_names[np.argsort(mi_class)[::-1][:top_k_class]])

    reg_available = mask_reg_train == 1
    if reg_available.sum() >= 5:
        mi_reg = mutual_info_regression(
            X_train_imp[reg_available],
            y_reg_train[reg_available],
            random_state=random_state,
        )
        selected.update(feature_names[np.argsort(mi_reg)[::-1][:top_k_reg]])

    selected = sorted(selected)
    return X_train_raw[selected], X_val_raw[selected], selected


def train_one_fold(
    config: Config,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_class_train: np.ndarray,
    y_class_val: np.ndarray,
    y_reg_train: np.ndarray,
    y_reg_val: np.ndarray,
    mask_class_train: np.ndarray,
    mask_class_val: np.ndarray,
    mask_reg_train: np.ndarray,
    mask_reg_val: np.ndarray,
):
    train_dataset = MaskedMultiTaskDataset(
        X_train,
        y_class_train,
        y_reg_train,
        mask_class_train,
        mask_reg_train,
    )
    val_dataset = MaskedMultiTaskDataset(
        X_val,
        y_class_val,
        y_reg_val,
        mask_class_val,
        mask_reg_val,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = MultiTaskGACNet(n_features=X_train.shape[1]).to(config.device)

    available_class = mask_class_train == 1
    n_pos = np.sum(y_class_train[available_class] == 1)
    n_neg = np.sum(y_class_train[available_class] == 0)

    pos_weight = None
    if n_pos > 0:
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=30,
    )

    checkpoint = TaskCheckpoint(
        task_names=["binary_class", "logBV10"],
        smoothing_alpha=config.checkpoint_smoothing_alpha,
    )

    best_total_loss = np.inf
    best_threshold = 0.5
    patience_counter = 0

    for _ in range(config.epochs):
        model.train()

        for xb, ycb, yrb, mcb, mrb in train_loader:
            xb = xb.to(config.device)
            ycb = ycb.to(config.device)
            yrb = yrb.to(config.device)
            mcb = mcb.to(config.device)
            mrb = mrb.to(config.device)

            optimizer.zero_grad()
            class_logit, reg_pred = model(xb)

            loss_class = masked_bce_loss(class_logit, ycb, mcb, pos_weight)
            loss_reg = masked_huber_loss(reg_pred, yrb, mrb)
            loss = config.class_loss_weight * loss_class + config.reg_loss_weight * loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_class_losses = []
        val_reg_losses = []
        val_total_losses = []
        val_class_true = []
        val_class_prob = []

        with torch.no_grad():
            for xb, ycb, yrb, mcb, mrb in val_loader:
                xb = xb.to(config.device)
                ycb = ycb.to(config.device)
                yrb = yrb.to(config.device)
                mcb = mcb.to(config.device)
                mrb = mrb.to(config.device)

                class_logit, reg_pred = model(xb)
                loss_class = masked_bce_loss(class_logit, ycb, mcb, pos_weight)
                loss_reg = masked_huber_loss(reg_pred, yrb, mrb)
                loss_total = config.class_loss_weight * loss_class + config.reg_loss_weight * loss_reg

                val_class_losses.append(loss_class.item())
                val_reg_losses.append(loss_reg.item())
                val_total_losses.append(loss_total.item())

                prob = torch.sigmoid(class_logit).detach().cpu().numpy()
                mask = mcb.detach().cpu().numpy() == 1
                if mask.sum() > 0:
                    val_class_true.extend(ycb.detach().cpu().numpy()[mask])
                    val_class_prob.extend(prob[mask])

        val_class_loss = float(np.mean(val_class_losses))
        val_reg_loss = float(np.mean(val_reg_losses))
        val_total_loss = float(np.mean(val_total_losses))

        scheduler.step(val_total_loss)
        checkpoint.update(
            model,
            {
                "binary_class": val_class_loss,
                "logBV10": val_reg_loss,
            },
        )

        if val_total_loss < best_total_loss:
            best_total_loss = val_total_loss
            if len(np.unique(val_class_true)) == 2:
                best_threshold = find_best_threshold(
                    np.array(val_class_true).astype(int),
                    np.array(val_class_prob),
                )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    return model, checkpoint.get_best_state("binary_class"), checkpoint.get_best_state("logBV10"), best_threshold


def evaluate_cv(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_everything(config.random_state)

    df, X, y_class, y_reg, mask_class, mask_reg, groups = build_multitask_dataset(config)

    split_y = np.nan_to_num(y_class, nan=0).astype(int)
    cv = StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    fold_metrics = []
    predictions = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, split_y, groups), start=1):
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]

        y_class_train = y_class[train_idx]
        y_class_val = y_class[val_idx]
        y_reg_train = y_reg[train_idx]
        y_reg_val = y_reg[val_idx]
        mask_class_train = mask_class[train_idx]
        mask_class_val = mask_class[val_idx]
        mask_reg_train = mask_reg[train_idx]
        mask_reg_val = mask_reg[val_idx]

        X_train_sel, X_val_sel, selected_features = select_features_inside_fold(
            X_train_raw,
            X_val_raw,
            y_class_train,
            y_reg_train,
            mask_class_train,
            mask_reg_train,
            config.top_k_class,
            config.top_k_reg,
            config.random_state,
        )

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train_sel))
        X_val_processed = scaler.transform(imputer.transform(X_val_sel))

        reg_train_available = mask_reg_train == 1
        y_reg_mean = np.nanmean(y_reg_train[reg_train_available])
        y_reg_std = np.nanstd(y_reg_train[reg_train_available])

        if y_reg_std == 0 or np.isnan(y_reg_std):
            y_reg_std = 1.0

        y_reg_train_scaled = y_reg_train.copy()
        y_reg_val_scaled = y_reg_val.copy()
        y_reg_train_scaled[reg_train_available] = (
            y_reg_train[reg_train_available] - y_reg_mean
        ) / y_reg_std

        reg_val_available = mask_reg_val == 1
        y_reg_val_scaled[reg_val_available] = (
            y_reg_val[reg_val_available] - y_reg_mean
        ) / y_reg_std

        model, class_state, reg_state, threshold = train_one_fold(
            config,
            X_train_processed,
            X_val_processed,
            y_class_train,
            y_class_val,
            y_reg_train_scaled,
            y_reg_val_scaled,
            mask_class_train,
            mask_class_val,
            mask_reg_train,
            mask_reg_val,
        )

        class_true = np.array([])
        class_prob = np.array([])
        class_pred = np.array([])
        reg_true = np.array([])
        reg_pred = np.array([])

        if mask_class_val.sum() > 0 and class_state is not None:
            model.load_state_dict(class_state)
            model.to(config.device)
            model.eval()
            with torch.no_grad():
                xb = torch.tensor(X_val_processed, dtype=torch.float32).to(config.device)
                class_logit, _ = model(xb)
                all_class_prob = torch.sigmoid(class_logit).cpu().numpy()

            class_idx = mask_class_val == 1
            class_true = y_class_val[class_idx].astype(int)
            class_prob = all_class_prob[class_idx]
            class_pred = (class_prob >= threshold).astype(int)

        if mask_reg_val.sum() > 0 and reg_state is not None:
            model.load_state_dict(reg_state)
            model.to(config.device)
            model.eval()
            with torch.no_grad():
                xb = torch.tensor(X_val_processed, dtype=torch.float32).to(config.device)
                _, all_reg_pred_scaled = model(xb)
                all_reg_pred = all_reg_pred_scaled.cpu().numpy() * y_reg_std + y_reg_mean

            reg_idx = mask_reg_val == 1
            reg_true = y_reg_val[reg_idx]
            reg_pred = all_reg_pred[reg_idx]

        metrics = {
            "fold": fold,
            "n_selected_features": len(selected_features),
            "decision_threshold": threshold,
            "classification_n": len(class_true),
            "regression_n": len(reg_true),
        }

        if len(class_true) > 0:
            metrics.update(
                {
                    "accuracy": accuracy_score(class_true, class_pred),
                    "balanced_accuracy": balanced_accuracy_score(class_true, class_pred),
                },
            )
            if len(np.unique(class_true)) == 2:
                metrics["roc_auc"] = roc_auc_score(class_true, class_prob)
                metrics["pr_auc"] = average_precision_score(class_true, class_prob)

        if len(reg_true) > 0:
            metrics.update(
                {
                    "mae": mean_absolute_error(reg_true, reg_pred),
                    "rmse": np.sqrt(mean_squared_error(reg_true, reg_pred)),
                    "r2": r2_score(reg_true, reg_pred),
                },
            )

        fold_metrics.append(metrics)

        val_rows = df.iloc[val_idx][["merge_smiles", "source_dataset", "binary_class", "logBV10"]].copy()
        val_rows["fold"] = fold
        val_rows["predicted_class_probability"] = np.nan
        val_rows["predicted_class"] = np.nan
        val_rows["predicted_logBV10"] = np.nan

        if mask_class_val.sum() > 0 and class_state is not None:
            val_rows.loc[mask_class_val == 1, "predicted_class_probability"] = class_prob
            val_rows.loc[mask_class_val == 1, "predicted_class"] = class_pred

        if mask_reg_val.sum() > 0 and reg_state is not None:
            val_rows.loc[mask_reg_val == 1, "predicted_logBV10"] = reg_pred

        predictions.append(val_rows)

        print(f"Finished fold {fold} with {len(selected_features)} selected features.")

    metrics_df = pd.DataFrame(fold_metrics)
    predictions_df = pd.concat(predictions, ignore_index=True)

    return metrics_df, predictions_df


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    numeric = metrics_df.select_dtypes(include=[np.number])
    summary = pd.DataFrame(
        {
            "mean": numeric.mean(numeric_only=True),
            "std": numeric.std(numeric_only=True),
        },
    )
    return summary.reset_index(names="metric")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate a masked multi-task neural network for RE classification and logBV10 regression.",
    )
    parser.add_argument("--classification-path", type=Path, default=Path("data/RE_classification_set.tsv"))
    parser.add_argument("--regression-path", type=Path, default=Path("data/BV10_regression_set.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/multitask_model"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--top-k-class", type=int, default=50)
    parser.add_argument("--top-k-reg", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    return Config(
        classification_path=args.classification_path,
        regression_path=args.regression_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_splits=args.n_splits,
        top_k_class=args.top_k_class,
        top_k_reg=args.top_k_reg,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df, predictions_df = evaluate_cv(config)
    summary_df = summarize_metrics(metrics_df)

    metrics_df.to_csv(config.output_dir / "fold_metrics.tsv", sep="\t", index=False)
    predictions_df.to_csv(config.output_dir / "cross_validated_predictions.tsv", sep="\t", index=False)
    summary_df.to_csv(config.output_dir / "metric_summary.tsv", sep="\t", index=False)

    print("\nMetric summary")
    print(summary_df.to_string(index=False))
    print(f"\nResults written to: {config.output_dir}")


if __name__ == "__main__":
    main()
