"""Evaluación por lotes (chunks) sobre el archivo completo de clientes sin fuga de entrenamiento."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from proyecto_utils import (
    TARGET,
    CLASS_ORDER,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    clean_numeric_text,
)
from preprocesamiento import clean_dataframe
from src.paths import DATA


def evaluate_in_chunks(model, path: str | Path, chunksize: int = 50000, target: str = TARGET, train_path: str | Path | None = None) -> dict:
    """Evalúa el clasificador entrenado procesando el archivo CSV por bloques en memoria acotada.

    Cada bloque pasa por la limpieza determinista (`clean_dataframe`) y excluye los
    identificadores presentes en `train.csv` para garantizar una medición honesta fuera de muestra.
    """
    started = time.perf_counter()
    y_true_batches, y_pred_batches, y_prob_batches = [], [], []
    rows_evaluated = 0
    removed_by_cleaning = 0
    excluded_train_ids = 0

    if train_path is None:
        default_train = DATA / "train.csv"
        train_path = default_train if default_train.exists() else None
    train_ids = set()
    if train_path is not None and Path(train_path).exists():
        train_df = pd.read_csv(train_path, usecols=["id_cliente"])
        train_ids = set(
            pd.to_numeric(train_df["id_cliente"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

    dtype = {col: "string" for col in CATEGORICAL_COLUMNS + [target]}
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=dtype, low_memory=False):
        n_raw = len(chunk)
        for column in NUMERIC_COLUMNS:
            if column in chunk:
                chunk[column] = pd.to_numeric(clean_numeric_text(chunk[column]), errors="coerce")

        cleaned_chunk = clean_dataframe(chunk)
        removed_by_cleaning += int(n_raw - len(cleaned_chunk))
        if cleaned_chunk.empty:
            continue

        if train_ids and "id_cliente" in cleaned_chunk:
            in_train = cleaned_chunk["id_cliente"].astype(int).isin(train_ids)
            excluded_train_ids += int(in_train.sum())
            cleaned_chunk = cleaned_chunk.loc[~in_train].copy()
        if cleaned_chunk.empty:
            continue

        y_batch = cleaned_chunk.pop(target)
        prob_batch = model.predict_proba(cleaned_chunk)
        classes = np.asarray(model.classes_)
        pred_batch = classes[np.argmax(prob_batch, axis=1)]

        y_true_batches.append(y_batch.to_numpy())
        y_pred_batches.append(pred_batch)
        y_prob_batches.append(prob_batch)
        rows_evaluated += len(y_batch)

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)
    y_prob = np.vstack(y_prob_batches)
    elapsed = time.perf_counter() - started
    return {
        "path": str(path),
        "rows": rows_evaluated,
        "rows_evaluated": rows_evaluated,
        "removed_by_cleaning": removed_by_cleaning,
        "excluded_train_ids": excluded_train_ids,
        "skipped_rows": removed_by_cleaning + excluded_train_ids,
        "elapsed_seconds": elapsed,
        "throughput_rows_per_second": rows_evaluated / elapsed if elapsed else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc_ovr": float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=model.classes_)),
        "log_loss": float(log_loss(y_true, y_prob, labels=model.classes_)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
    }
