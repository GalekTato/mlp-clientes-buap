"""Evaluación por chunks para el archivo oficial o el dataset local disponible."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from proyecto_utils import TARGET, CLASS_ORDER, NUMERIC_COLUMNS

def evaluate_in_chunks(model, path, chunksize=50000, target=TARGET):
    started, y_true, y_pred, y_prob, rows, skipped = time.perf_counter(), [], [], [], 0, 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        y = chunk.pop(target)
        for column in NUMERIC_COLUMNS:
            if column in chunk:
                chunk[column] = pd.to_numeric(
                    chunk[column].astype("string").str.replace(r"[$,]|MXN", "", regex=True),
                    errors="coerce",
                )
        valid = ~(chunk.isna().any(axis=1) | y.isna())
        skipped += int((~valid).sum())
        chunk, y = chunk.loc[valid], y.loc[valid]
        if chunk.empty:
            continue
        pred = model.predict(chunk)
        prob = model.predict_proba(chunk)
        y_true.append(y.to_numpy()); y_pred.append(pred); y_prob.append(prob); rows += len(y)
    y_true, y_pred, y_prob = np.concatenate(y_true), np.concatenate(y_pred), np.vstack(y_prob)
    elapsed = time.perf_counter() - started
    return {
        "path": str(path), "rows": rows, "skipped_rows": skipped, "elapsed_seconds": elapsed,
        "throughput_rows_per_second": rows / elapsed if elapsed else 0,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc_ovr": roc_auc_score(y_true, y_prob, multi_class="ovr", labels=model.classes_),
        "log_loss": log_loss(y_true, y_prob, labels=model.classes_),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
    }
