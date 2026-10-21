"""Comparación de k en validación cruzada estratificada."""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score
from proyecto_utils import TARGET, CLASS_ORDER, ensure_dir, read_dataset, write_json
from src.paths import first_existing


def get_stratified_cv(k=5):
    """Devuelve folds estratificados y barajados con semilla fija."""
    if k not in (3, 5, 10):
        raise ValueError("k debe ser 3, 5 o 10")
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


def run(input_path="train.csv", report_dir="reportes", ks=(3, 5, 10), fast=True, n_jobs=-1):
    if not Path(input_path).exists():
        input_path = first_existing("train.csv")
    if report_dir == "reportes":
        report_dir = "resultados"
    df = read_dataset(input_path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    rows, details = [], {}
    for k in ks:
        cv = get_stratified_cv(k)
        sizes = []
        for _, val_idx in cv.split(X, y):
            sizes.append(int((y.iloc[val_idx] == "Alto_valor").sum()))
        start = time.perf_counter()
        # Se evalúa el pipeline MLP real para que la selección de k represente
        # el estimador que se entregará, no únicamente un clasificador de referencia.
        from entrenamiento import build_grid
        pipe, grid = build_grid(fast=fast)
        if isinstance(grid, list):
            grid = grid[0]
        params = {key: values[0] for key, values in grid.items()}
        pipe.set_params(**params)
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring={"f1_macro": make_scorer(f1_score, average="macro"),
                     "balanced_accuracy": make_scorer(balanced_accuracy_score)},
            n_jobs=n_jobs, return_train_score=False,
        )
        elapsed = time.perf_counter() - start
        rows.append({"k": k, "minority_min_validation": min(sizes), "minority_mean_validation": np.mean(sizes),
                     "mlp_f1_macro_mean": scores["test_f1_macro"].mean(),
                     "mlp_f1_macro_std": scores["test_f1_macro"].std(ddof=1),
                     "mlp_balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
                     "mlp_balanced_accuracy_std": scores["test_balanced_accuracy"].std(ddof=1),
                     "standard_error": scores["test_f1_macro"].std(ddof=1) / np.sqrt(k),
                     "elapsed_seconds": elapsed})
        details[str(k)] = sizes
    report = pd.DataFrame(rows)
    # k=5 es el compromiso documentado: suficiente estabilidad sin duplicar
    # el costo de k=10; el reporte conserva la comparación empírica completa.
    chosen = 5 if 5 in report["k"].values else int(report.iloc[0]["k"])
    ensure_dir(report_dir)
    report.to_csv(Path(report_dir) / "reporte_crossval.csv", index=False)
    write_json({"chosen_k": chosen, "fold_minority_sizes": details}, Path(report_dir) / "reporte_crossval.json")
    print(f"k recomendado: {chosen}; tamaños mínimos Alto_valor: {details[str(chosen)]}")
    return chosen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(first_existing("train.csv")))
    parser.add_argument("--report-dir", default="resultados")
    parser.add_argument("--full", action="store_true", help="usa la configuración no rápida del MLP")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    run(args.input, args.report_dir, fast=not args.full, n_jobs=args.n_jobs)
