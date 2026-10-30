"""Selección empírica de k en validación cruzada estratificada."""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score
from proyecto_utils import (
    SEED,
    TARGET,
    build_base_pipeline,
    ensure_dir,
    read_dataset,
    write_json,
)
from src.paths import first_existing


def get_stratified_cv(k: int = 5) -> StratifiedKFold:
    """Devuelve un particionador StratifiedKFold barajado con semilla fija."""
    if k not in (3, 5, 10):
        raise ValueError("k debe ser 3, 5 o 10")
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)


def select_best_k(report: pd.DataFrame, min_minority_cases: int = 500) -> int:
    """Selecciona el mayor k que conserve al menos min_minority_cases de Alto_valor por fold."""
    candidatos_validos = report[report["minority_min_validation"] >= min_minority_cases]
    if not candidatos_validos.empty:
        ordenados = candidatos_validos.sort_values(
            by=["k", "mlp_f1_macro_std"],
            ascending=[False, True],
        )
        return int(ordenados.iloc[0]["k"])
    respaldo = report.sort_values(
        by=["minority_min_validation", "mlp_f1_macro_std"],
        ascending=[False, True],
    )
    return int(respaldo.iloc[0]["k"])


def run(input_path="train.csv", report_dir="reportes", ks=(3, 5, 10), fast=True, n_jobs=-1):
    if not Path(input_path).exists():
        input_path = first_existing("train.csv")
    if report_dir == "reportes":
        report_dir = "resultados"
    df = read_dataset(input_path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    filas_resumen, conteos_por_fold = [], {}
    max_iter = 80 if fast else 150

    for k in ks:
        cv = get_stratified_cv(k)
        conteos_minoritaria = [
            int((y.iloc[indices_val] == "Alto_valor").sum())
            for _, indices_val in cv.split(X, y)
        ]
        inicio = time.perf_counter()
        pipeline_base = build_base_pipeline(max_iter=max_iter)
        puntajes = cross_validate(
            pipeline_base,
            X,
            y,
            cv=cv,
            scoring={
                "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
                "balanced_accuracy": make_scorer(balanced_accuracy_score),
            },
            n_jobs=n_jobs,
            return_train_score=False,
        )
        segundos = time.perf_counter() - inicio
        filas_resumen.append({
            "k": k,
            "minority_min_validation": min(conteos_minoritaria),
            "minority_mean_validation": float(np.mean(conteos_minoritaria)),
            "mlp_f1_macro_mean": float(puntajes["test_f1_macro"].mean()),
            "mlp_f1_macro_std": float(puntajes["test_f1_macro"].std(ddof=1)),
            "mlp_balanced_accuracy_mean": float(puntajes["test_balanced_accuracy"].mean()),
            "mlp_balanced_accuracy_std": float(puntajes["test_balanced_accuracy"].std(ddof=1)),
            "standard_error": float(puntajes["test_f1_macro"].std(ddof=1) / np.sqrt(k)),
            "elapsed_seconds": segundos,
        })
        conteos_por_fold[str(k)] = conteos_minoritaria

    tabla_cv = pd.DataFrame(filas_resumen)
    umbral_minoritario = 500
    k_elegido = select_best_k(tabla_cv, min_minority_cases=umbral_minoritario)
    ensure_dir(report_dir)
    tabla_cv.to_csv(Path(report_dir) / "reporte_crossval.csv", index=False)
    write_json(
        {
            "chosen_k": k_elegido,
            "selection_rule": "max k in {3, 5, 10} with minority_min_validation (Alto_valor) >= 500; tie-breaker: min mlp_f1_macro_std",
            "min_minority_threshold": umbral_minoritario,
            "fold_minority_sizes": conteos_por_fold,
            "summary": tabla_cv.to_dict(orient="records"),
        },
        Path(report_dir) / "reporte_crossval.json",
    )
    print(f"k recomendado: {k_elegido}; tamaños mínimos Alto_valor: {conteos_por_fold[str(k_elegido)]}")
    return k_elegido


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(first_existing("train.csv")))
    parser.add_argument("--report-dir", default="resultados")
    parser.add_argument("--full", action="store_true", help="usa 150 épocas máximas en lugar de 80")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    run(args.input, args.report_dir, fast=not args.full, n_jobs=args.n_jobs)
