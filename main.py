"""Orquesta el flujo completo de preprocesamiento, muestreo, validación cruzada, entrenamiento y evaluación."""
from __future__ import annotations
import argparse
import json
import joblib
import preprocesamiento
import muestreo
import crossVal
import entrenamiento
from src.paths import DATA, RESULTS, default_dataset, ensure_layout
from src.evaluation_large import evaluate_in_chunks


def main():
    parser = argparse.ArgumentParser(description="Pipeline de clasificación multiclase de clientes con MLP")
    parser.add_argument("--input", default=str(default_dataset()), help="Ruta del archivo CSV de entrada")
    parser.add_argument("--fast", action="store_true", help="Ejecuta una rejilla abreviada para pruebas rápidas")
    parser.add_argument("--balance", choices=["baseline", "weighted"], default="weighted")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--large-validation", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    ensure_layout()
    if not args.skip_clean or not (DATA / "dataset_limpio.csv").exists():
        preprocesamiento.run(args.input)

    muestreo.run()
    k = crossVal.run(fast=args.fast, n_jobs=args.n_jobs)
    entrenamiento.run(
        fast=args.fast,
        k=k,
        balance=args.balance,
        gamma=args.gamma,
        calibrate_models=args.calibrate,
        n_jobs=args.n_jobs,
    )

    if args.large_validation:
        source = default_dataset()
        model = joblib.load("modelos/mejor_modelo_mlp.joblib")
        result = evaluate_in_chunks(model, source)
        raw_rows = result["rows"] + result.get("skipped_rows", 0)
        result["execution_status"] = (
            "executed_full_dataset" if raw_rows >= 1_000_000
            else "executed_local_dataset"
        )
        payload = json.dumps(result, indent=2)
        (RESULTS / "validacion_masiva.json").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
