"""Orquestador de las etapas del proyecto."""
from __future__ import annotations
import argparse
from pathlib import Path
import preprocesameinto
import muestreo
import crossVal
import entrenamiento
import json
import joblib
from src.paths import DATA, RESULTS, FIGURES, first_existing, default_dataset, ensure_layout
from src.evaluation_large import evaluate_in_chunks


def main():
    parser = argparse.ArgumentParser(description="Pipeline de clasificación multiclase de clientes")
    parser.add_argument("--input", default=str(default_dataset()))
    parser.add_argument("--fast", action="store_true", help="usa una rejilla mínima para validar el flujo")
    parser.add_argument("--balance", choices=["baseline", "weighted"], default="weighted")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--large-validation", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    args = parser.parse_args()
    ensure_layout()
    if not args.skip_clean or not (DATA / "dataset_limpio.csv").exists():
        preprocesameinto.run(args.input)
    muestreo.run()
    k = crossVal.run(fast=args.fast)
    entrenamiento.run(fast=args.fast, k=k, balance=args.balance, calibrate_models=args.calibrate)
    if args.large_validation:
        source = default_dataset()
        model = joblib.load("modelos/mejor_modelo_mlp.joblib")
        result = evaluate_in_chunks(model, source)
        result["execution_status"] = "executed_full_1m" if result["rows"] >= 1_000_000 else "executed_local_dataset_not_full_1m"
        payload = json.dumps(result, indent=2)
        (RESULTS / "validacion_masiva.json").write_text(payload, encoding="utf-8")
        (RESULTS / "validacion_masiva_v2.json").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
