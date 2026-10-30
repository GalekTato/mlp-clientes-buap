"""Modelos de referencia y curva de tamaño de muestra con el mismo preprocesamiento y particiones."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from proyecto_utils import (
    SEED,
    TARGET,
    ensure_dir,
    evaluate_predictions,
    make_preprocessor,
    read_dataset,
)
from src.feature_engineering import DomainFeatureTransformer
from src.paths import first_existing


def build_reference_models() -> dict[str, Pipeline]:
    return {
        "Dummy_most_frequent": Pipeline([
            ("features", DomainFeatureTransformer()),
            ("preprocessor", make_preprocessor("standard")),
            ("model", DummyClassifier(strategy="most_frequent", random_state=SEED)),
        ]),
        "Dummy_stratified": Pipeline([
            ("features", DomainFeatureTransformer()),
            ("preprocessor", make_preprocessor("standard")),
            ("model", DummyClassifier(strategy="stratified", random_state=SEED)),
        ]),
        "LogisticRegression": Pipeline([
            ("features", DomainFeatureTransformer()),
            ("preprocessor", make_preprocessor("standard")),
            ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("features", DomainFeatureTransformer()),
            ("preprocessor", make_preprocessor("standard", sparse_output=False)),
            ("model", HistGradientBoostingClassifier(random_state=SEED)),
        ]),
    }


def evaluate_model(model, df: pd.DataFrame, model_name: str, split: str) -> dict:
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    pred = model.predict(X)
    proba = model.predict_proba(X)
    resumen = evaluate_predictions(y, pred, proba, model.classes_, split)
    return {"model": model_name, **resumen}


def run(train_path="train.csv", validation_path="validation.csv", test_path="test.csv", report_dir="resultados"):
    train_path = first_existing(Path(train_path).name) if not Path(train_path).exists() else Path(train_path)
    validation_path = first_existing(Path(validation_path).name) if not Path(validation_path).exists() else Path(validation_path)
    test_path = first_existing(Path(test_path).name) if not Path(test_path).exists() else Path(test_path)
    train, validation, test = map(read_dataset, [train_path, validation_path, test_path])
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]

    report_dir = ensure_dir(report_dir)
    filas_referencia = []
    for nombre_modelo, pipe in build_reference_models().items():
        inicio = time.perf_counter()
        pipe.fit(X_train, y_train)
        segundos_fit = time.perf_counter() - inicio
        for nombre_particion, df_particion in [("train", train), ("validation", validation), ("test", test)]:
            fila = evaluate_model(pipe, df_particion, nombre_modelo, nombre_particion)
            fila["fit_seconds"] = segundos_fit
            filas_referencia.append(fila)

    df_referencia = pd.DataFrame(filas_referencia)
    df_referencia.to_csv(report_dir / "modelos_referencia.csv", index=False)

    filas_curva = []
    for fraccion in (0.10, 0.25, 0.50, 1.00):
        if fraccion < 1.0:
            sub_train, _ = train_test_split(train, train_size=fraccion, stratify=train[TARGET], random_state=SEED)
        else:
            sub_train = train
        hgb = build_reference_models()["HistGradientBoosting"]
        inicio = time.perf_counter()
        hgb.fit(sub_train.drop(columns=[TARGET]), sub_train[TARGET])
        segundos_fit = time.perf_counter() - inicio
        for nombre_particion, df_particion in [("train_sub", sub_train), ("validation", validation), ("test", test)]:
            fila = evaluate_model(hgb, df_particion, "HistGradientBoosting", nombre_particion)
            fila["train_fraction"] = fraccion
            fila["train_rows"] = len(sub_train)
            fila["fit_seconds"] = segundos_fit
            filas_curva.append(fila)

    df_curva = pd.DataFrame(filas_curva)
    df_curva.to_csv(report_dir / "curva_tamano_muestra.csv", index=False)
    return df_referencia, df_curva


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--validation", default="validation.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--report-dir", default="resultados")
    args = parser.parse_args()
    run(args.train, args.validation, args.test, args.report_dir)
