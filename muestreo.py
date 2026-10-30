"""Cálculo de tamaño de muestra de Cochran y particiones estratificadas reproducibles."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp
from sklearn.model_selection import train_test_split

from proyecto_utils import (
    CLASS_ORDER,
    NUMERIC_COLUMNS,
    SEED,
    TARGET,
    ensure_dir,
    finite_population_sample_size,
    read_dataset,
    write_json,
)
from src.paths import DATA, first_existing


def calculate_sample_size(df: pd.DataFrame, error: float = 0.005, confidence_z: float = 2.576) -> dict:
    """Calcula la muestra mínima con corrección por población finita y su afijación proporcional por clase."""
    total_poblacion = len(df)
    minimo_cochran = finite_population_sample_size(total_poblacion, confidence_z, error)
    conteos_por_clase = df[TARGET].value_counts()
    asignacion_por_clase = {
        clase: max(int(np.ceil(minimo_cochran * conteos_por_clase.get(clase, 0) / total_poblacion)), 1000)
        for clase in CLASS_ORDER
    }
    return {
        "N": total_poblacion,
        "z": confidence_z,
        "error": error,
        "cochran_fpc": minimo_cochran,
        "allocated_by_class": asignacion_por_clase,
        "allocated_total": int(sum(asignacion_por_clase.values())),
    }


def make_partitions(df: pd.DataFrame, sample_size: int | None = None, seed: int = SEED):
    """Divide la muestra estratificada en 70 % entrenamiento, 15 % validación y 15 % prueba."""
    if sample_size is None or sample_size >= len(df):
        muestra = df
    else:
        _, muestra = train_test_split(df, test_size=sample_size, stratify=df[TARGET], random_state=seed)
    train, resto = train_test_split(muestra, test_size=0.30, stratify=muestra[TARGET], random_state=seed)
    validation, test = train_test_split(resto, test_size=0.50, stratify=resto[TARGET], random_state=seed)
    return train, validation, test


def distribution_report(full, sample, train, validation, test) -> pd.DataFrame:
    particiones = {"original": full, "sample": sample, "train": train, "validation": validation, "test": test}
    filas = []
    for nombre_particion, subconjunto in particiones.items():
        proporciones = subconjunto[TARGET].value_counts(normalize=True)
        for clase in CLASS_ORDER:
            filas.append({
                "partition": nombre_particion,
                "class": clase,
                "count": int((subconjunto[TARGET] == clase).sum()),
                "proportion": float(proporciones.get(clase, 0)),
            })
    return pd.DataFrame(filas)


def run(input_path="dataset_limpio.csv", output_dir=".", report_dir="reportes", error=0.005):
    if not Path(input_path).exists():
        input_path = first_existing("dataset_limpio.csv")
    if output_dir == ".":
        output_dir = DATA
    if report_dir == "reportes":
        report_dir = "resultados"
    df = read_dataset(input_path)
    estadisticas_muestra = calculate_sample_size(df, error)
    tamano_solicitado = min(len(df), estadisticas_muestra["allocated_total"])
    train, validation, test = make_partitions(df, tamano_solicitado)
    muestra = pd.concat([train, validation, test], ignore_index=True)

    output_dir, report_dir = Path(output_dir), ensure_dir(report_dir)
    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(output_dir / "validation.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    distribucion = distribution_report(df, muestra, train, validation, test)
    distribucion.to_csv(report_dir / "distribucion_particiones.csv", index=False)

    resumenes_numericos = []
    for nombre_particion, subconjunto in {
        "original": df, "sample": muestra, "train": train, "validation": validation, "test": test
    }.items():
        resumen = subconjunto[NUMERIC_COLUMNS].agg(["mean", "std"]).T.reset_index(names="variable")
        resumen["partition"] = nombre_particion
        resumenes_numericos.append(resumen)
    pd.concat(resumenes_numericos).to_csv(report_dir / "resumen_numerico_particiones.csv", index=False)

    pruebas_bondad = {}
    for clase in CLASS_ORDER:
        observado = np.array([(muestra[TARGET] == clase).sum(), (muestra[TARGET] != clase).sum()])
        esperado = np.array([(df[TARGET] == clase).mean() * len(muestra), (df[TARGET] != clase).mean() * len(muestra)])
        pruebas_bondad[clase] = {"chi2_pvalue": float(chisquare(observado, esperado).pvalue)}
    for col in NUMERIC_COLUMNS:
        valores_poblacion, valores_muestra = df[col].dropna(), muestra[col].dropna()
        pruebas_bondad[f"ks_{col}"] = {"ks_pvalue": float(ks_2samp(valores_poblacion, valores_muestra).pvalue)}

    estadisticas_muestra["actual_sample_size"] = len(muestra)
    estadisticas_muestra["tests"] = pruebas_bondad
    write_json(estadisticas_muestra, report_dir / "calculo_muestra.json")
    print(f"Muestra: {len(muestra):,}; train={len(train):,}, validation={len(validation):,}, test={len(test):,}")
    return train, validation, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(first_existing("dataset_limpio.csv")))
    parser.add_argument("--output-dir", default=str(DATA))
    parser.add_argument("--report-dir", default="resultados")
    parser.add_argument("--error", type=float, default=0.005)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.report_dir, args.error)
