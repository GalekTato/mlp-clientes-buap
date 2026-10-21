"""Cálculo de muestra y particiones estratificadas reproducibles."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from proyecto_utils import CLASS_ORDER, NUMERIC_COLUMNS, TARGET, finite_population_sample_size, ensure_dir, read_dataset, write_json
from src.paths import DATA, first_existing


def calculate_sample_size(df: pd.DataFrame, error=0.005, confidence_z=2.576) -> dict:
    total = len(df)
    minimum = finite_population_sample_size(total, confidence_z, error)
    counts = df[TARGET].value_counts()
    allocation = {c: max(int(np.ceil(minimum * counts.get(c, 0) / total)), 1000) for c in CLASS_ORDER}
    return {"N": total, "z": confidence_z, "error": error, "cochran_fpc": minimum,
            "allocated_by_class": allocation, "allocated_total": int(sum(allocation.values()))}


def make_partitions(df, sample_size=None, seed=42):
    if sample_size is None or sample_size >= len(df):
        sample = df
    else:
        _, sample = train_test_split(df, test_size=sample_size, stratify=df[TARGET], random_state=seed)
    train, rest = train_test_split(sample, test_size=0.30, stratify=sample[TARGET], random_state=seed)
    validation, test = train_test_split(rest, test_size=0.50, stratify=rest[TARGET], random_state=seed)
    return train, validation, test


def distribution_report(full, sample, train, validation, test):
    pieces = {"original": full, "sample": sample, "train": train, "validation": validation, "test": test}
    rows = []
    for name, part in pieces.items():
        proportions = part[TARGET].value_counts(normalize=True)
        for cls in CLASS_ORDER:
            rows.append({"partition": name, "class": cls, "count": int((part[TARGET] == cls).sum()),
                         "proportion": float(proportions.get(cls, 0))})
    return pd.DataFrame(rows)


def run(input_path="dataset_limpio.csv", output_dir=".", report_dir="reportes", error=0.005):
    if not Path(input_path).exists():
        input_path = first_existing("dataset_limpio.csv")
    if output_dir == ".":
        output_dir = DATA
    if report_dir == "reportes":
        report_dir = "resultados"
    df = read_dataset(input_path)
    stats = calculate_sample_size(df, error)
    requested = min(len(df), stats["allocated_total"])
    train, validation, test = make_partitions(df, requested)
    sample = pd.concat([train, validation, test], ignore_index=True)
    output_dir, report_dir = Path(output_dir), ensure_dir(report_dir)
    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(output_dir / "validation.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    dist = distribution_report(df, sample, train, validation, test)
    dist.to_csv(report_dir / "distribucion_particiones.csv", index=False)
    pivot = dist.pivot(index="class", columns="partition", values="proportion").reindex(CLASS_ORDER)
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Proporción"); plt.title("Distribución de clases antes y después del muestreo")
    plt.tight_layout(); plt.savefig(Path(report_dir).parent / "graficas_distribucion_clases.png", dpi=300); plt.close()
    numeric = []
    for name, part in {"original": df, "sample": sample, "train": train, "validation": validation, "test": test}.items():
        summary = part[NUMERIC_COLUMNS].agg(["mean", "std"]).T.reset_index(names="variable")
        summary["partition"] = name
        numeric.append(summary)
    pd.concat(numeric).to_csv(report_dir / "resumen_numerico_particiones.csv", index=False)
    tests = {}
    for cls in CLASS_ORDER:
        observed = np.array([(sample[TARGET] == cls).sum(), (sample[TARGET] != cls).sum()])
        expected = np.array([(df[TARGET] == cls).mean() * len(sample), (df[TARGET] != cls).mean() * len(sample)])
        tests[cls] = {"chi2_pvalue": float(chisquare(observed, expected).pvalue)}
    for col in NUMERIC_COLUMNS:
        a, b = df[col].dropna(), sample[col].dropna()
        tests[f"ks_{col}"] = {"ks_pvalue": float(ks_2samp(a, b).pvalue)}
    stats["actual_sample_size"] = len(sample)
    stats["tests"] = tests
    write_json(stats, report_dir / "calculo_muestra.json")
    print(f"Muestra: {len(sample):,}; train={len(train):,}, validation={len(validation):,}, test={len(test):,}")
    return train, validation, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(first_existing("dataset_limpio.csv")))
    parser.add_argument("--output-dir", default=str(DATA))
    parser.add_argument("--report-dir", default="resultados")
    parser.add_argument("--error", type=float, default=0.005)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.report_dir, args.error)
