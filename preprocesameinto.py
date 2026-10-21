"""Auditoría y limpieza reproducible del dataset comercial.

La política de calidad es conservadora: se corrigen únicamente magnitudes
derivadas cuya fuente determina un valor inequívoco (por ejemplo, gasto =
compras por ticket redondeado a centavos). Las filas con identificadores,
objetivos o dominios categóricos imposibles se eliminan porque no existe una
reconstrucción defendible. Las demás violaciones se reportan y se eliminan,
evitando fabricar evidencia para el modelo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from proyecto_utils import (
    CATEGORICAL_COLUMNS, CLASS_ORDER, DOMAINS, NUMERIC_COLUMNS, TARGET,
    ensure_dir, read_dataset, write_json,
)
from src.paths import DATA, DATASET_CANDIDATES, first_existing, default_dataset, ensure_layout

RANGES = {
    "edad": (18, 75), "antiguedad_dias": (365, 1825), "compras_365d": (0, 80),
    "dias_ultima_compra": (0, 366), "ticket_promedio_mxn": (0, np.inf),
    "gasto_total_365d_mxn": (0, np.inf), "unidades_365d": (0, np.inf),
    "categorias_compradas_365d": (0, 6), "sesiones_365d": (0, np.inf),
    "paginas_por_sesion": (1, 30), "duracion_sesion_min": (0.2, 60),
    "carritos_abandonados_365d": (0, np.inf), "compras_con_descuento_365d": (0, np.inf),
    "descuento_promedio_pct": (0, 50), "emails_enviados_365d": (12, 104),
    "clics_anuncios_365d": (0, np.inf), "puntos_lealtad": (0, 50000),
    "miembro_lealtad": (0, 1), "devoluciones_365d": (0, np.inf),
    "reclamos_365d": (0, np.inf), "entrega_promedio_dias": (1, 10),
    "satisfaccion_promedio": (1, 5), "emails_abiertos_365d": (0, np.inf),
}


def audit_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"input_rows": int(len(df)), "nulls": df.isna().sum().astype(int).to_dict(),
              "duplicate_rows": int(df.duplicated().sum()),
              "duplicate_ids": int(df["id_cliente"].duplicated().sum()),
              "invalid_domains": {}, "range_violations": {}, "logical_violations": {},
              "corrections": {}, "removed_rows": 0}
    numeric_for_analysis = [c for c in ["compras_365d", "ticket_promedio_mxn", "gasto_total_365d_mxn"] if c in df]
    report["collinearity"] = df[numeric_for_analysis].corr().round(4).to_dict() if numeric_for_analysis else {}
    report["outliers_iqr"] = {}
    for col in NUMERIC_COLUMNS:
        if col in df:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            report["outliers_iqr"][col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
    bad = pd.Series(False, index=df.index)
    for col, domain in DOMAINS.items():
        if col in df:
            invalid = df[col].notna() & ~df[col].isin(domain)
            report["invalid_domains"][col] = int(invalid.sum())
            bad |= invalid
    for col, (lo, hi) in RANGES.items():
        invalid = df[col].isna() | (df[col] < lo) | (df[col] > hi)
        report["range_violations"][col] = int(invalid.sum())
        bad |= invalid
    # Una fila con cualquier nulo no puede pasar al codificador ni al MLP:
    # no se imputan categorías/objetivo porque no hay una inferencia segura.
    bad |= df.isna().any(axis=1)
    bad |= df["id_cliente"].isna() | df[TARGET].isna() | df["id_cliente"].duplicated(keep=False)

    no_purchase = df["compras_365d"].eq(0)
    zero_cols = ["ticket_promedio_mxn", "gasto_total_365d_mxn", "unidades_365d",
                 "categorias_compradas_365d", "compras_con_descuento_365d", "devoluciones_365d"]
    for col in zero_cols:
        invalid = no_purchase & df[col].ne(0)
        report["logical_violations"][f"no_purchase_{col}"] = int(invalid.sum())
        # Estos campos están determinados por compras=0 y se corrigen de forma segura.
        df.loc[no_purchase, col] = 0
    invalid = no_purchase & df["dias_ultima_compra"].ne(366)
    report["logical_violations"]["no_purchase_dias_ultima_compra"] = int(invalid.sum())
    df.loc[no_purchase, "dias_ultima_compra"] = 366

    expected = (df["compras_365d"] * df["ticket_promedio_mxn"]).round(2)
    mismatch = ~np.isclose(df["gasto_total_365d_mxn"], expected, atol=0.005, equal_nan=False)
    report["logical_violations"]["gasto_derivado"] = int(mismatch.sum())
    df.loc[~bad, "gasto_total_365d_mxn"] = expected[~bad]
    coupled = {
        "devoluciones_mayor_compras": df["devoluciones_365d"] > df["compras_365d"],
        "descuento_mayor_compras": df["compras_con_descuento_365d"] > df["compras_365d"],
        "categorias_mayor_limite": df["categorias_compradas_365d"] > np.minimum(6, df["compras_365d"]),
        "sesiones_menor_compras": df["sesiones_365d"] < df["compras_365d"],
        "carritos_mayor_disponibles": df["carritos_abandonados_365d"] > (df["sesiones_365d"] - df["compras_365d"]),
        "emails_abiertos_mayor_enviados": df["emails_abiertos_365d"] > df["emails_enviados_365d"],
    }
    for name, invalid in coupled.items():
        report["logical_violations"][name] = int(invalid.sum())
        bad |= invalid
    report["corrections"]["derived_gasto"] = int((~bad & mismatch).sum())
    bad |= mismatch & ~no_purchase
    cleaned = df.loc[~bad].drop_duplicates().copy()
    report["removed_rows"] = int(len(df) - len(cleaned))
    report["output_rows"] = int(len(cleaned))
    report["class_distribution"] = cleaned[TARGET].value_counts().to_dict()
    return cleaned, report


def run(input_path=None, output_path="dataset_limpio.csv",
        report_dir="reportes"):
    ensure_layout()
    input_path = default_dataset() if input_path is None else Path(input_path)
    if not input_path.exists():
        input_path = first_existing(input_path.name, *DATASET_CANDIDATES)
    if output_path == "dataset_limpio.csv":
        output_path = DATA / "dataset_limpio.csv"
    if report_dir == "reportes":
        report_dir = "resultados"
    df = read_dataset(input_path)
    cleaned, report = audit_and_clean(df)
    cleaned.to_csv(output_path, index=False, encoding="utf-8")
    ensure_dir(report_dir)
    write_json(report, Path(report_dir) / "auditoria_preprocesamiento.json")
    Path(report_dir, "auditoria_preprocesamiento.md").write_text(
        "# Auditoría de preprocesamiento\n\n```json\n" +
        __import__("json").dumps(report, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(f"Filas de entrada: {len(df):,}; salida limpia: {len(cleaned):,}; eliminadas: {report['removed_rows']:,}")
    return cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(default_dataset()))
    parser.add_argument("--output", default=str(DATA / "dataset_limpio.csv"))
    parser.add_argument("--report-dir", default="resultados")
    args = parser.parse_args()
    run(args.input, args.output, args.report_dir)
