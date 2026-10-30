"""Auditoría y limpieza reproducible del dataset comercial.

Política técnica de calidad de datos:
Se normaliza o corrige un atributo cuando el valor correcto está determinado de
forma inequívoca por las reglas del dominio:
1) formato monetario en ``ticket_promedio_mxn`` (símbolo ``$``, comas, espacios
   y sufijo ``MXN``) cuyo valor numérico subyacente es exacto;
2) variantes de escritura en variables categóricas (espacios extremos,
   mayúsculas/minúsculas, acentos o separadores como ``cartera digital``) que
   corresponden sin ambigüedad a un valor canónico de ``DOMAINS``; y
3) ``gasto_total_365d_mxn``, definido determinísticamente en la especificación
   técnica como ``round(compras_365d * ticket_promedio_mxn, 2)``.

Cuando el dato real no puede reconstruirse de forma determinista (valores nulos,
valores fuera de rango operativo, categorías ajenas al dominio o violaciones de
dependencias lógicas entre conteos), el registro se descarta para no introducir
información sintética artificial en el entrenamiento.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from proyecto_utils import (
    CATEGORICAL_COLUMNS,
    CLASS_ORDER,
    DOMAINS,
    NUMERIC_COLUMNS,
    TARGET,
    clean_numeric_text,
    ensure_dir,
    normalize_categorical_series,
    read_dataset,
    write_json,
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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza determinista y filtrado de calidad sin calcular estadísticas de reporte."""
    limpio, _ = _procesar_calidad(df, incluir_estadisticas=False)
    return limpio


def audit_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ejecuta la auditoría completa de calidad y devuelve el DataFrame limpio junto con el reporte."""
    return _procesar_calidad(df, incluir_estadisticas=True)


def _procesar_calidad(df: pd.DataFrame, incluir_estadisticas: bool = True) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    conteos_moneda = dict(df.attrs.get("normalized_currency_counts", {}))
    for col in NUMERIC_COLUMNS:
        if col in df:
            if not pd.api.types.is_numeric_dtype(df[col]):
                conversion_directa = pd.to_numeric(df[col], errors="coerce")
                conversion_limpia = pd.to_numeric(clean_numeric_text(df[col]), errors="coerce")
                recuperados = int((df[col].notna() & conversion_directa.isna() & conversion_limpia.notna()).sum())
                if recuperados > 0:
                    conteos_moneda[col] = conteos_moneda.get(col, 0) + recuperados
                df[col] = conversion_limpia
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    categoricas_normalizadas: dict[str, int] = {}
    for col, dominio in DOMAINS.items():
        if col in df:
            df[col], total_normalizados = normalize_categorical_series(df[col], dominio)
            if col in CATEGORICAL_COLUMNS or total_normalizados > 0:
                categoricas_normalizadas[col] = total_normalizados

    report: dict = {
        "input_rows": int(len(df)),
        "nulls": df.isna().sum().astype(int).to_dict() if incluir_estadisticas else {},
        "duplicate_rows": int(df.duplicated().sum()) if incluir_estadisticas else 0,
        "duplicate_ids": int(df["id_cliente"].duplicated().sum()) if (incluir_estadisticas and "id_cliente" in df) else 0,
        "invalid_domains": {},
        "range_violations": {},
        "logical_violations": {},
        "corrections": {
            "normalized_categorical": categoricas_normalizadas,
            "normalized_categorical_total": int(sum(categoricas_normalizadas.values())),
            "normalized_currency_ticket": int(conteos_moneda.get("ticket_promedio_mxn", 0)),
            "derived_gasto": 0,
        },
        "removed_by_cause": {},
        "removed_rows": 0,
    }

    if incluir_estadisticas:
        columnas_colinealidad = [c for c in ["compras_365d", "ticket_promedio_mxn", "gasto_total_365d_mxn"] if c in df]
        report["collinearity"] = df[columnas_colinealidad].corr().round(4).to_dict() if columnas_colinealidad else {}
        report["outliers_iqr"] = {}
        for col in NUMERIC_COLUMNS:
            if col in df:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                report["outliers_iqr"][col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())

    mascara_nulos = df.isna().any(axis=1)
    if "id_cliente" in df:
        mascara_nulos |= df["id_cliente"].isna()
    if TARGET in df:
        mascara_nulos |= df[TARGET].isna()

    mascara_dominio = pd.Series(False, index=df.index)
    for col, dominio in DOMAINS.items():
        if col in df:
            fuera_de_dominio = df[col].notna() & ~df[col].isin(dominio)
            report["invalid_domains"][col] = int(fuera_de_dominio.sum())
            mascara_dominio |= fuera_de_dominio

    mascara_rango = pd.Series(False, index=df.index)
    for col, (minimo, maximo) in RANGES.items():
        if col in df:
            fuera_de_rango = df[col].notna() & ((df[col] < minimo) | (df[col] > maximo))
            report["range_violations"][col] = int(fuera_de_rango.sum())
            mascara_rango |= fuera_de_rango

    sin_compras = df["compras_365d"].eq(0)
    columnas_cero_sin_compra = [
        "ticket_promedio_mxn", "gasto_total_365d_mxn", "unidades_365d",
        "categorias_compradas_365d", "compras_con_descuento_365d", "devoluciones_365d",
    ]
    for col in columnas_cero_sin_compra:
        violacion = sin_compras & df[col].notna() & df[col].ne(0)
        report["logical_violations"][f"no_purchase_{col}"] = int(violacion.sum())
    recencia_invalida = sin_compras & df["dias_ultima_compra"].notna() & df["dias_ultima_compra"].ne(366)
    report["logical_violations"]["no_purchase_dias_ultima_compra"] = int(recencia_invalida.sum())

    gasto_esperado = (df["compras_365d"] * df["ticket_promedio_mxn"]).round(2)
    desfase_gasto = (
        df["gasto_total_365d_mxn"].notna()
        & gasto_esperado.notna()
        & ~np.isclose(df["gasto_total_365d_mxn"], gasto_esperado, atol=0.005, equal_nan=False)
    )
    report["logical_violations"]["gasto_derivado"] = int(desfase_gasto.sum())

    reglas_acopladas = {
        "no_purchase_ticket_positivo": sin_compras & df["ticket_promedio_mxn"].notna() & df["ticket_promedio_mxn"].ne(0),
        "no_purchase_unidades_positivas": sin_compras & df["unidades_365d"].notna() & df["unidades_365d"].ne(0),
        "no_purchase_categorias_positivas": sin_compras & df["categorias_compradas_365d"].notna() & df["categorias_compradas_365d"].ne(0),
        "no_purchase_descuentos_positivos": sin_compras & df["compras_con_descuento_365d"].notna() & df["compras_con_descuento_365d"].ne(0),
        "no_purchase_devoluciones_positivas": sin_compras & df["devoluciones_365d"].notna() & df["devoluciones_365d"].ne(0),
        "no_purchase_recencia_invalida": recencia_invalida,
        "devoluciones_mayor_compras": df["devoluciones_365d"] > df["compras_365d"],
        "descuento_mayor_compras": df["compras_con_descuento_365d"] > df["compras_365d"],
        "categorias_mayor_limite": df["categorias_compradas_365d"] > np.minimum(6, df["compras_365d"]),
        "sesiones_menor_compras": df["sesiones_365d"] < df["compras_365d"],
        "carritos_mayor_disponibles": df["carritos_abandonados_365d"] > (df["sesiones_365d"] - df["compras_365d"]),
        "emails_abiertos_mayor_enviados": df["emails_abiertos_365d"] > df["emails_enviados_365d"],
    }
    mascara_dependencias = pd.Series(False, index=df.index)
    for nombre_regla, violacion in reglas_acopladas.items():
        violacion = violacion.fillna(False)
        if nombre_regla not in report["logical_violations"]:
            report["logical_violations"][nombre_regla] = int(violacion.sum())
        mascara_dependencias |= violacion

    # Cada fila inválida se contabiliza en la primera causa que cumple para que las causas sean mutuamente excluyentes
    causa_nulos = mascara_nulos
    causa_rango = ~causa_nulos & (mascara_rango | mascara_dominio)
    causa_dependencias = ~causa_nulos & ~causa_rango & mascara_dependencias
    descarte_previo_duplicados = causa_nulos | causa_rango | causa_dependencias

    mascara_duplicados = pd.Series(False, index=df.index)
    if "id_cliente" in df:
        mascara_duplicados |= df["id_cliente"].duplicated(keep="first")
    mascara_duplicados |= df.duplicated(keep="first")
    causa_duplicados = ~descarte_previo_duplicados & mascara_duplicados

    filas_descartadas = causa_nulos | causa_rango | causa_dependencias | causa_duplicados

    # Corrección determinista del gasto total en todas las filas válidas conservadas
    gasto_corregido = ~filas_descartadas & desfase_gasto
    df.loc[~filas_descartadas, "gasto_total_365d_mxn"] = gasto_esperado.loc[~filas_descartadas]
    report["corrections"]["derived_gasto"] = int(gasto_corregido.sum())

    cleaned = df.loc[~filas_descartadas].copy()
    report["removed_by_cause"] = {
        "nulos": int(causa_nulos.sum()),
        "fuera_de_rango": int(causa_rango.sum()),
        "dependencias": int(causa_dependencias.sum()),
        "duplicados": int(causa_duplicados.sum()),
    }
    report["removed_rows"] = int(len(df) - len(cleaned))
    report["output_rows"] = int(len(cleaned))
    report["class_distribution"] = cleaned[TARGET].value_counts().to_dict() if TARGET in cleaned else {}
    return cleaned, report


def run(input_path=None, output_path="dataset_limpio.csv", report_dir="reportes"):
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
        "# Auditoría de preprocesamiento\n\n```json\n"
        + json.dumps(report, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(f"Filas de entrada: {len(df):,}; salida limpia: {len(cleaned):,}; eliminadas: {report['removed_rows']:,}")
    return cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(default_dataset()))
    parser.add_argument("--output", default=str(DATA / "dataset_limpio.csv"))
    parser.add_argument("--report-dir", default="resultados")
    args = parser.parse_args()
    run(args.input, args.output, args.report_dir)
