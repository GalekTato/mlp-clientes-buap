"""Utilidades compartidas del proyecto de minería de datos."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd

SEED = 42
TARGET = "clase_cliente"
CATEGORICAL_COLUMNS = [
    "region", "zona", "canal_preferido", "dispositivo_preferido",
    "categoria_preferida", "metodo_pago_preferido",
]
NUMERIC_COLUMNS = [
    "edad", "antiguedad_dias", "compras_365d", "dias_ultima_compra",
    "ticket_promedio_mxn", "gasto_total_365d_mxn", "unidades_365d",
    "categorias_compradas_365d", "sesiones_365d", "paginas_por_sesion",
    "duracion_sesion_min", "carritos_abandonados_365d",
    "compras_con_descuento_365d", "descuento_promedio_pct",
    "emails_enviados_365d", "clics_anuncios_365d", "puntos_lealtad",
    "miembro_lealtad", "devoluciones_365d", "reclamos_365d",
    "entrega_promedio_dias", "satisfaccion_promedio",
    "emails_abiertos_365d",
]
DOMAIN_FEATURE_COLUMNS = [
    "ratio_descuento", "frecuencia_anualizada", "intensidad_recencia",
    "unidades_por_compra", "tasa_abandono_carrito", "tasa_apertura_emails",
    "tasa_devoluciones", "indice_valor_cliente",
]
CLASS_ORDER = ["Inactivo", "Alto_valor", "Promocional", "Frecuente", "Ocasional"]
DOMAINS = {
    "region": {"Centro", "Norte", "Occidente", "Sur", "Sureste"},
    "zona": {"Urbana", "Suburbana", "Rural"},
    "canal_preferido": {"App", "Web", "Tienda"},
    "dispositivo_preferido": {"Movil", "Escritorio", "Tablet"},
    "categoria_preferida": {"Tecnologia", "Hogar", "Moda", "Deportes", "Belleza", "Alimentos"},
    "metodo_pago_preferido": {"Tarjeta", "Transferencia", "Efectivo", "Cartera_digital"},
    TARGET: set(CLASS_ORDER),
}


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Lee el CSV con dtype explícito para evitar inferencias inconsistentes."""
    path = Path(path)
    dtype = {c: "string" for c in CATEGORICAL_COLUMNS + [TARGET]}
    dtype.update({"id_cliente": "Int64", "miembro_lealtad": "Int8"})
    df = pd.read_csv(path, dtype=dtype, low_memory=False)
    for col in NUMERIC_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def write_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def make_preprocessor(scaler="standard"):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

    scaler_obj = RobustScaler() if scaler == "robust" else StandardScaler()
    return ColumnTransformer(
        transformers=[
            ("num", scaler_obj, NUMERIC_COLUMNS + DOMAIN_FEATURE_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def finite_population_sample_size(N: int, z: float = 2.576, e: float = 0.005, p: float = 0.5) -> int:
    """Cochran para proporciones con corrección por población finita."""
    n0 = (z**2 * p * (1 - p)) / e**2
    n = n0 / (1 + (n0 - 1) / N)
    return int(np.ceil(n))


def numeric_summary(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    cols = [c for c in columns if c in df]
    return df[cols].agg(["mean", "std", "min", "max"]).T.reset_index(names="variable")
