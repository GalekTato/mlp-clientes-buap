"""Utilidades compartidas de lectura, normalización, preprocesamiento y evaluación."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

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
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _clave_canonica(texto: str) -> str:
    """Normaliza una cadena quitando acentos, espacios extremos y unificando separadores con guion bajo."""
    sin_acentos = "".join(
        caracter for caracter in unicodedata.normalize("NFD", texto.strip().lower())
        if unicodedata.category(caracter) != "Mn"
    )
    return "_".join(sin_acentos.split())


def clean_numeric_text(series: pd.Series) -> pd.Series:
    """Elimina símbolos monetarios ($), comas, espacios y sufijo MXN antes de la conversión numérica."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    return (
        series.astype("string")
        .str.replace(r"[\$,\s]|MXN", "", regex=True, case=False)
        .replace("", pd.NA)
    )


def normalize_categorical_series(series: pd.Series, domain: Iterable[str]) -> tuple[pd.Series, int]:
    """Mapea variantes léxicas de una columna categórica a su etiqueta canónica definida en DOMAINS."""
    conjunto_dominio = set(domain)
    mapa_canonico = {_clave_canonica(valor): valor for valor in conjunto_dominio}
    serie_texto = series.astype("string")
    era_valido = serie_texto.notna() & serie_texto.isin(conjunto_dominio)

    valores_unicos = serie_texto.dropna().unique()
    traducciones = {}
    for valor_crudo in valores_unicos:
        if valor_crudo in conjunto_dominio:
            traducciones[valor_crudo] = valor_crudo
        else:
            clave = _clave_canonica(str(valor_crudo))
            traducciones[valor_crudo] = mapa_canonico.get(clave, valor_crudo)

    serie_normalizada = serie_texto.map(traducciones).astype("string")
    es_valido_ahora = serie_normalizada.notna() & serie_normalizada.isin(conjunto_dominio)
    total_corregidos = int((~era_valido & es_valido_ahora).sum())
    return serie_normalizada, total_corregidos


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Carga un archivo CSV aplicando tipos explícitos y limpieza determinista de formato monetario."""
    path = Path(path)
    dtypes = {col: "string" for col in CATEGORICAL_COLUMNS + [TARGET]}
    dtypes.update({"id_cliente": "Int64", "miembro_lealtad": "Int8"})
    df = pd.read_csv(path, dtype=dtypes, low_memory=False)
    conteos_moneda: dict[str, int] = {}
    for col in NUMERIC_COLUMNS:
        if col in df:
            columna_cruda = df[col]
            if not pd.api.types.is_numeric_dtype(columna_cruda):
                conversion_directa = pd.to_numeric(columna_cruda, errors="coerce")
                conversion_limpia = pd.to_numeric(clean_numeric_text(columna_cruda), errors="coerce")
                recuperados = int((columna_cruda.notna() & conversion_directa.isna() & conversion_limpia.notna()).sum())
                if recuperados > 0:
                    conteos_moneda[col] = recuperados
                df[col] = conversion_limpia
            else:
                df[col] = pd.to_numeric(columna_cruda, errors="coerce")
    df.attrs["normalized_currency_counts"] = conteos_moneda
    return df


def write_json(data: dict | list, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def make_preprocessor(scaler: str = "standard", sparse_output: bool = True):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

    escalador = RobustScaler() if scaler == "robust" else StandardScaler()
    return ColumnTransformer(
        transformers=[
            ("num", escalador, NUMERIC_COLUMNS + DOMAIN_FEATURE_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output), CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def build_base_pipeline(
    hidden_layer_sizes: tuple[int, ...] = (64,),
    max_iter: int = 80,
    learning_rate_init: float = 0.001,
    solver: str = "adam",
    activation: str = "relu",
    alpha: float = 0.001,
    batch_size: int = 256,
    random_state: int = SEED,
) -> Pipeline:
    """Construye el pipeline estándar de ingeniería de variables, preprocesamiento y MLPClassifier."""
    from src.feature_engineering import DomainFeatureTransformer

    return Pipeline([
        ("features", DomainFeatureTransformer()),
        ("preprocessor", make_preprocessor("standard")),
        ("model", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            solver=solver,
            activation=activation,
            alpha=alpha,
            batch_size=batch_size,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4,
        )),
    ])


def evaluate_predictions(y_true, y_pred, y_proba, classes, split_name: str) -> dict:
    """Calcula el conjunto completo de métricas de clasificación multiclase sobre predicciones ya generadas."""
    return {
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr": float(roc_auc_score(y_true, y_proba, multi_class="ovr", labels=classes)),
    }


def finite_population_sample_size(N: int, z: float = 2.576, e: float = 0.005, p: float = 0.5) -> int:
    """Fórmula de Cochran para proporciones con corrección por población finita."""
    n0 = (z**2 * p * (1 - p)) / e**2
    n = n0 / (1 + (n0 - 1) / N)
    return int(np.ceil(n))


def numeric_summary(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    columnas_presentes = [col for col in columns if col in df]
    return df[columnas_presentes].agg(["mean", "std", "min", "max"]).T.reset_index(names="variable")
