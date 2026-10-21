"""Variables de dominio para distinguir recurrencia, descuento y valor."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = [
    "ratio_descuento", "frecuencia_anualizada", "intensidad_recencia",
    "unidades_por_compra", "tasa_abandono_carrito", "tasa_apertura_emails",
    "tasa_devoluciones", "indice_valor_cliente",
]

def add_domain_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    n = np.maximum(pd.to_numeric(df["compras_365d"]), 1)
    df["ratio_descuento"] = pd.to_numeric(df["compras_con_descuento_365d"]) / n
    df["frecuencia_anualizada"] = pd.to_numeric(df["compras_365d"]) / (pd.to_numeric(df["antiguedad_dias"]) / 365.0)
    df["intensidad_recencia"] = pd.to_numeric(df["compras_365d"]) / (pd.to_numeric(df["dias_ultima_compra"]) + 1.0)
    df["unidades_por_compra"] = pd.to_numeric(df["unidades_365d"]) / n
    df["tasa_abandono_carrito"] = pd.to_numeric(df["carritos_abandonados_365d"]) / np.maximum(pd.to_numeric(df["sesiones_365d"]), 1)
    df["tasa_apertura_emails"] = pd.to_numeric(df["emails_abiertos_365d"]) / np.maximum(pd.to_numeric(df["emails_enviados_365d"]), 1)
    df["tasa_devoluciones"] = pd.to_numeric(df["devoluciones_365d"]) / n
    df["indice_valor_cliente"] = pd.to_numeric(df["gasto_total_365d_mxn"]) / (pd.to_numeric(df["dias_ultima_compra"]) + 1.0)
    return df

class DomainFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return add_domain_features(X)
