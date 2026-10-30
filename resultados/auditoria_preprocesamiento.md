# Auditoría de preprocesamiento

```json
{
  "input_rows": 1000000,
  "nulls": {
    "id_cliente": 0,
    "edad": 4297,
    "region": 4234,
    "zona": 0,
    "antiguedad_dias": 0,
    "canal_preferido": 0,
    "dispositivo_preferido": 0,
    "categoria_preferida": 0,
    "compras_365d": 0,
    "dias_ultima_compra": 0,
    "ticket_promedio_mxn": 4287,
    "gasto_total_365d_mxn": 0,
    "unidades_365d": 0,
    "categorias_compradas_365d": 0,
    "sesiones_365d": 0,
    "paginas_por_sesion": 4264,
    "duracion_sesion_min": 4349,
    "carritos_abandonados_365d": 0,
    "compras_con_descuento_365d": 0,
    "descuento_promedio_pct": 0,
    "emails_enviados_365d": 0,
    "clics_anuncios_365d": 0,
    "puntos_lealtad": 0,
    "miembro_lealtad": 0,
    "devoluciones_365d": 0,
    "reclamos_365d": 0,
    "entrega_promedio_dias": 0,
    "satisfaccion_promedio": 4238,
    "metodo_pago_preferido": 4331,
    "emails_abiertos_365d": 0,
    "clase_cliente": 0
  },
  "duplicate_rows": 0,
  "duplicate_ids": 0,
  "invalid_domains": {
    "region": 0,
    "zona": 0,
    "canal_preferido": 0,
    "dispositivo_preferido": 0,
    "categoria_preferida": 0,
    "metodo_pago_preferido": 0,
    "clase_cliente": 0
  },
  "range_violations": {
    "edad": 10026,
    "antiguedad_dias": 0,
    "compras_365d": 0,
    "dias_ultima_compra": 0,
    "ticket_promedio_mxn": 0,
    "gasto_total_365d_mxn": 0,
    "unidades_365d": 0,
    "categorias_compradas_365d": 0,
    "sesiones_365d": 0,
    "paginas_por_sesion": 9965,
    "duracion_sesion_min": 10009,
    "carritos_abandonados_365d": 0,
    "compras_con_descuento_365d": 0,
    "descuento_promedio_pct": 0,
    "emails_enviados_365d": 0,
    "clics_anuncios_365d": 0,
    "puntos_lealtad": 0,
    "miembro_lealtad": 0,
    "devoluciones_365d": 0,
    "reclamos_365d": 0,
    "entrega_promedio_dias": 0,
    "satisfaccion_promedio": 0,
    "emails_abiertos_365d": 0
  },
  "logical_violations": {
    "no_purchase_ticket_promedio_mxn": 0,
    "no_purchase_gasto_total_365d_mxn": 1034,
    "no_purchase_unidades_365d": 0,
    "no_purchase_categorias_compradas_365d": 0,
    "no_purchase_compras_con_descuento_365d": 0,
    "no_purchase_devoluciones_365d": 0,
    "no_purchase_dias_ultima_compra": 0,
    "gasto_derivado": 30000,
    "no_purchase_ticket_positivo": 0,
    "no_purchase_unidades_positivas": 0,
    "no_purchase_categorias_positivas": 0,
    "no_purchase_descuentos_positivos": 0,
    "no_purchase_devoluciones_positivas": 0,
    "no_purchase_recencia_invalida": 0,
    "devoluciones_mayor_compras": 0,
    "descuento_mayor_compras": 0,
    "categorias_mayor_limite": 0,
    "sesiones_menor_compras": 0,
    "carritos_mayor_disponibles": 0,
    "emails_abiertos_mayor_enviados": 0
  },
  "corrections": {
    "normalized_categorical": {
      "region": 7517,
      "zona": 0,
      "canal_preferido": 7469,
      "dispositivo_preferido": 0,
      "categoria_preferida": 0,
      "metodo_pago_preferido": 7452
    },
    "normalized_categorical_total": 22438,
    "normalized_currency_ticket": 7562,
    "derived_gasto": 30000
  },
  "removed_by_cause": {
    "nulos": 30000,
    "fuera_de_rango": 30000,
    "dependencias": 0,
    "duplicados": 0
  },
  "removed_rows": 60000,
  "collinearity": {
    "compras_365d": {
      "compras_365d": 1.0,
      "ticket_promedio_mxn": 0.3297,
      "gasto_total_365d_mxn": 0.7558
    },
    "ticket_promedio_mxn": {
      "compras_365d": 0.3297,
      "ticket_promedio_mxn": 1.0,
      "gasto_total_365d_mxn": 0.732
    },
    "gasto_total_365d_mxn": {
      "compras_365d": 0.7558,
      "ticket_promedio_mxn": 0.732,
      "gasto_total_365d_mxn": 1.0
    }
  },
  "outliers_iqr": {
    "edad": 9971,
    "antiguedad_dias": 0,
    "compras_365d": 28011,
    "dias_ultima_compra": 83991,
    "ticket_promedio_mxn": 40498,
    "gasto_total_365d_mxn": 67978,
    "unidades_365d": 25086,
    "categorias_compradas_365d": 0,
    "sesiones_365d": 14914,
    "paginas_por_sesion": 37621,
    "duracion_sesion_min": 46254,
    "carritos_abandonados_365d": 37868,
    "compras_con_descuento_365d": 19781,
    "descuento_promedio_pct": 1284,
    "emails_enviados_365d": 0,
    "clics_anuncios_365d": 6043,
    "puntos_lealtad": 58827,
    "miembro_lealtad": 0,
    "devoluciones_365d": 20309,
    "reclamos_365d": 233446,
    "entrega_promedio_dias": 3498,
    "satisfaccion_promedio": 3648,
    "emails_abiertos_365d": 4951
  },
  "output_rows": 940000,
  "class_distribution": {
    "Ocasional": 368300,
    "Inactivo": 305513,
    "Frecuente": 116485,
    "Promocional": 86011,
    "Alto_valor": 63691
  }
}
```
