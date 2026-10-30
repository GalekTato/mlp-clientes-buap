# Manifiesto del Dataset

## Archivo de entrada

El proyecto opera sobre el archivo completo de clientes de comercio electrónico:

- **`1,000,000`** de registros de clientes.
- **`31`** columnas: `1` identificador (`id_cliente`), `29` variables predictivas (`23` numéricas y `6` categóricas) y `1` variable objetivo (`clase_cliente`).
- Codificación UTF-8, separador coma y encabezado en la primera línea.

## Clases objetivo (ventana prospectiva de 90 días)

`Inactivo` (`32.50%`), `Alto_valor` (`6.78%`), `Promocional` (`9.15%`), `Frecuente` (`12.39%`) y `Ocasional` (`39.18%`).

## Reglas de calidad verificadas en `preprocesamiento.py`

1. **Limpieza monetaria determinista:** rescate de `7,562` registros en `ticket_promedio_mxn` con caracteres `$`, comas o sufijo `MXN`.
2. **Normalización léxica categórica:** rescate de `22,438` registros con variantes de acentuación, capitalización o espacios en `region` (`7,517`), `canal_preferido` (`7,469`) y `metodo_pago_preferido` (`7,452`).
3. **Identidad de gasto anual:** recálculo determinista de `gasto_total_365d_mxn = round(compras_365d * ticket_promedio_mxn, 2)` en `30,000` registros (`1,034` con `compras_365d = 0` y `28,966` con `compras_365d > 0`).
4. **Filtrado de registros irrecuperables (`60,000` filas):**
   - `30,000` registros con valores nulos reales en `edad`, `region`, `ticket_promedio_mxn`, `paginas_por_sesion`, `duracion_sesion_min`, `satisfaccion_promedio` o `metodo_pago_preferido`.
   - `30,000` registros fuera de rango en `edad` (`[18, 80]`), `paginas_por_sesion` (`[1, 25]`) o `descuento_promedio_pct` (`[0, 65]`).
   - `0` registros duplicados y `0` violaciones de dependencias lógicas.

## Resultado de la auditoría

La ejecución sobre el millón de registros conserva exactamente **`940,000` registros limpios** en `datos/dataset_limpio.csv` y documenta el desglose completo en `resultados/auditoria_preprocesamiento.json` y `resultados/auditoria_preprocesamiento.md`.
