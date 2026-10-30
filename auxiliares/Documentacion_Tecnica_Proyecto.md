# Documentación Técnica Integral del Proyecto

## 1. Descripción general

El proyecto implementa un sistema de clasificación supervisada multiclase para segmentar clientes de comercio electrónico en cinco perfiles futuros durante los 90 días posteriores al corte del 30 de junio de 2026, a partir de su actividad histórica de los 365 días previos:

- `Inactivo` (`32.50%`)
- `Alto_valor` (`6.78%`)
- `Promocional` (`9.15%`)
- `Frecuente` (`12.39%`)
- `Ocasional` (`39.18%`)

La solución articula en un único flujo reproducible (`SEED = 42`): auditoría y limpieza determinista (`preprocesamiento.py`), muestreo estratificado de Cochran (`muestreo.py`), selección de pliegues de validación cruzada (`crossVal.py`), selección de ponderación por clase y búsqueda exhaustiva de hiperparámetros para un Perceptrón Multicapa (`entrenamiento.py`), comparación con familias de referencia (`modelos_referencia.py`), calibración probabilística con `FrozenEstimator` (`src/calibration.py`) y evaluación masiva por lotes (`src/evaluation_large.py`).

## 2. Datos y reglas de integridad del dominio

El archivo de entrada consta de `1,000,000` de registros y `31` columnas:

- `id_cliente`: identificador único, excluido del entrenamiento.
- `29` variables predictivas (`23` numéricas y `6` categóricas).
- `clase_cliente`: variable objetivo multiclase.

### Reglas de negocio verificadas en `preprocesamiento.py`

1. Cuando `compras_365d == 0`, entonces `ticket_promedio_mxn`, `gasto_total_365d_mxn`, `unidades_365d`, `categorias_compradas_365d`, `compras_con_descuento_365d`, `descuento_promedio_pct` y `devoluciones_365d` valen `0`, y `dias_ultima_compra == 366`.
2. Identidad determinista de gasto anual: `gasto_total_365d_mxn == round(compras_365d * ticket_promedio_mxn, 2)`.
3. `devoluciones_365d <= compras_365d` y `compras_con_descuento_365d <= compras_365d`.
4. `categorias_compradas_365d <= min(6, compras_365d)`.
5. `sesiones_365d >= compras_365d` y `carritos_abandonados_365d <= sesiones_365d - compras_365d`.
6. `emails_abiertos_365d <= emails_enviados_365d`.

## 3. Preprocesamiento y limpieza (`preprocesamiento.py`)

El preprocesamiento distingue entre **errores de formato reparables determinísticamente** y **observaciones irrecuperables**:

1. **Rescate de formato monetario (`7,562` registros):** limpieza de símbolos `$`, comas, espacios y sufijo `MXN` en `ticket_promedio_mxn`.
2. **Normalización léxica categórica (`22,438` registros):** unificación de acentos, mayúsculas/minúsculas y separadores hacia las etiquetas canónicas del diccionario (`region`: `7,517`; `canal_preferido`: `7,469`; `metodo_pago_preferido`: `7,452`, incluyendo `"Cartera digital"` → `"Cartera_digital"`).
3. **Recálculo determinista de gasto (`30,000` registros):** aplicación de `round(compras_365d * ticket_promedio_mxn, 2)` en las `30,000` filas con discrepancia aritmética (`1,034` con `compras_365d = 0` y `28,966` con `compras_365d > 0`).
4. **Eliminación por causa excluyente (`60,000` registros):**
   - `nulos`: `30,000` filas con valores faltantes reales en 7 atributos.
   - `fuera_de_rango`: `30,000` filas fuera de los límites del diccionario (`edad`: `10,026`; `paginas_por_sesion`: `9,935`; `descuento_promedio_pct`: `10,039`).
   - `dependencias`: `0` filas.
   - `duplicados`: `0` filas.

| Métrica de auditoría | Valor |
|---|---:|
| Registros de entrada (`input_rows`) | `1,000,000` |
| Registros reparados y conservados | `60,000` |
| Registros eliminados (`removed_rows`) | `60,000` |
| Registros limpios conservados (`output_rows`) | `940,000` |
| Correlación `compras_365d` vs. `gasto_total_365d_mxn` | `0.7558` |
| Correlación `ticket_promedio_mxn` vs. `gasto_total_365d_mxn` | `0.7318` |

Los valores extremos según IQR se conservan dentro de los rangos válidos para no eliminar a los clientes del segmento `Alto_valor`.

## 4. Muestreo de Cochran y particiones (`muestreo.py`)

Aplicando la fórmula de Cochran con corrección por población finita sobre `N = 940,000`, con `Z = 2.576` (confianza del 99%), `p = 0.5` y margen de error `e = 0.005`:

- Tamaño mínimo con corrección por población finita (`cochran_fpc`): **`61,983`** registros.
- Muestra estratificada por afijación proporcional (`actual_sample_size`): **`61,985`** registros.

| Partición | Proporción | Registros totales | Casos de `Alto_valor` |
|---|---:|---:|---:|
| `train.csv` | 70% | `43,389` | `2,940` |
| `validation.csv` | 15% | `9,298` | `630` |
| `test.csv` | 15% | `9,298` | `630` |

Las pruebas de bondad de ajuste $\chi^2$ entre la población limpia y la muestra entregan $p \ge 0.9966$ en las cinco clases, y las pruebas de Kolmogorov–Smirnov arrojan $p > 0.05$ en las 23 variables numéricas.

## 5. Validación cruzada estratificada (`crossVal.py`)

Se evaluaron $k \in \{3, 5, 10\}$ pliegues sobre `train.csv` (`43,389` registros) bajo la regla de decisión: **elegir el mayor $k$ que garantice al menos `500` casos de validación de la clase minoritaria `Alto_valor` por pliegue**.

| $k$ | `Alto_valor` por fold | F1-Macro medio | Desv. estándar | Error estándar | Tiempo |
|---:|---:|---:|---:|---:|---:|
| `3` | `980` | `0.2740` | `0.0054` | `0.0031` | `5.86 s` |
| **`5`** | **`588`** | **`0.2675`** | **`0.0143`** | **`0.0064`** | **`5.78 s`** |
| `10` | `294` (`< 500`) | `0.2732` | `0.0098` | `0.0031` | `5.95 s` |

Se selecciona **`k = 5`**, preservando `588` observaciones de `Alto_valor` en cada pliegue de validación.

## 6. Ingeniería de variables e importancia por permutación

`DomainFeatureTransformer` construye 8 razones comerciales dentro del pipeline: `ratio_descuento`, `frecuencia_anualizada`, `intensidad_recencia`, `unidades_por_compra`, `tasa_abandono_carrito`, `tasa_apertura_emails`, `tasa_devoluciones` e `indice_valor_cliente`.

La importancia por permutación (`n_repeats = 10` sobre `validation.csv`, permutando las 29 variables originales a la entrada del pipeline completo) identificó que la señal se concentra en:

1. `sesiones_365d` (`+0.0389 ± 0.0034`)
2. `descuento_promedio_pct` (`+0.0345 ± 0.0025`)
3. `compras_365d` (`+0.0212 ± 0.0030`)
4. `ticket_promedio_mxn` (`+0.0044 ± 0.0017`)
5. `antiguedad_dias` (`+0.0033 ± 0.0011`)

## 7. Selección de $\gamma^*$ y búsqueda en rejilla del MLP (`entrenamiento.py`)

1. **Barrido de $\gamma \in \{0.00, 0.25, 0.50, 0.75\}$ en 5-fold CV sobre `train.csv`:** mediante la regla 1-SE se seleccionó **`γ* = 0.50`** (`F1-Macro CV = 0.3249 ± 0.0017`, `Balanced Accuracy = 0.3372`, `Accuracy = 0.3914`).
2. **Rejilla exhaustiva (`472` combinaciones = `216` con `adam` + `256` con `sgd`, `2,360` ajustes totales, `0` fallos):**
   - Configuración ganadora: `hidden_layer_sizes = (128, 64)`, `solver = 'sgd'`, `momentum = 0.95`, `nesterovs_momentum = True`, `activation = 'tanh'`, `alpha = 0.001`, `learning_rate = 'constant'`, `learning_rate_init = 0.001`, `batch_size = 256`, `max_iter = 150` (convergencia por parada temprana en la época `29`).
   - Mejor `F1-Macro` medio en CV (5 folds): **`0.3349 ± 0.0029`** (`IC 95%: [0.3313, 0.3385]`).

## 8. Comparación de balanceo, particiones y variación estadística

### Comparación de esquemas de balanceo dentro de cada fold (`k = 5`)

| Esquema | F1-Macro medio | Balanced Accuracy | Recall `Frecuente` | Recall `Promocional` |
|---|---:|---:|---:|---:|
| Sin ponderación (`baseline`) | `0.2816` | `0.2775` | `0.0684` | `0.0970` |
| **Ponderación suave (`γ = 0.50`)** | **`0.3349`** | **`0.3391`** | **`0.2925`** | **`0.4020`** |
| Sobremuestreo aleatorio (`RandomOverSampler`) | `0.2597` | `0.3282` | `0.3121` | `0.4063` |

### Desempeño del modelo ganador por partición e IC 95% Bootstrap en `test` (`B = 1,000`)

| Partición | Accuracy | Balanced Accuracy | F1-Macro | F1-Weighted | ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| `train` (`43,389`) | `0.3957` | `0.3532` | `0.3435` | `0.3935` | `0.6946` |
| `validation` (`9,298`) | `0.3885` | `0.3442` | `0.3340` | `0.3865` | `0.6865` |
| **`test` (`9,298`)** | **`0.3914`** | **`0.3503`** | **`0.3383`** | **`0.3891`** | **`0.6901`** |
| `test` (IC 95% Bootstrap) | `[0.3822, 0.4014]` | `[0.3395, 0.3619]` | `[0.3279, 0.3494]` | --- | `[0.6830, 0.6971]` |
| `test` con multiplicadores post-hoc | `0.3926` | `0.3526` | **`0.3457`** | `0.3908` | `0.6901` |

La brecha `Train - Test` en `F1-Macro` es de apenas **`0.0052`**, confirmando ausencia de sobreajuste.

## 9. Calibración probabilística y evaluación masiva por lotes

- **Calibración post-hoc (`FrozenEstimator` sobre `validation.csv`, evaluada en `test.csv`):**
  - `uncalibrated`: `log_loss = 1.3251`, `brier = 0.7279`, `ece = 0.0566` (`F1-Macro = 0.3457` con multiplicadores `[0.9, 1.3, 1.2, 1.0, 1.2]`).
  - `sigmoid`: `log_loss = 1.2829`, `brier = 0.7089`, `ece = 0.0320`.
  - `isotonic`: `log_loss = 1.2773`, `brier = 0.7060`, `ece = 0.0252`.
- **Evaluación masiva por lotes (`src/evaluation_large.py`):**
  - Del millón de registros crudos, se descartan `60,000` por limpieza y se excluyen los `43,389` IDs de `train.csv`, evaluando exactamente **`896,611` registros limpios fuera de entrenamiento** (`60,000 + 43,389 + 896,611 = 1,000,000`).
  - Velocidad de procesamiento: **`40,321.3` filas/segundo** (`22.24 s`).
  - Desempeño global fuera de entrenamiento: `Accuracy = 0.3878`, `Balanced Accuracy = 0.3463`, `F1-Macro = 0.3348`, `F1-Weighted = 0.3858`, `ROC-AUC OvR = 0.6881`.
