# Documentación técnica integral del proyecto

## 1. Descripción general

El proyecto implementa un sistema de clasificación multiclase para segmentar
clientes de comercio electrónico en cinco perfiles:

- `Inactivo`
- `Alto_valor`
- `Promocional`
- `Frecuente`
- `Ocasional`

El objetivo es estimar el comportamiento del cliente durante los siguientes
90 días a partir de variables históricas de los 365 días anteriores. La
solución integra auditoría de calidad, limpieza determinista, muestreo
estadístico, validación cruzada, ingeniería de variables, balanceo,
optimización de un `MLPClassifier`, calibración probabilística y evaluación
por lotes.

No se trata de componentes aislados: todos forman un único flujo reproducible.
La semilla principal es `42`, las transformaciones se ajustan dentro del
pipeline y las particiones se generan de forma estratificada.

## 2. Datos y significado de las variables

El archivo completo contiene 1,000,000 de registros y 31 columnas:

- `id_cliente`: identificador único, excluido del modelo.
- 29 variables predictivas numéricas y categóricas.
- `clase_cliente`: variable objetivo.

Las variables describen edad, región, zona, antigüedad, canal, dispositivo,
categoría preferida, compras, recencia, ticket, gasto, unidades, sesiones,
carritos, descuentos, correos, clics, puntos, devoluciones, reclamos, entrega,
satisfacción y método de pago.

### Reglas de negocio verificadas

1. Si `compras_365d == 0`, ticket, gasto, unidades, categorías, descuentos y
   devoluciones deben ser cero, y `dias_ultima_compra` debe valer 366.
2. `gasto_total_365d_mxn` debe ser
   `round(compras_365d * ticket_promedio_mxn, 2)`.
3. Devoluciones y compras con descuento no superan el número de compras.
4. Categorías compradas no supera `min(6, compras_365d)`.
5. Las sesiones son mayores o iguales a las compras.
6. Los carritos abandonados no superan `sesiones - compras`.
7. Los correos abiertos no superan los correos enviados.

## 3. Auditoría y limpieza

`preprocesameinto.py` —nombre requerido por la rúbrica— y
`preprocesamiento.py` —alias legible— realizan la auditoría completa.

Se revisan:

- valores nulos;
- filas duplicadas e identificadores repetidos;
- tipos numéricos y categóricos;
- categorías fuera de dominio;
- límites de cada variable;
- relaciones lógicas entre columnas;
- outliers mediante rango intercuartílico;
- colinealidad entre compras, ticket y gasto.

La política de corrección es conservadora. Solo se corrigen valores derivados
cuyo valor correcto se puede reconstruir inequívocamente, como el gasto total.
Las filas con nulos críticos, categorías inválidas, rangos imposibles o
violaciones relacionales no reparables se eliminan y se contabilizan.

### Resultado de la auditoría completa

| Elemento | Resultado |
|---|---:|
| Registros auditados | 1,000,000 |
| Correcciones de gasto derivado | 28,966 |
| Registros descartados | 118,966 |
| Registros limpios | 881,034 |
| Correlación compras-gasto | 0.7558 |
| Correlación ticket-gasto | 0.7321 |

Los outliers válidos no se eliminan automáticamente. Esta decisión evita
descartar clientes legítimos de alto valor.

## 4. Muestreo y particiones

Se utiliza la fórmula de Cochran con corrección por población finita:

```text
n0 = Z² p (1-p) / e²
n  = n0 / (1 + (n0-1)/N)
```

Parámetros:

- `Z = 2.576`, confianza aproximada del 99 %;
- `p = 0.5`, máxima variabilidad;
- `e = 0.005`, error máximo permitido;
- `N = 881,034`, población limpia.

El tamaño calculado es 61,710 y la afijación proporcional redondeada produce
61,712 observaciones:

| Partición | Registros |
|---|---:|
| Train | 43,198 |
| Validation | 9,257 |
| Test | 9,257 |

La clase `Alto_valor` conserva 4,178 casos en la muestra total y 585 casos por
fold cuando se utiliza `k=5`.

## 5. Validación cruzada

`crossVal.py` usa `StratifiedKFold` y evalúa el pipeline MLP real con `k=3`,
`k=5` y `k=10`.

| k | Alto_valor mínimo por fold | F1 macro medio | Desv. estándar | Tiempo |
|---:|---:|---:|---:|---:|
| 3 | 975 | 0.2802 | 0.0054 | 15.73 s |
| 5 | 585 | 0.2769 | 0.0096 | 9.17 s |
| 10 | 292 | 0.2753 | 0.0104 | 19.56 s |

Se selecciona `k=5` como compromiso entre representación minoritaria, costo y
estabilidad. El valor no se elige solo por la media de una corrida: también se
consideran la cantidad de ejemplos por fold y la variabilidad.

## 6. Ingeniería de variables

`DomainFeatureTransformer` crea ocho variables dentro del pipeline:

```python
ratio_descuento = compras_con_descuento_365d / max(compras_365d, 1)
frecuencia_anualizada = compras_365d / (antiguedad_dias / 365.0)
intensidad_recencia = compras_365d / (dias_ultima_compra + 1.0)
unidades_por_compra = unidades_365d / max(compras_365d, 1)
tasa_abandono_carrito = carritos_abandonados_365d / max(sesiones_365d, 1)
tasa_apertura_emails = emails_abiertos_365d / max(emails_enviados_365d, 1)
tasa_devoluciones = devoluciones_365d / max(compras_365d, 1)
indice_valor_cliente = gasto_total_365d_mxn / (dias_ultima_compra + 1.0)
```

Estas variables representan recurrencia, intensidad, dependencia promocional,
interacción digital, devoluciones y valor monetario. Sus denominadores están
protegidos para evitar divisiones por cero.

La importancia por permutación en validación identificó como variables más
influyentes `descuento_promedio_pct` (0.0347), `sesiones_365d` (0.0205),
`frecuencia_anualizada` (0.0092), `unidades_por_compra` (0.0077) y
`tasa_abandono_carrito` (0.0071).

## 7. Arquitectura del modelo

El pipeline ejecuta:

1. `DomainFeatureTransformer`.
2. `ColumnTransformer`.
3. `StandardScaler` para las variables numéricas.
4. `OneHotEncoder(handle_unknown="ignore")` para las categóricas.
5. `MLPClassifier`.

La búsqueda considera:

- número de neuronas y capas ocultas;
- `max_iter`;
- activación `relu` o `tanh`;
- `alpha` como regularización L2;
- `batch_size`;
- `learning_rate_init`;
- learning rate constante o adaptativa;
- solvers Adam y SGD;
- momentum y Nesterov;
- `early_stopping`;
- `validation_fraction`;
- `n_iter_no_change`;
- `tol`.

`GridSearchCV` selecciona con `F1-Macro`. La ejecución usa caché de
transformaciones, paralelismo `loky` y limitación de hilos BLAS/OpenMP.

## 8. Balanceo dentro de los folds

La comparación se realiza sin contaminar validación:

- **Distribución original:** no modifica el fold de entrenamiento.
- **Ponderación:** calcula pesos con `compute_class_weight` usando solo el
  fold de entrenamiento y aplica `gamma=0.75`.
- **Sobremuestreo:** usa `RandomOverSampler` exclusivamente sobre el fold de
  entrenamiento.

Resultados medios en cinco folds:

| Esquema | F1 macro | Balanced accuracy | Recall Frecuente | Recall Promocional |
|---|---:|---:|---:|---:|
| Distribución original | 0.2769 | 0.2744 | 0.0359 | 0.1176 |
| Ponderación | 0.3122 | 0.3480 | 0.2257 | 0.3989 |
| Sobremuestreo | 0.2614 | 0.3267 | 0.3132 | 0.4034 |

La ponderación ofrece el mejor compromiso de F1 macro y balanced accuracy.
El sobremuestreo captura más casos de `Frecuente` y `Promocional`, pero
reduce el F1 macro global.

## 9. Resultados del modelo seleccionado

| Partición | F1 macro | Balanced accuracy | F1 weighted | ROC-AUC OvR |
|---|---:|---:|---:|---:|
| Train | 0.3429 | 0.3760 | 0.3821 | 0.6997 |
| Validation | 0.3185 | 0.3499 | 0.3600 | 0.6856 |
| Test | 0.3276 | 0.3574 | 0.3751 | 0.6895 |

La brecha train-test de F1 macro es 0.0153. Esta diferencia es reducida y no
indica sobreajuste severo bajo este protocolo.

## 10. Calibración y umbrales

Se comparan calibración `sigmoid` e `isotonic`. La isotónica obtuvo:

- F1 macro ajustado en validación: **0.2751**;
- log-loss: **1.3133**;
- Brier multiclase: **0.6853**;
- ECE: **0.0043**.

Los multiplicadores de probabilidad se ajustan únicamente con validación:
`[1.5, 1.5, 1.1, 0.7, 1.5]`.

## 11. Evaluación del archivo completo

La inferencia se ejecuta por chunks para no cargar simultáneamente todas las
filas en memoria. Se procesaron:

- 970,000 filas válidas;
- 30,000 filas omitidas por campos no convertibles;
- throughput aproximado: 19,309 registros por segundo;
- F1 macro: 0.3219;
- balanced accuracy: 0.3523;
- log-loss: 1.4139.

La evaluación por lotes es una medida de comportamiento global sobre el archivo
completo, mientras que las métricas de test se calculan sobre una partición
estratificada independiente.

## 12. Gráficas y significado

Las figuras se generan con una paleta común entre Python y LaTeX:

| Color | Hexadecimal | Clase o función |
|---|---|---|
| `seupurple` | `#8E44AD` | `Alto_valor`, títulos |
| `lachaiseblue` | `#2980B9` | `Frecuente` |
| `seublue` | `#3498DB` | `Ocasional` |
| `seuteal` | `#16A085` | `Promocional` |
| `seuamber` | `#D35400` | `Inactivo` |

Figuras principales:

- `01_distribucion_clases_cochran`: conserva la proporción de clases.
- `02_analisis_k_crossval`: muestra representación minoritaria, F1 y costo.
- `03_evolucion_rejilla_paralela`: compara configuraciones.
- `04_curvas_aprendizaje_convergencia`: permite observar convergencia.
- `05_impacto_balanceo_frecuente_promocional`: muestra el efecto del balanceo.
- `06_curvas_calibracion_probabilidades`: compara calidad probabilística.
- `07_variacion_estadistica_boxplots`: muestra estabilidad entre folds.
- `08_matrices_confusion_test_y_archivo`: muestra errores por clase.
- `09_curvas_roc_pr_multiclase`: contiene curvas ROC y Precision-Recall.
- `10_auc_area_multiclase`: representa el área bajo las curvas ROC y PR por
  clase. Esta es la gráfica de AUC/área bajo la curva incorporada al reporte.

## 13. Reproducibilidad

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python preprocesameinto.py
python muestreo.py
python crossVal.py --n-jobs 2
python entrenamiento.py --balance weighted --calibrate --n-jobs 2
python main.py --skip-clean --balance weighted --calibrate --large-validation
python -m src.generate_figures
python -m src.generate_extra_figures
python empaquetar_entrega.py
```

La ejecución rápida usa `--fast`; los resultados finales deben distinguirse de
una comprobación rápida y conservar los archivos de configuración y métricas
generados.
