# Documentación técnica del sistema de clasificación multiclase

## 1. Propósito

El proyecto construye un sistema reproducible de clasificación multiclase para segmentar clientes de comercio electrónico en cinco perfiles: `Inactivo`, `Alto_valor`, `Promocional`, `Frecuente` y `Ocasional`. La implementación usa Python, pandas, NumPy, SciPy, scikit-learn, imbalanced-learn, matplotlib, seaborn y joblib.

El archivo disponible en `datos/` contiene 31 columnas: un identificador, 29 variables predictivas y `clase_cliente`. La unidad de análisis es un cliente y los atributos históricos resumen el comportamiento de los 365 días previos al corte.

## 2. Comprensión del dataset

### Variables y reglas

- `id_cliente`: identificador; se excluye del modelo.
- Variables demográficas y de preferencia: `edad`, `region`, `zona`, `canal_preferido`, `dispositivo_preferido`, `categoria_preferida` y `metodo_pago_preferido`.
- Variables transaccionales: compras, días desde la última compra, ticket, gasto, unidades, categorías, sesiones y carritos.
- Variables de comunicación y fidelidad: correos, clics, puntos, membresía, devoluciones, reclamos, entrega y satisfacción.
- `clase_cliente`: objetivo multiclase.

Las relaciones de dominio más importantes son:

1. Si `compras_365d == 0`, el ticket, gasto, unidades, categorías, descuentos y devoluciones deben ser cero y `dias_ultima_compra` debe ser 366.
2. `gasto_total_365d_mxn` debe ser `round(compras_365d * ticket_promedio_mxn, 2)`.
3. Devoluciones y compras con descuento no pueden superar compras.
4. Categorías compradas no supera `min(6, compras_365d)`.
5. Sesiones deben ser al menos las compras.
6. Carritos abandonados no puede superar `sesiones - compras`.
7. Correos abiertos no puede superar correos enviados.

## 3. Flujo de ejecución

```text
CSV -> auditoría y limpieza -> muestra estratificada -> train/validation/test
    -> variables de dominio -> ColumnTransformer -> MLPClassifier
    -> búsqueda de hiperparámetros -> balanceo/calibración -> evaluación
    -> tablas, gráficas, modelo serializado y reporte PDF
```

### `preprocesameinto.py` y `preprocesamiento.py`

El primer archivo conserva el nombre solicitado por la rúbrica. El segundo es un alias limpio. Ambos:

- leen desde la raíz o desde `datos/`;
- convierten columnas numéricas y categóricas;
- auditan nulos, duplicados, dominios, rangos y relaciones;
- corrigen variables derivadas cuando la regla es inequívoca;
- eliminan filas que no pueden repararse de forma técnicamente defendible;
- escriben `datos/dataset_limpio.csv` y reportes en `resultados/`.

Resultado observado:

| Elemento | Conteo |
|---|---:|
| Filas auditadas | 1,000,000 |
| Correcciones deterministas de gasto | 4,978 |
| Filas descartadas | 20,466 |
| Filas limpias | 881,034 |

Los valores extremos se reportan mediante el criterio IQR, pero no se eliminan automáticamente si cumplen el dominio. Esto evita eliminar clientes legítimos de alto valor.

### `muestreo.py`

Calcula el tamaño de muestra con Cochran corregido por población finita:

```text
n0 = Z² p (1-p) / e²
n  = n0 / (1 + (n0-1)/N)
```

Se usan `Z=2.576`, `p=0.5` y `e=0.005`. La muestra se distribuye proporcionalmente por `clase_cliente` y se divide en 70% entrenamiento, 15% validación y 15% prueba con `random_state=42`.

Archivos producidos:

- `datos/train.csv`
- `datos/validation.csv`
- `datos/test.csv`
- `resultados/calculo_muestra.json`
- `resultados/distribucion_particiones.csv`

### `crossVal.py`

Expone `get_stratified_cv(k)` y compara `k=3`, `k=5` y `k=10` con `StratifiedKFold` sobre el pipeline MLP real. Los resultados fueron F1 macro medio de 0.2713, 0.2635 y 0.2696, respectivamente. Se elige `k=5` por el equilibrio entre estabilidad estadística, 433 casos de `Alto_valor` por fold y costo computacional.

### `src/feature_engineering.py`

Implementa `DomainFeatureTransformer`, compatible con scikit-learn. Las variables son:

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

El transformer se ejecuta dentro del pipeline, antes del `ColumnTransformer`, por lo que cada fold calcula sus transformaciones sin usar información de validación.

### `entrenamiento.py`

El pipeline está compuesto por:

1. `DomainFeatureTransformer`.
2. `ColumnTransformer`.
3. `StandardScaler` para variables numéricas.
4. `OneHotEncoder(handle_unknown="ignore")` para categóricas.
5. `MLPClassifier`.

La rejilla considera arquitectura, activación, `alpha`, `batch_size`, `max_iter`, tasa de aprendizaje, solver, momentum y early stopping. La métrica principal es `f1_macro`; también se registran `f1_weighted`, `balanced_accuracy`, precisión, recall y `roc_auc_ovr`.

La búsqueda usa `GridSearchCV`, caché `joblib.Memory`, backend `loky` y `threadpool_limits` para evitar sobresuscripción de hilos. El modelo entrenado se guarda en `modelos/`.

### Balanceo

`src/balancing.py` contiene:

- baseline con distribución original;
- pesos balanceados controlados por `gamma`;
- `RandomOverSampler` y `SMOTE` cuando `imbalanced-learn` está instalado;
- fallback de sobremuestreo estratificado.

La comparación localizada por fold obtuvo:

| Esquema | F1 macro medio | Balanced accuracy media | Recall Frecuente | Recall Promocional |
|---|---:|---:|---:|---:|
| Baseline | 0.2635 | 0.2651 | 0.0311 | 0.0901 |
| Ponderado, gamma=0.75 | 0.3171 | 0.3564 | 0.2189 | 0.3975 |
| Sobremuestreo aleatorio | 0.2555 | 0.3279 | 0.3113 | 0.4097 |

En la evaluación final de prueba, el esquema ponderado obtuvo F1 macro 0.3276, balanced accuracy 0.3574 y F1 weighted 0.3751.

La ponderación ofrece el mejor compromiso global; el sobremuestreo aumenta el recall de las clases objetivo, pero reduce F1 global. Los tres esquemas se ajustan únicamente con el subconjunto de entrenamiento de cada fold.

### Calibración y umbrales

`src/calibration.py` compara:

- calibración `sigmoid` mediante Platt scaling;
- calibración `isotonic`;
- Log-Loss;
- Brier multiclase;
- Expected Calibration Error;
- multiplicadores por clase ajustados únicamente con validación.

La calibración isotónica alcanzó un `F1 macro` ajustado de 0.2751 en validación. Los multiplicadores aprendidos fueron `[1.5, 1.5, 1.1, 0.7, 1.5]` en el orden de clases del estimador.

### Evaluación por chunks

`src/evaluation_large.py` permite inferir archivos grandes en bloques. Convierte importes con símbolos monetarios, descarta filas no convertibles de forma explícita y calcula throughput, accuracy, F1, balanced accuracy, ROC-AUC, log-loss y matriz de confusión.

Sobre el archivo completo se procesaron 970,000 filas válidas, se omitieron 30,000 con campos no convertibles y se midieron aproximadamente 19,309 registros por segundo. El F1 macro fue 0.3219, la balanced accuracy 0.3523 y el log-loss 1.4139.

## 4. Gráficas

`src/plotting_theme.py` centraliza la paleta:

| Nombre | Hexadecimal | Uso |
|---|---|---|
| `seupurple` | `#8E44AD` | títulos y Alto_valor |
| `lachaiseblue` | `#2980B9` | subtítulos y Frecuente |
| `seublue` | `#3498DB` | acentos y Ocasional |
| `seuteal` | `#16A085` | métricas y Promocional |
| `seuamber` | `#D35400` | alertas e Inactivo |
| `seugray` | `#7F8C8D` | texto secundario |
| `seubg` | `#F0F8FF` | fondos |

Las nueve familias se encuentran en `graficas/` en PNG y PDF a 300 DPI.

## 5. Ejecución

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python preprocesameinto.py
python muestreo.py
python crossVal.py
python entrenamiento.py --balance weighted --calibrate
python main.py --fast --balance weighted --calibrate
python empaquetar_entrega.py
```

Para usar rutas alternativas:

```powershell
python entrenamiento.py --train datos\train.csv --validation datos\validation.csv --test datos\test.csv --report-dir resultados --graph-dir graficas
```

## 6. Interpretación final

El modelo no debe evaluarse solo con accuracy. El objetivo es capturar de manera equilibrada los cinco perfiles. La ponderación mejora la sensibilidad global a clases con menos observaciones; la calibración permite interpretar mejor las probabilidades; y la ingeniería de variables hace explícitas las relaciones comerciales que el MLP tendría que aprender indirectamente.
