# Documentación Técnica Paso a Paso, Justificación de la Precisión y Guía de Interpretación de Resultados

Este documento explica de manera exhaustiva, paso a paso, cómo está construido el proyecto de clasificación multiclase de clientes con un Perceptrón Multicapa (`MLPClassifier`), qué hace cada módulo de código, por qué se tomó cada decisión metodológica, **por qué las métricas de exactitud (`Accuracy`) y `F1-Macro` toman los valores observados** y **cómo deben interpretarse técnica y comercialmente todos los resultados**.

---

## 1. Planteamiento del Problema y Naturaleza de la Variable Objetivo

### 1.1 ¿Qué se predice exactamente?
El dataset `clientes_comercio_1000000_alumnos (1).csv` contiene **`1,000,000` de registros y `31` columnas**:
- **`1` columna identificadora (`id_cliente`):** clave única por cliente (no predictiva).
- **`29` variables predictivas de entrada ($X$):** `23` numéricas y `6` categóricas que resumen el comportamiento histórico del cliente durante los **365 días previos a la fecha de corte** (`30 de junio de 2026`).
- **`1` variable objetivo (`clase_cliente`, $y$):** perfil de comportamiento real del cliente durante los **90 días posteriores al corte** (`1 de julio al 28 de septiembre de 2026`).

### 1.2 Las 5 clases y su jerarquía de prioridad en la ventana futura de 90 días
De acuerdo con el documento de especificación técnica del dataset, la etiqueta `clase_cliente` se asigna evaluando tres cantidades medidas en los **90 días posteriores al corte**:
- $N$: número de compras realizadas en los 90 días futuros.
- $D$: número de compras con descuento realizadas en los 90 días futuros.
- $G$: gasto total acumulado (MXN) en los 90 días futuros.

La asignación sigue un **orden estricto de prioridad excluyente** (si un cliente cumple la Prioridad 1 se asigna a ella; si no, se evalúa la Prioridad 2, y así sucesivamente):

| Prioridad | Clase (`clase_cliente`) | Proporción limpia (`N = 940,000`) | Regla de asignación en los 90 días futuros |Significado comercial |
|:---:|---|---:|---|---|
| **1** | `Inactivo` | `32.5014%` (`305,513`) | $N = 0$ | Clientes que no realizan ninguna compra en el trimestre siguiente. |
| **2** | `Alto_valor` | `6.7756%` (`63,691`) | $N \ge 1$ y ($G \ge \$14,000$ o [$G \ge \$10,500$ y $N \ge 7$]) | Clientes de mayor facturación trimestral; segmento crítico de retención VIP. |
| **3** | `Promocional` | `9.1501%` (`86,011`) | No es `Alto_valor`, $N \ge 1$ y ([$D \ge 2$ y $D/N \ge 0.60$] o [$D \ge 4$ y $\text{desc\_hist} \ge 20\%$]) | Clientes cuya actividad depende fuertemente de rebajas y cupones. |
| **4** | `Frecuente` | `12.3920%` (`116,485`) | No cumple las anteriores, $N \ge 5$, $D/N < 0.60$ y $G \ge \$3,200$ | Clientes con compra recurrente orgánica sin dependencia de descuentos. |
| **5** | `Ocasional` | `39.1809%` (`368,300`) | Cualquier cliente activo ($N \ge 1$) que no cumpla las prioridades 2, 3 o 4 | Categoría residual intermedia (compras esporádicas o de bajo monto). |

### 1.3 El límite teórico de predictibilidad (Error Irreducible o Techo de Bayes)
Comprender la tabla anterior es indispensable para interpretar cualquier modelo entrenado sobre esta base:
1. **Las variables futuras $(N, D, G)$ de los siguientes 90 días NO están en el CSV.** El modelo únicamente conoce el resumen acumulado de los 365 días pasados.
2. **La ocurrencia de compras en una ventana corta futura (90 días) es un proceso estocástico.** Por ejemplo, un cliente que compró 3 veces en el último año ($\text{tasa media} = 0.75\text{ compras por trimestre}$) tiene, bajo un proceso de Poisson homogéneo, una probabilidad $P(N=0) = e^{-0.75} \approx 47.2\%$ de caer en `Inactivo` y una probabilidad $P(N \ge 1) \approx 52.8\%$ de caer en `Ocasional` en el trimestre siguiente.
3. **Dos clientes con el mismo vector histórico $X$ pueden tener etiquetas verdaderas distintas en los 90 días futuros.** En teoría de aprendizaje estadístico, esto significa que la distribución condicional $P(Y \mid X = \mathbf{x})$ no es determinista ($\max_c P(Y = c \mid X = \mathbf{x}) < 1$), lo que define una **tasa de error irreducible de Bayes ($\varepsilon_{\text{Bayes}}$)** que ningún algoritmo —por complejo que sea— puede superar sin caer en sobreajuste.

---

## 2. Explicación Paso a Paso de los Módulos del Proyecto

### Paso 1: Preprocesamiento y Auditoría de Calidad (`preprocesamiento.py`)

#### ¿Qué problema resuelve?
El archivo crudo de `1,000,000` de filas contiene **cuatro bloques disjuntos de `30,000` registros cada uno** con anomalías inyectadas (`120,000` filas afectadas en total). Un preprocesamiento ingenuo que elimine cualquier fila con texto no numérico o cadenas mal escritas descartaría `118,966` o `120,000` filas, perdiendo innecesariamente `60,000` registros cuyos valores son 100% recuperables de forma exacta y determinista.

#### ¿Cómo funciona paso a paso?
1. **Lectura tipada y rescate de formato monetario (`read_dataset` y `clean_numeric_text` en `proyecto_utils.py`):**
   - En `ticket_promedio_mxn`, exactamente **`7,562` registros** vienen formateados como cadenas monetarias (por ejemplo, `"$1,245.50 MXN"`).
   - Si se aplicara `pd.to_numeric(..., errors="coerce")` directamente, esas `7,562` celdas se convertirían en falsos `NaN`.
   - `clean_numeric_text` limpia mediante expresiones regulares los símbolos `$`, las comas `,`, los espacios y el sufijo `MXN` antes de convertir a flotante, **rescatando íntegramente las `7,562` filas**.
2. **Normalización léxica de variables categóricas (`normalize_categorical_series`):**
   - Exactamente **`22,438` registros** contienen variantes de escritura en tres columnas categóricas:
     - `region`: **`7,517`** casos (ej. `"norte"`, `"SUR "`, `"centro"`).
     - `canal_preferido`: **`7,469`** casos (ej. `"web"`, `"APP"`, `"tienda"`).
     - `metodo_pago_preferido`: **`7,452`** casos (ej. `"Cartera digital"` con espacio en vez de guion bajo, `"tarjeta"`, `"efectivo"`).
   - La función `_clave_canonica` descompone acentos (`unicodedata.normalize("NFD", ...)`), pasa a minúsculas, recorta espacios y une palabras con `_`, mapeando cada variante a su etiqueta canónica oficial del diccionario (`DOMAINS`). **Se rescatan las `22,438` filas** (`7,562 + 22,438 = 30,000` filas del primer bloque reparadas sin pérdida).
3. **Recálculo determinista de la identidad de gasto anual (`derived_gasto = 30,000`):**
   - El diccionario de datos establece la ley exacta:
     $$\text{gasto\_total\_365d\_mxn} = \operatorname{round}(\text{compras\_365d} \times \text{ticket\_promedio\_mxn},\, 2)$$
   - Se detectan **`30,000` registros** donde `gasto_total_365d_mxn` difiere de ese producto en más de `$0.05` MXN (`1,034` filas donde `compras_365d == 0` pero el gasto registrado era positivo, y `28,966` filas con `compras_365d > 0` y gasto corrompido).
   - Como `compras_365d` y `ticket_promedio_mxn` están presentes y dentro de rango válido en esas `30,000` filas, `preprocesamiento.py` recalcula `gasto_total_365d_mxn` aplicando la fórmula exacta **antes** de evaluar las reglas de eliminación, **rescatando las `30,000` filas**.
4. **Eliminación de registros verdaderamente irrecuperables con atribución por causa excluyente (`removed_rows = 60,000`):**
   - **Causa 1 — `nulos` (`30,000` filas eliminadas):** registros con valores `NaN` reales distribuidos en 7 columnas (`duracion_sesion_min`: `4,349`; `metodo_pago_preferido`: `4,331`; `edad`: `4,297`; `ticket_promedio_mxn`: `4,287`; `paginas_por_sesion`: `4,264`; `satisfaccion_promedio`: `4,238`; `region`: `4,234`). No existe ninguna ecuación determinista que permita reconstruir la edad, región o satisfacción de un cliente a partir de otras columnas; inventarlos con la media o moda antes del particionado contaminaría la distribución real.
   - **Causa 2 — `fuera_de_rango` (`30,000` filas eliminadas):** registros con valores físicamente imposibles o fuera de las cotas del diccionario: `edad` $\notin [18, 80]$ (`10,026` filas), `paginas_por_sesion` $\notin [1, 25]$ (`9,935` filas) y `descuento_promedio_pct` $\notin [0, 65]$ (`10,039` filas).
   - **Causa 3 — `dependencias` (`0` filas):** una vez reparado el gasto derivado, `0` filas violan las restricciones relacionales (`compras_con_descuento <= compras`, `devoluciones <= compras`, `emails_abiertos <= emails_enviados`, etc.).
   - **Causa 4 — `duplicados` (`0` filas):** `0` filas duplicadas y `0` `id_cliente` repetidos.
5. **Conservación deliberada de *outliers* estadísticos (IQR):**
   - En los `940,000` registros limpios existen `57,806` observaciones que superan $Q_3 + 1.5\,\text{IQR}$ en `gasto_total_365d_mxn`. **No se eliminan**, porque no son errores de captura sino los compradores intensivos que conforman la clase `Alto_valor` (`63,691` clientes).

---

### Paso 2: Muestreo Estadístico de Cochran y Particionado Estratificado (`muestreo.py`)

#### ¿Por qué se aplica muestreo y cómo se calcula?
Entrenar `2,360` redes neuronales (472 combinaciones $\times$ 5 folds) sobre `940,000` filas requeriría decenas de horas de CPU sin mejorar la capacidad de generalización (como lo demuestra la curva de aprendizaje por tamaño de muestra en la Sección 3.3). Para dimensionar rigurosamente el subconjunto representativo sobre la población finita limpia $N = 940,000$, se utiliza la **fórmula de Cochran con corrección por población finita (FPC)**:

$$n_0 = \frac{Z^2 \cdot p(1 - p)}{e^2} = \frac{(2.576)^2 \cdot 0.5 \cdot 0.5}{(0.005)^2} = 66,357.76$$

$$n_{\text{FPC}} = \left\lceil \frac{n_0}{1 + \dfrac{n_0 - 1}{N}} \right\rceil = \left\lceil \frac{66,357.76}{1 + \dfrac{66,356.76}{940,000}} \right\rceil = \mathbf{61,983}$$

Donde:
- $Z = 2.576$ corresponde a un **nivel de confianza del 99%**.
- $p = 0.5$ es el supuesto de **máxima varianza posible** ($p(1-p) = 0.25$), que garantiza el tamaño muestral más conservador para cualquier proporción.
- $e = 0.005$ es un **margen de error máximo de $\pm 0.5\%$**.

Al asignar proporcionalmente por estrato con redondeo hacia arriba ($n_c = \lceil n_{\text{FPC}} \cdot N_c / N \rceil$), se obtienen **`61,985` observaciones**, que se dividen de forma estratificada (`SEED = 42`) en:
- **`datos/train.csv` (70%):** `43,389` filas (`2,940` de `Alto_valor`).
- **`datos/validation.csv` (15%):** `9,298` filas (`630` de `Alto_valor`).
- **`datos/test.csv` (15%):** `9,298` filas (`630` de `Alto_valor`).

#### ¿Cómo se interpreta la validación estadística de la muestra (`calculo_muestra.json`)?
- **Prueba $\chi^2$ por clase:** compara la proporción observada en la muestra contra la población limpia de `940,000` filas. Todos los $p$-valores están entre **`0.9966` y `0.9999`** (muy por encima de $\alpha = 0.05$), lo que significa que la diferencia entre las proporciones de la muestra y de la población es prácticamente cero ($< 0.0003\%$).
- **Prueba Kolmogorov–Smirnov (KS) en las 23 variables numéricas:** compara la función de distribución acumulada empírica de cada variable entre la población y la muestra. Las 23 variables tienen **`p > 0.05`** (desde `0.0525` hasta `0.9999`), demostrando que la muestra conserva intacta la forma, media, dispersión y colas de todas las variables originales.

---

### Paso 3: Selección del Número de Pliegues de Validación Cruzada (`crossVal.py`)

#### ¿Cómo se evita la fuga de datos y cómo se elige $k^*$?
`crossVal.py` evalúa $k \in \{3, 5, 10\}$ usando `StratifiedKFold(n_splits=k, shuffle=True, random_state=42)` exclusivamente sobre `train.csv` (`43,389` filas). Dentro de cada pliegue, el pipeline completo (`DomainFeatureTransformer` $\rightarrow$ `ColumnTransformer` con `StandardScaler` y `OneHotEncoder` $\rightarrow$ `MLPClassifier`) se ajusta únicamente con el subconjunto de entrenamiento del fold activo.

La elección de $k^*$ no se deja fija ni arbitraria: se aplica una **regla basada en datos**:
> **Regla:** Elegir el mayor $k \in \{3, 5, 10\}$ tal que cada pliegue de validación contenga al menos **`500` casos de la clase minoritaria (`Alto_valor`)**; en caso de empate, elegir el de menor desviación estándar en `F1-Macro`.

| $k$ | Casos de `Alto_valor` por fold de validación | ¿Cumple $\ge 500$? | `F1-Macro` medio (config. base) | Desv. Est. (`DE`) | Error Est. (`SE`) | Decisión |
|:---:|---:|:---:|---:|---:|---:|---|
| `3` | `980` | Sí | `0.2740` | `0.0054` | `0.0031` | Factible, pero cada modelo entrena solo con el `66.7%` de `train`. |
| **`5`** | **`588`** | **Sí** | **`0.2675`** | **`0.0143`** | **`0.0064`** | **Elegido ($k^* = 5$): garantiza `588 >= 500` casos de `Alto_valor` y entrena con el `80%` de `train`.** |
| `10` | `294` | **No** (`< 500`) | `0.2732` | `0.0098` | `0.0031` | Descartado: deja solo `294` casos de `Alto_valor` por pliegue y duplica el costo de cómputo. |

---

### Paso 4: Ingeniería de Variables e Importancia por Permutación (`src/feature_engineering.py`)

#### Las 8 variables de dominio comercial
Dentro del pipeline se calculan 8 razones numéricas protegidas contra división por cero ($\max(\cdot, 1)$):
1. `ratio_descuento = compras_con_descuento_365d / max(compras_365d, 1)`: captura directamente el umbral histórico del 60% que distingue a `Promocional` de `Frecuente`.
2. `frecuencia_anualizada = compras_365d / max(antiguedad_dias / 365, 0.05)`: mide el ritmo de compra por año de antigüedad.
3. `intensidad_recencia = compras_365d / (dias_ultima_compra + 1)`: combina volumen anual con cercanía de la última compra.
4. `unidades_por_compra = unidades_365d / max(compras_365d, 1)`: tamaño medio de cesta física.
5. `tasa_abandono_carrito = carritos_abandonados_365d / max(sesiones_365d, 1)`: fricción de conversión.
6. `tasa_apertura_emails = emails_abiertos_365d / max(emails_enviados_365d, 1)`: receptividad a campañas de marketing.
7. `tasa_devoluciones = devoluciones_365d / max(compras_365d, 1)`: fricción posventa.
8. `indice_valor_cliente = gasto_total_365d_mxn / (dias_ultima_compra + 1)`: intensidad monetaria reciente.

#### ¿Por qué la importancia por permutación se mide sobre las 29 variables originales?
Si se permutara una columna *después* de `DomainFeatureTransformer`, al desordenar `compras_365d` quedarían intactas `gasto_total_365d_mxn`, `ratio_descuento` y `frecuencia_anualizada`, ocultando la importancia real de `compras_365d`. Al permutar cada una de las **29 variables originales a la entrada del `Pipeline` completo** sobre `validation.csv` (`n_repeats = 10`), el efecto se propaga automáticamente a todas las transformaciones derivadas (`resultados/importancia_permutacion.csv`):

| Posición | Variable original | Caída media en `F1-Macro` | Desv. Est. | Por qué es determinante |
|:---:|---|---:|---:|---|
| **1** | `sesiones_365d` | **`+0.0389`** | `0.0034` | Principal indicador de actividad futura: separa `Inactivo` y `Ocasional` de los perfiles activos. |
| **2** | `descuento_promedio_pct` | **`+0.0345`** | `0.0025` | Variable crítica para identificar el perfil `Promocional` ($\text{desc\_hist} \ge 20\%$). |
| **3** | `compras_365d` | **`+0.0212`** | `0.0030` | Determina tanto el gasto acumulado como la recurrencia anual. |
| **4** | `ticket_promedio_mxn` | **`+0.0044`** | `0.0017` | Separa a los compradores de ticket alto (`Alto_valor`) del resto de activos. |
| **5** | `antiguedad_dias` | **`+0.0033`** | `0.0011` | Modula la frecuencia anualizada de compra. |

---

### Paso 5: Ponderación por Clase ($\gamma^* = 0.50$) y Búsqueda en Rejilla de 472 Combinaciones (`entrenamiento.py`)

#### 5.1 Selección de $\gamma^*$ con la regla de un error estándar (1-SE)
Para compensar el desbalance durante el entrenamiento del MLP, cada cliente de la clase $c$ recibe un peso inversamente proporcional a la frecuencia de su clase amortiguado por el exponente $\gamma$:

$$w_c(\gamma) = \left(\frac{N_{\text{train}}}{K \cdot N_c}\right)^{\gamma}$$

- Si $\gamma = 0.00$, todos los pesos valen $1.0$ (sin balanceo: el modelo favorece `Ocasional` e `Inactivo`).
- Si $\gamma = 1.00$, el balanceo es extremo (`Alto_valor` pesa $5.8\times$ más que `Ocasional`), multiplicando los falsos positivos.

En `sweep_gamma`, se evaluaron $\gamma \in \{0.00, 0.25, 0.50, 0.75\}$ mediante validación cruzada de 5 pliegues sobre `train.csv` (`resultados/barrido_gamma.csv`):

| $\gamma$ | `F1-Macro` CV | Desv. Est. | Error Est. (`SE`) | `Balanced Accuracy` CV | `Accuracy` CV | Decisión por regla 1-SE |
|:---:|---:|---:|---:|---:|---:|---|
| `0.00` | `0.2782` | `0.0099` | `0.0044` | `0.2757` | `0.4227` | Insuficiente en clases minoritarias (`F1-Macro` bajo). |
| `0.25` | `0.3204` | `0.0085` | `0.0038` | `0.3143` | `0.4158` | Debajo del umbral 1-SE (`0.3247`). |
| **`0.50`** | **`0.3249`** | **`0.0038`** | **`0.0017`** | **`0.3372`** | **`0.3914`** | **Elegido ($\gamma^* = 0.50$): supera el umbral `0.3247` y mantiene `+2.76%` más `Accuracy` que $\gamma=0.75$.** |
| `0.75` | `0.3261` | `0.0032` | `0.0014` | `0.3529` | `0.3638` | Máximo nominal (`0.3261 - 0.0014 = 0.3247`), pero sacrifica demasiada exactitud global. |

#### 5.2 Exploración de las 472 combinaciones de la rejilla y modelo ganador
Con $k^* = 5$ y $\gamma^* = 0.50$, se ejecutó `GridSearchCV` sobre **`472` combinaciones de hiperparámetros** (`2,360` ajustes en total, con **`0` combinaciones fallidas**):
- **Bloque Adam (`216` combinaciones):** arquitecturas `(64,)`, `(128, 64)` y `(128, 64, 32)` (1, 2 y 3 capas ocultas); `max_iter` $\in \{150, 300\}$; `learning_rate_init` $\in \{0.0005, 0.001, 0.005\}$; `activation` $\in \{\text{relu}, \text{tanh}\}$; `alpha` $\in \{10^{-4}, 10^{-3}, 10^{-2}\}$; `batch_size` $\in \{256, 512\}$.
- **Bloque SGD con Momentum y Nesterov (`256` combinaciones):** arquitecturas `(64,)` y `(128, 64)`; `max_iter` $\in \{150, 300\}$; `learning_rate_init` $\in \{0.001, 0.005\}$; `learning_rate` $\in \{\text{constant}, \text{adaptive}\}$; `momentum` $\in \{0.90, 0.95\}$; `nesterovs_momentum = True`; `activation` $\in \{\text{relu}, \text{tanh}\}$; `alpha` $\in \{10^{-4}, 10^{-3}\}$; `batch_size` $\in \{256, 512\}$.

**Configuración ganadora (`resultados/mejor_modelo.json`):**
- **Capas y neuronas (`hidden_layer_sizes`):** `(128, 64)` (2 capas ocultas).
- **Activación (`activation`):** `tanh` (al estar acotada en $(-1, 1)$ y centrada en cero, estabiliza los gradientes ponderados por `sample_weight` mejor que `relu`).
- **Optimizador (`solver`):** `sgd` con **`momentum = 0.95`** y **`nesterovs_momentum = True`**.
- **Tasa de aprendizaje:** `constant` con `learning_rate_init = 0.001`.
- **Regularización $L_2$ (`alpha`):** `0.001`.
- **Tamaño de lote (`batch_size`):** `256`.
- **Épocas (`max_iter`):** `150`, deteniéndose por **parada temprana (`early_stopping=True`) en la época `29`**.
- **Puntaje en validación cruzada (5 folds):** **`F1-Macro = 0.3349 ± 0.0029`** (`IC 95%: [0.3313, 0.3385]`).

---

## 3. Justificación Detallada de la Precisión (~39.1% / ~42.3%) y Guía para Interpretar los Resultados

Una pregunta central al revisar las tablas del proyecto es: **¿Por qué la exactitud global (`Accuracy`) del modelo es `39.14%` (con ponderación $\gamma^* = 0.50$) o `42.27%` (sin ponderar / tras calibración isotónica), y por qué ese resultado es técnicamente sólido y esperado?**

A continuación se presentan las **cuatro pruebas empíricas y matemáticas** que lo explican:

### 3.1 La trampa del `Accuracy` en un problema desbalanceado de 5 clases
En este dataset:
- Si el clasificador fuera completamente aleatorio uniforme entre 5 clases, tendría `Accuracy = 20.0%` y `F1-Macro = 0.2000`.
- Si un clasificador trivial (`DummyClassifier(strategy="most_frequent")`) predice el 100% de las veces la clase mayoritaria `Ocasional`, obtiene **`Accuracy = 39.18%`**, pero **su sensibilidad (`Recall`) en `Alto_valor`, `Promocional`, `Frecuente` e `Inactivo` es exactamente `0.00%`**, dando un **`F1-Macro = 0.1126`** y **`Balanced Accuracy = 0.2000`**.
- Por lo tanto, evaluar el modelo únicamente por `Accuracy` premiaría a un modelo ciego a las clases minoritarias.

### 3.2 El compromiso matemático entre `Accuracy` y `F1-Macro` al ponderar clases
Cuando entrenamos el MLP sin ponderar clases ($\gamma = 0.00$) o aplicamos calibración isotónica que devuelve las probabilidades a las frecuencias a priori de la población:
- La **`Accuracy` sube a `42.27%`**, porque el modelo asigna la gran mayoría de las predicciones dudosas a `Ocasional` (`39.2%`) e `Inactivo` (`32.5%`).
- Sin embargo, con $\gamma = 0.00$ el **`F1-Macro` cae a `0.2782`** porque el modelo casi deja de predecir `Frecuente` (`Recall = 6.84%`) y `Promocional` (`Recall = 9.70%`).

Al aplicar la ponderación óptima **$\gamma^* = 0.50$**:
- El modelo acepta reducir su `Accuracy` global en ~3.1 puntos (`42.27%` $\rightarrow$ **`39.14%`**) para abrir espacio de decisión a las tres clases minoritarias de alto interés comercial (`Alto_valor`, `Promocional` y `Frecuente`).
- Gracias a ello, el **`F1-Macro` se triplica respecto al clasificador trivial** (`0.1126` $\rightarrow$ **`0.3383`**, y **`0.3457`** con multiplicadores ajustados en validación) y la **`Balanced Accuracy` sube de `0.2000` a `0.3503`** (`+75%` de ganancia relativa).

### 3.3 Prueba definitiva del Techo de Bayes: tres familias de algoritmos convergen al mismo resultado (`modelos_referencia.py`)
Para descartar que el ~39.1% de `Accuracy` (o `0.3383` de `F1-Macro` y `0.6901` de `ROC-AUC`) se deba a un defecto del MLP, se entrenaron bajo exactamente las mismas particiones y pesos ($\gamma = 0.50$) un modelo lineal regularizado (`LogisticRegression`) y un ensamble de árboles de gradiente de última generación (`HistGradientBoostingClassifier`), además de variar el tamaño de entrenamiento desde `10,000` hasta `300,000` filas (`resultados/modelos_referencia.csv` y `resultados/curva_tamano_muestra.csv`):

| Modelo / Experimento | `Accuracy` en Test | `Balanced Accuracy` | `F1-Macro` en Test | `F1-Weighted` | `ROC-AUC OvR` |
|---|---:|---:|---:|---:|---:|
| `DummyClassifier (most_frequent)` | `0.3918` | `0.2000` | `0.1126` | `0.2206` | `0.5000` |
| `LogisticRegression` ($\gamma = 0.50$) | `0.3902` | `0.3519` | `0.3396` | `0.3887` | `0.6887` |
| **`MLPClassifier` ganador `(128, 64)` ($\gamma = 0.50$)** | **`0.3914`** | **`0.3503`** | **`0.3383`** | **`0.3891`** | **`0.6901`** |
| `HistGradientBoostingClassifier` ($\gamma = 0.50$) | `0.3859` | `0.3490` | `0.3390` | `0.3855` | `0.6960` |
| MLP entrenado con `300,000` filas limpias | `0.3851` | `0.3510` | `0.3388` | `0.3849` | `0.6951` |

**¿Qué demuestra esta tabla?**
1. **No es un problema del modelo ni de falta de datos:** `LogisticRegression`, `HistGradientBoosting` y `MLPClassifier` convergen en una banda estrechísima de **`F1-Macro` entre `0.3383` y `0.3396`** y **`ROC-AUC` entre `0.6887` y `0.6960`**, e incrementar los datos de entrenamiento hasta `300,000` filas da exactamente `0.3388`.
2. **El modelo ya alcanzó el límite de información de las variables históricas:** toda la señal predictiva presente en los 365 días previos está siendo capturada. El error restante proviene exclusivamente de la aleatoriedad intrínseca de las compras futuras $(N, D, G)$ en la ventana de 90 días posteriores al corte.

### 3.4 Cómo interpretar el desempeño por clase (`reporte_clasificacion_test.csv` y `auc_por_clase.csv`)

| Clase de Cliente | Soporte en Test | Precisión | Sensibilidad (`Recall`) | `F1-Score` | `ROC-AUC OvR` | `PR-AUC` | Cómo interpretarlo |
|---|---:|---:|---:|---:|---:|---:|---|
| **`Alto_valor`** | `630` (`6.78%`) | `0.3249` | `0.2460` | `0.2800` | **`0.8021`** | `0.2461` | **Mejor capacidad discriminativa (`AUC = 0.8021`):** aunque representa solo el 6.78% de la base, la precisión del modelo (`32.5%`) es **4.8 veces superior al azar (`6.78%`)**. |
| **`Promocional`** | `851` (`9.15%`) | `0.2528` | **`0.4277`** | `0.3178` | **`0.7412`** | `0.2145` | **Alta sensibilidad (`42.8%`) y buen ranking (`AUC = 0.7412`):** `descuento_promedio_pct` y `ratio_descuento` permiten detectar más de 4 de cada 10 clientes promocionales futuros. |
| **`Frecuente`** | `1,152` (`12.39%`) | `0.2496` | `0.2847` | `0.2660` | `0.6854` | `0.2179` | Duplica la precisión del azar (`25.0%` vs `12.4%`) y cuadruplica el recall respecto al modelo sin pesos (`28.5%` vs `6.8%`). |
| **`Inactivo`** | `3,022` (`32.50%`) | `0.4368` | **`0.5179`** | **`0.4739`** | `0.6717` | `0.4534` | **Mayor `F1-Score` (`0.4739`):** detecta correctamente a más de la mitad de los clientes que no comprarán en el trimestre siguiente. |
| **`Ocasional`** | `3,643` (`39.18%`) | `0.4055` | `0.2750` | `0.3278` | `0.5498` | `0.4239` | **Por qué tiene el menor `ROC-AUC` (`0.5498`):** es la categoría intermedia por defecto; en el horizonte futuro de 90 días se traslapa con clientes que hacen 0 compras (`Inactivo`) o que suben a `Frecuente`/`Promocional`. |

---

## 4. Control de Sobreajuste, Variación Estadística, Calibración e Inferencia Masiva

### 4.1 Demostración de ausencia de sobreajuste (`evaluacion_splits.csv`)
| Partición | Filas | `Accuracy` | `Balanced Accuracy` | `F1-Macro` | `F1-Weighted` | `ROC-AUC OvR` |
|---|---:|---:|---:|---:|---:|---:|
| `train` (70%) | `43,389` | `0.3957` | `0.3532` | `0.3435` | `0.3935` | `0.6946` |
| `validation` (15%) | `9,298` | `0.3885` | `0.3442` | `0.3340` | `0.3865` | `0.6865` |
| **`test` (15%)** | **`9,298`** | **`0.3914`** | **`0.3503`** | **`0.3383`** | **`0.3891`** | **`0.6901`** |
| **Brecha `Train - Test`** | --- | **`+0.0043`** | **`+0.0029`** | **`+0.0052`** | **`+0.0044`** | **`+0.0045`** |

La diferencia entre entrenamiento y prueba es de solo **`0.0052`** (medio punto porcentual), gracias a la regularización $L_2$ (`alpha = 0.001`), la parada temprana en la época `29` y el momento de Nesterov (`0.95`). No existe sobreajuste.

### 4.2 Variación estadística en Validación Cruzada y Bootstrap en Test (`B = 1,000`)
- **En los 5 pliegues de validación cruzada (`variacion_estadistica_mejor_modelo.csv`):**
  - `F1-Macro`: `0.3349 ± 0.0029` (`SE = 0.0013`, `CV = 0.86%`, `IC 95% = [0.3313, 0.3385]`).
  - `Balanced Accuracy`: `0.3391 ± 0.0026` (`SE = 0.0012`, `CV = 0.76%`, `IC 95% = [0.3359, 0.3423]`).
  - `ROC-AUC OvR`: `0.6863 ± 0.0053` (`SE = 0.0024`, `CV = 0.78%`, `IC 95% = [0.6797, 0.6930]`).
- **En `test.csv` mediante `1,000` réplicas Bootstrap con reemplazo (`variacion_estadistica_test_bootstrap.csv`):**
  - `Accuracy`: `0.3914` (`SE = 0.0049`, **`IC 95% = [0.3822, 0.4014]`**).
  - `Balanced Accuracy`: `0.3503` (`SE = 0.0059`, **`IC 95% = [0.3395, 0.3619]`**).
  - `F1-Macro`: `0.3383` (`SE = 0.0055`, **`IC 95% = [0.3279, 0.3494]`**).
  - `ROC-AUC OvR`: `0.6901` (`SE = 0.0036`, **`IC 95% = [0.6830, 0.6971]`**).

### 4.3 Calibración probabilística sin reentrenar (`FrozenEstimator`) y multiplicadores post-hoc
En `src/calibration.py`, el modelo ganador se envuelve con `FrozenEstimator(model)` para ajustar sobre `validation.csv` dos calibradores (`sigmoid` de Platt e `isotonic`) **sin reentrenar los pesos del MLP**, y además se optimizan multiplicadores por clase (`[0.9, 1.3, 1.2, 1.0, 1.2]`) usando solo `validation.csv`:
- **Con multiplicadores sobre el modelo base:** el `F1-Macro` en `test.csv` sube de `0.3383` a **`0.3457`** (`Accuracy = 0.3926`, `Balanced Accuracy = 0.3526`).
- **Con calibración isotónica:** el `Log-Loss` en `test.csv` baja de `1.3251` a **`1.2773`**, el puntaje de `Brier` baja de `0.7279` a **`0.7060`**, el error de calibración (`ECE`) baja de `0.0566` a **`0.0252`**, y la `Accuracy` directa sube a **`0.4227`**.

### 4.4 Evaluación masiva fuera de muestra sobre el archivo completo (`src/evaluation_large.py`)
Para comprobar que el modelo generaliza a toda la base comercial sin fugas ni desbordamiento de RAM, `evaluate_in_chunks` procesa el CSV original en bloques de `50,000` filas, aplica `clean_dataframe` a cada bloque y excluye los `43,389` `id_cliente` pertenecientes a `train.csv`:

$$1,000,000\text{ filas crudas} = 60,000\text{ eliminadas por limpieza} + 43,389\text{ en train} + \mathbf{896,611\text{ evaluadas fuera de entrenamiento}}$$

- **Rendimiento computacional:** `896,611` filas evaluadas en `22.24 segundos` (**`40,321.3 filas/segundo`**).
- **Métricas globales en las `896,611` filas:** `Accuracy = 0.3878`, `Balanced Accuracy = 0.3463`, `F1-Macro = 0.3348`, `F1-Weighted = 0.3858`, `ROC-AUC OvR = 0.6881`. La diferencia contra `test.csv` es menor a `0.0035`, confirmando la representatividad exacta del muestreo de Cochran.

---

## 5. Guía de Lectura de las 11 Gráficas del Proyecto (`graficas/01` a `11`)

| Archivo en `graficas/` | Qué visualiza y cómo interpretarla |
|---|---|
| `01_distribucion_clases_cochran.png` | Compara las barras de proporción de las 5 clases en `population_clean`, `sample_cochran`, `train`, `validation` y `test`. Las 5 barras de cada clase tienen exactamente la misma altura, demostrando visualmente la estratificación perfecta. |
| `02_analisis_k_crossval.png` | Eje izquierdo: `F1-Macro` medio ($\pm 1\text{ DE}$) para $k \in \{3, 5, 10\}$. Eje derecho: casos de `Alto_valor` por pliegue frente a la línea punteada del umbral mínimo (`500` casos). Muestra por qué $k = 5$ (`588` casos) cumple la restricción y $k = 10$ (`294` casos) queda debajo del umbral. |
| `03_evolucion_rejilla_paralela.png` | Muestra las 10 mejores arquitecturas distintas de la rejilla de 472 combinaciones con sus barras de desviación estándar, encabezadas por `sgd tanh (128, 64)` (`F1-Macro CV = 0.3349`). |
| `04_curvas_aprendizaje_convergencia.png` | **Izquierda:** evolución por época de la pérdida (`loss_curve_`) y exactitud interna de validación del MLP ganador hasta la parada temprana en la época 29. **Derecha:** barras de `F1-Macro` y `Balanced Accuracy` en `train`, `validation` y `test` frente a `DummyClassifier`, evidenciando que no hay sobreajuste. |
| `05_impacto_balanceo_frecuente_promocional.png` | Compara los valores de $\gamma \in \{0.0, 0.25, 0.50, 0.75\}$ sobre `F1-Macro`, `Balanced Accuracy` y el `Recall` de `Frecuente` y `Promocional`, mostrando cómo $\gamma = 0.0$ colapsa en las clases minoritarias y $\gamma^* = 0.50$ equilibra las métricas. |
| `06_curvas_calibracion_probabilidades.png` | Compara `Log-Loss`, `Brier Multiclase` y `ECE` en `test.csv` para el modelo sin calibrar vs. `sigmoid` e `isotonic`, mostrando la reducción sistemática del error probabilístico tras calibrar con `FrozenEstimator`. |
| `07_variacion_estadistica_boxplots.png` | **Izquierda:** *boxplots* con los puntos de los 5 pliegues CV en `train`. **Derecha:** estimación puntual e intervalos de confianza al 95% en `test` con `1,000` réplicas *bootstrap*. Demuestra la estrechez de la varianza muestral. |
| `08_matrices_confusion_test_y_archivo.png` | Matrices de confusión normalizadas por fila (la diagonal principal es el `Recall` de cada clase) en `test.csv` (`9,298` filas) y en la evaluación masiva (`896,611` filas). Ambas matrices son prácticamente idénticas, confirmando estabilidad poblacional. |
| `09_curvas_roc_pr_multiclase.png` | **Izquierda:** curvas ROC One-vs-Rest por clase en `test` (destacando `Alto_valor` con `AUC = 0.802` y `Promocional` con `AUC = 0.741`). **Derecha:** curvas Precision–Recall por clase. |
| `10_auc_area_multiclase.png` | Gráfico de barras agrupadas que resume `ROC-AUC` y `PR-AUC` para las cinco clases en `test.csv`. |
| `11_barrido_gamma.png` | Curva de selección de $\gamma^* = 0.50$ en validación cruzada con la banda de error estándar y el umbral horizontal de la regla 1-SE (`0.3247`). |

---

## 6. Comandos de Ejecución Reproducible

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python preprocesamiento.py
python muestreo.py
python crossVal.py --n-jobs -1
python entrenamiento.py --balance weighted --calibrate --n-jobs -1
python modelos_referencia.py --n-jobs -1
python main.py --skip-clean --balance weighted --calibrate --large-validation --n-jobs -1
python -m src.generate_figures
python -m src.generate_extra_figures
python empaquetar_entrega.py
```
