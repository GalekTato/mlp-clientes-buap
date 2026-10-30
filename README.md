# Clasificación Multiclase de Clientes con Perceptrón Multicapa (MLP)

## Datos académicos

- **Institución:** Benemérita Universidad Autónoma de Puebla (BUAP)
- **Unidad académica:** Facultad de Ciencias de la Computación (FCC)
- **Materia:** Minería de Datos
- **Clave:** ICCS-257
- **NRC:** 25467
- **Profesor:** Dr. Iván Olmos Pineda
- **Alumno:** Gael Askary Razo Montañez
- **Matrícula:** 202352659
- **Periodo:** Otoño 2026
- **Evaluación:** 1er Examen Parcial — Clasificación Multiclase con RNA (MLP)
- **Fecha de entrega:** 3 de noviembre de 2026

Este repositorio contiene la implementación completa y reproducible para clasificar clientes de comercio electrónico en cinco perfiles prospectivos a 90 días (`Inactivo`, `Alto_valor`, `Promocional`, `Frecuente` y `Ocasional`) a partir de su actividad histórica de 365 días (`1,000,000` de registros y `31` columnas).

El flujo integra auditoría y limpieza determinista (`940,000` registros limpios conservados), muestreo estratificado de Cochran con corrección por población finita (`61,985` observaciones divididas 70/15/15 en `train`, `validation` y `test`), validación cruzada estratificada (`k = 5`), selección de ponderación por clase con regla 1-SE (`γ = 0.50`), búsqueda paralela en rejilla de `472` combinaciones de hiperparámetros para `MLPClassifier`, calibración probabilística sin reentrenamiento (`FrozenEstimator`), estimación de variación estadística con `1,000` réplicas *bootstrap* y evaluación masiva por lotes sobre las `896,611` filas limpias fuera de entrenamiento.

## Datos de entrada

El dataset crudo contiene `1,000,000` de registros y `31` columnas. Por su tamaño no se versiona en GitHub. Colócalo localmente en `datos/` con cualquiera de los siguientes nombres reconocidos automáticamente:

- `clientes_comercio_1000000.csv`
- `clientes_comercio_1000000_alumnos (1).csv`
- `clientes_comercio_1000000_alumnos.csv`

Consulta `auxiliares/MANIFIESTO_DATOS.md` para el detalle del esquema, dominios válidos y reglas de integridad.

## Instalación

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Ejecución reproducible

```powershell
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

## Estructura del proyecto

| Ruta | Contenido |
|---|---|
| `preprocesamiento.py` | Auditoría y limpieza determinista del dataset (`1,000,000` → `940,000` filas limpias) |
| `muestreo.py` | Muestreo estratificado de Cochran (`n = 61,985`) y particiones `train` (`43,389`), `validation` (`9,298`) y `test` (`9,298`) |
| `crossVal.py` | Evaluación estratificada de `k in {3, 5, 10}` y selección basada en soporte mínimo de la clase minoritaria (`k = 5`) |
| `entrenamiento.py` | Barrido de `γ`, rejilla de `472` combinaciones para `MLPClassifier`, balanceo, calibración, *bootstrap* e importancia por permutación |
| `modelos_referencia.py` | Comparación contra `DummyClassifier`, `LogisticRegression` y `HistGradientBoostingClassifier`, y curva de tamaño de muestra |
| `main.py` | Orquestador del flujo completo e inferencia masiva por lotes |
| `proyecto_utils.py` | Utilidades compartidas de lectura tipada, normalización, preprocesamiento y métricas |
| `empaquetar_entrega.py` | Compilación del reporte en PDF y generación del archivo ZIP de entrega |
| `src/` | Módulos internos de ingeniería de variables, calibración, evaluación por lotes y visualización |
| `resultados/` | Archivos CSV, JSON y Markdown con las tablas y métricas verificadas |
| `graficas/` | Catálogo de 11 figuras diagnósticas en formato PNG y PDF |
| `reporte/` | Código fuente LaTeX, logotipos institucionales BUAP/FCC y PDF compilado del reporte técnico |
| `auxiliares/` | Documentación técnica detallada, guion del video de 3 minutos, prompt de presentación y manifiesto de datos |

## Resumen de resultados verificados

- **Población limpia (`N`):** `940,000` registros (`60,000` rescatados determinísticamente y `60,000` eliminados: `30,000` nulos y `30,000` fuera de rango).
- **Muestra de Cochran (`Z = 2.576`, `e = 0.005`):** `61,983` (`61,985` por afijación proporcional estratificada: `43,389` en `train`, `9,298` en `validation`, `9,298` en `test`).
- **Validación cruzada:** `k = 5` (`588` casos de `Alto_valor` por pliegue de validación).
- **Mejor modelo MLP (`472` combinaciones evaluadas):** `hidden_layer_sizes = (128, 64)`, `solver = 'sgd'`, `momentum = 0.95`, `nesterovs_momentum = True`, `activation = 'tanh'`, `alpha = 0.001`, `learning_rate = 'constant'` (`0.001`), `batch_size = 256`, `γ = 0.50` (parada temprana en la época `29`).
- **Métricas en `test` (`9,298` filas, IC 95% Bootstrap `B = 1,000`):**
  - `Accuracy`: `0.3914` (`[0.3822, 0.4014]`)
  - `Balanced Accuracy`: `0.3503` (`[0.3395, 0.3619]`)
  - `F1-Macro`: `0.3383` (`[0.3279, 0.3494]`, y `0.3457` con multiplicadores ajustados en validación)
  - `F1-Weighted`: `0.3891`
  - `ROC-AUC OvR`: `0.6901` (`[0.6830, 0.6971]`)
  - Brecha de generalización `Train - Test` en `F1-Macro`: `0.0052`.
- **Evaluación masiva por lotes (`896,611` filas limpias fuera de `train`):** `Accuracy = 0.3878`, `Balanced Accuracy = 0.3463`, `F1-Macro = 0.3348`, `ROC-AUC OvR = 0.6881` (`40,321.3` filas/s).
