# Clasificación multiclase de clientes con MLP

Proyecto de Minería de Datos — BUAP, Facultad de Ciencias de la Computación.

El sistema clasifica clientes de comercio electrónico en `Inactivo`, `Alto_valor`,
`Promocional`, `Frecuente` y `Ocasional` mediante un pipeline reproducible con
ingeniería de variables, preprocesamiento leakage-safe, validación cruzada,
balanceo, búsqueda de hiperparámetros, calibración y evaluación por lotes.

## Datos

El dataset completo contiene 1,000,000 de registros y 31 columnas. Por su tamaño
no se versiona en GitHub. Colócalo localmente en `datos/` con uno de estos nombres:

- `clientes_comercio_1000000.csv`
- `clientes_comercio_1000000_alumnos (1).csv`
- `clientes_comercio_1000000_alumnos.csv`

El cargador prioriza automáticamente el archivo completo. Consulta
`docs/MANIFIESTO_DATOS.md` para el esquema, reglas de calidad y validaciones.

## Instalación

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Ejecución reproducible

```powershell
python preprocesameinto.py
python muestreo.py
python crossVal.py --n-jobs 2
python entrenamiento.py --balance weighted --calibrate --n-jobs 2
python main.py --skip-clean --balance weighted --calibrate --large-validation
python -m src.generate_v2_figures
python -m src.generate_extra_figures
python empaquetar_entrega.py
```

Para una comprobación rápida:

```powershell
python main.py --fast --balance weighted
```

## Estructura

| Ruta | Contenido |
|---|---|
| `preprocesameinto.py` | Auditoría y limpieza requerida por la rúbrica |
| `muestreo.py` | Cochran y particiones estratificadas |
| `crossVal.py` | Evaluación de `k=3,5,10` con MLP |
| `entrenamiento.py` | Pipeline, rejilla, balanceo y calibración |
| `src/` | Componentes reutilizables |
| `datos/` | Datos locales y particiones no versionadas |
| `resultados/` | Métricas y reportes regenerables |
| `graficas/` | Figuras institucionales |
| `reporte/` | LaTeX y documentación técnica |
| `entrega/` | PDFs y ZIPs finales |

## Calidad y reproducibilidad

- Semilla global: `42`.
- Transformaciones numéricas y categóricas ajustadas dentro del pipeline.
- Balanceo ejecutado exclusivamente en los folds de entrenamiento.
- Métrica de selección: `F1-Macro`.
- Métricas complementarias: balanced accuracy, F1 weighted, precisión/recall,
  ROC-AUC OvR, log-loss, Brier multiclase y ECE.
- Plan de evolución: `docs/PLAN_MEJORA_CONTINUA.md`.

## GitHub

El repositorio público contiene código, documentación, figuras y automatización,
pero no el CSV completo, particiones, modelos serializados ni resultados
temporales. Esto evita superar límites de almacenamiento y mantiene los datos
fuera del historial público.
