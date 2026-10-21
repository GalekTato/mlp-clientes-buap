# Clasificación multiclase de clientes con MLP

## Datos académicos

- **Institución:** Benemérita Universidad Autónoma de Puebla
- **Unidad académica:** Facultad de Ciencias de la Computación
- **Materia:** Minería de Datos
- **Clave:** ICCS-257
- **NRC:** 25467
- **Profesor:** Dr. Iván Olmos Pineda
- **Alumno:** Razo Montañez Gael Askary
- **Matrícula:** 202352659
- **Periodo:** Otoño 2026
- **Evaluación:** 1er Examen Parcial — Clasificación Multiclase con RNA (MLP)
- **Fecha de entrega:** 3 de noviembre de 2026

Este repositorio contiene la implementación reproducible del sistema solicitado
para el primer examen parcial.

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
`auxiliares/MANIFIESTO_DATOS.md` para el esquema, reglas de calidad y validaciones.

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
python -m src.generate_figures
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
| `preprocesamiento.py` | Alias legible hacia `preprocesameinto.py` |
| `muestreo.py` | Muestreo de Cochran y particiones estratificadas (`train`, `validation`, `test`) |
| `crossVal.py` | Evaluación estratificada de `k=3,5,10` con MLP |
| `entrenamiento.py` | Pipeline, búsqueda en rejilla, balanceo y calibración |
| `main.py` | Orquestador completo del flujo reproducible |
| `proyecto_utils.py` | Utilidades compartidas de carga y rutas |
| `empaquetar_entrega.py` | Compilación del reporte PDF y empaquetado del ZIP de entrega |
| `src/` | Módulos internos reutilizables |
| `datos/` | Dataset local y particiones estratificadas |
| `modelos/` | Modelo MLP entrenado y serializado (`.joblib`) |
| `resultados/` | Métricas, tablas y reportes de auditoría/experimentos |
| `graficas/` | Figuras institucionales generadas |
| `auxiliares/` | Explicación técnica detallada, prompt de Gamma, guion del video y documentación complementaria |
| `reporte/` | Código fuente LaTeX del reporte técnico |
| `entrega/` | PDF y ZIP finales solicitados |

## Calidad y reproducibilidad

- Semilla global: `42`.
- Transformaciones numéricas y categóricas ajustadas dentro del pipeline.
- Balanceo ejecutado exclusivamente en los folds de entrenamiento.
- Métrica de selección: `F1-Macro`.
- Métricas complementarias: balanced accuracy, F1 weighted, precisión/recall,
  ROC-AUC OvR, log-loss, Brier multiclase y ECE.
- Explicación técnica completa a detalle: `auxiliares/Documentacion_Tecnica_Proyecto.md`.
- Guion del video: `auxiliares/Guion_Video_3_Minutos.md`.
- Prompt de presentación Gamma: `auxiliares/Prompt_Gamma_AI.md`.
- Plan de evolución: `auxiliares/PLAN_MEJORA_CONTINUA.md`.

## GitHub

El repositorio público contiene código, documentación, figuras y automatización,
pero no el CSV completo, particiones, modelos serializados ni resultados
temporales. Esto evita superar límites de almacenamiento y mantiene los datos
fuera del historial público.
