# Plan de Mejora Continua y Despliegue

## Ciclo operativo del pipeline

1. **Calidad de datos (`preprocesamiento.py`):** aplicar rescate determinista de formatos monetarios, categorías y gasto derivado antes de filtrar nulos reales o valores fuera de rango.
2. **Muestreo y particionado (`muestreo.py`):** conservar afijación proporcional estratificada de Cochran y validar representatividad con pruebas $\chi^2$ y Kolmogorov–Smirnov.
3. **Validación cruzada (`crossVal.py`) y ponderación (`entrenamiento.py`):** garantizar soporte mínimo de `500` casos de la clase minoritaria por pliegue (`k = 5`) y seleccionar el exponente $\gamma$ mediante la regla de un error estándar (`γ* = 0.50`).
4. **Optimización y calibración (`entrenamiento.py`, `src/calibration.py`):** seleccionar hiperparámetros por `F1-Macro`, calibrar probabilidades con `FrozenEstimator` sobre `validation.csv` y cuantificar incertidumbre con *bootstrap* (`B = 1,000`) en `test.csv`.
5. **Inferencia escalable (`src/evaluation_large.py`):** procesar lotes en memoria acotada excluyendo los identificadores de entrenamiento.

## Líneas de evolución futura

- **Variables temporales multi-horizonte:** incorporar ventanas móviles de 30, 60 y 90 días previos al corte para capturar cambios de tendencia intra-anual.
- **Matriz de costos comerciales asimétricos:** ajustar los multiplicadores de decisión en función del margen neto esperado por segmento y del costo de campañas de retención.
- **Optimización bayesiana con poda temprana:** complementar la rejilla exhaustiva con búsqueda adaptativa (TPE / Hyperband) al incorporar nuevas fuentes de datos.
- **Monitoreo de deriva poblacional (PSI / KS):** automatizar alertas sobre cambios en la distribución de entradas o en la calibración de probabilidades durante temporadas promocionales.
