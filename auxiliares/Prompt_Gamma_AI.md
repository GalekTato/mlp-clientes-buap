# Prompt para Gamma AI

```text
Actúa como diseñador editorial académico y crea una presentación en español
de 12 diapositivas sobre el proyecto:

“Preprocesamiento, Muestreo de Cochran, Validación Cruzada y Optimización de un
Perceptrón Multicapa (MLP) para Clasificación Multiclase de Clientes”.

Explícalo como un sistema único, riguroso y completo, desde la auditoría del
millón de registros hasta la inferencia masiva fuera de entrenamiento.

## Identificación institucional

- Institución: Benemérita Universidad Autónoma de Puebla (BUAP)
- Unidad académica: Facultad de Ciencias de la Computación (FCC)
- Materia: Minería de Datos (ICCS-257 | NRC: 25467)
- Profesor: Dr. Iván Olmos Pineda
- Alumno: Gael Askary Razo Montañez (Matrícula: 202352659)
- Periodo: Otoño 2026

## Estructura por diapositiva

1. Portada institucional BUAP – FCC.
2. Planteamiento del problema: predicción del perfil a 90 días futuros a partir de 365 días históricos (5 clases: Inactivo, Alto_valor, Promocional, Frecuente, Ocasional) y límite estocástico de Bayes.
3. Preprocesamiento determinista (`preprocesamiento.py`):
   - 1,000,000 de filas de entrada -> 940,000 filas limpias conservadas (94.0%).
   - 60,000 filas rescatadas: 7,562 con formato monetario en ticket, 22,438 con variantes léxicas categóricas y 30,000 con recálculo exacto de gasto (`round(compras * ticket, 2)`).
   - 60,000 filas eliminadas por causa excluyente: 30,000 nulos irrecuperables y 30,000 fuera de rango (0 duplicados, 0 violaciones de dependencias).
4. Muestreo estratificado de Cochran (`muestreo.py`):
   - Población limpia N = 940,000, Z = 2.576, e = 0.005 -> n_FPC = 61,983; muestra estratificada = 61,985.
   - Particiones: train = 43,389 (70%), validation = 9,298 (15%), test = 9,298 (15%).
   - Pruebas ji-cuadrada (p >= 0.9966) y Kolmogorov–Smirnov (p > 0.05 en las 23 variables numéricas).
5. Validación cruzada estratificada (`crossVal.py`):
   - Comparación de k in {3, 5, 10} sobre train.csv.
   - Regla de decisión: mayor k con al menos 500 casos de Alto_valor por pliegue (k=3: 980, k=5: 588, k=10: 294). Selección: k* = 5.
6. Ingeniería de variables e importancia por permutación:
   - 8 variables de dominio dentro del pipeline (`DomainFeatureTransformer`).
   - Top variables originales por permutación en validación: `sesiones_365d` (+0.0389), `descuento_promedio_pct` (+0.0345) y `compras_365d` (+0.0212).
7. Selección del exponente de balanceo gamma* (`entrenamiento.py`):
   - Barrido en 5-fold CV sobre train: gamma in {0.0, 0.25, 0.50, 0.75}.
   - Elección por regla 1-SE: gamma* = 0.50 (F1-Macro CV = 0.3249 ± 0.0017).
8. Búsqueda exhaustiva en rejilla del MLP (472 combinaciones):
   - 216 combinaciones Adam + 256 combinaciones SGD con momentum y Nesterov (2,360 fits, 0 fallos).
   - Configuración ganadora: capas (128, 64), activación tanh, solver sgd, momentum 0.95, Nesterov True, alpha 0.001, lr 0.001, batch 256 (parada temprana en época 29, F1-Macro CV = 0.3349 ± 0.0029).
9. Resultados en Train, Validation y Test y ausencia de sobreajuste:
   - Train: Accuracy 0.3957, Balanced Accuracy 0.3532, F1-Macro 0.3435, ROC-AUC 0.6946.
   - Validation: Accuracy 0.3885, Balanced Accuracy 0.3442, F1-Macro 0.3340, ROC-AUC 0.6865.
   - Test: Accuracy 0.3914, Balanced Accuracy 0.3503, F1-Macro 0.3383 (0.3457 con multiplicadores), ROC-AUC 0.6901.
   - Brecha Train - Test en F1-Macro: 0.0052.
10. Variación estadística (Bootstrap B = 1,000 en Test) y comparación inter-algorítmica:
   - IC 95% en Test: F1-Macro [0.3279, 0.3494], Balanced Accuracy [0.3395, 0.3619], ROC-AUC [0.6830, 0.6971].
   - Convergencia de LogisticRegression (F1 0.3396, AUC 0.6887), HistGradientBoosting (F1 0.3390, AUC 0.6960) y MLP (F1 0.3383, AUC 0.6901).
11. Calibración probabilística (`FrozenEstimator`) y evaluación masiva por lotes:
   - Calibración isotónica en Test: Log-Loss 1.2773, Brier 0.7060, ECE 0.0252.
   - Evaluación por lotes: 896,611 filas limpias fuera de train (+ 60,000 eliminadas + 43,389 train = 1,000,000) a 40,321.3 filas/s, F1-Macro = 0.3348, ROC-AUC = 0.6881.
12. Conclusiones, fortalezas arquitectónicas y líneas de mejora futura.

## Gráficas del proyecto a integrar

- `graficas/01_distribucion_clases_cochran.png`
- `graficas/02_analisis_k_crossval.png`
- `graficas/11_barrido_gamma.png`
- `graficas/03_evolucion_rejilla_paralela.png`
- `graficas/04_curvas_aprendizaje_convergencia.png`
- `graficas/05_impacto_balanceo_frecuente_promocional.png`
- `graficas/06_curvas_calibracion_probabilidades.png`
- `graficas/07_variacion_estadistica_boxplots.png`
- `graficas/08_matrices_confusion_test_y_archivo.png`
- `graficas/09_curvas_roc_pr_multiclase.png`
- `graficas/10_auc_area_multiclase.png`

## Paleta institucional BUAP – FCC

- Azul marino BUAP: `#003B5C`
- Azul FCC: `#0090D0`
- Naranja FCC: `#F08A00`
- Verde azulado: `#16A085`
- Fondo claro: `#F2F6FA`
```
