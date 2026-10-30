# Guion para Video de Presentación — Máximo 3 Minutos

## Objetivo del video

Exponer de forma clara y estructurada el problema de clasificación multiclase, el preprocesamiento determinista, el muestreo de Cochran, la validación cruzada, la optimización del Perceptrón Multicapa y los resultados estadísticos obtenidos.

## Guion sugerido

### 0:00–0:20 — Presentación e introducción

> Hola, soy Gael Askary Razo Montañez, estudiante de la Facultad de Ciencias de la Computación de la BUAP. En este proyecto desarrollé un sistema de clasificación multiclase con un Perceptrón Multicapa para predecir el perfil de comportamiento de clientes de comercio electrónico en los 90 días posteriores al corte, a partir de su historial de 365 días.

Mostrar la portada institucional del reporte con los logotipos de la BUAP y la FCC.

### 0:20–0:50 — Comprensión de datos y preprocesamiento (`preprocesamiento.py`)

> La base original contiene un millón de registros y 31 columnas con cinco clases desbalanceadas: Ocasional, Inactivo, Frecuente, Promocional y Alto_valor. En el preprocesamiento separé los errores de formato reparables de los datos irrecuperables: rescaté 7,562 importes con formato monetario en ticket promedio, normalicé 22,438 etiquetas categóricas con variantes de escritura y recalculé de forma exacta 30,000 registros de gasto anual mediante el producto de compras por ticket promedio. Únicamente eliminé 30,000 filas con valores nulos reales y 30,000 fuera de rango, conservando 940,000 registros limpios.

Mostrar la Tabla 1 de contabilidad del preprocesamiento y el diagrama de flujo.

### 0:50–1:20 — Muestreo de Cochran y validación cruzada (`muestreo.py` y `crossVal.py`)

> Con una población limpia de 940,000 clientes, apliqué la fórmula de Cochran con corrección por población finita al 99% de confianza y 0.5% de error, obteniendo una muestra estratificada de 61,985 registros dividida en 43,389 para entrenamiento, 9,298 para validación y 9,298 para prueba, verificada con pruebas ji-cuadrada y Kolmogorov–Smirnov. En validación cruzada comparé k igual a 3, 5 y 10, seleccionando k igual a 5 porque garantiza 588 casos de la clase minoritaria Alto_valor en cada pliegue de validación.

Mostrar las Figuras 01 (distribución de clases) y 02 (selección de $k = 5$).

### 1:20–1:55 — Balanceo $\gamma^*$ y búsqueda en rejilla del MLP (`entrenamiento.py`)

> Para mitigar el desbalance sin perder precisión global, evalué el exponente de ponderación gamma en validación cruzada y seleccioné gamma igual a 0.50 con la regla de un error estándar. Después ejecuté una búsqueda en rejilla de 472 combinaciones explorando arquitecturas de una a tres capas ocultas, activaciones ReLU y tanh, regularización L2 y optimizadores Adam y SGD con momento de Nesterov. El mejor modelo fue una red de dos capas ocultas de 128 y 64 neuronas con activación tanh y SGD con momento de 0.95 y Nesterov, que convergió en la época 29 alcanzando un F1-Macro en validación cruzada de 0.3349.

Mostrar las Figuras 11 (barrido de $\gamma$), 03 (rejilla de 472 combinaciones) y 04 (curva de convergencia).

### 1:55–2:35 — Resultados en prueba, techo de predictibilidad y evaluación masiva

> En el conjunto de prueba independiente, el MLP obtuvo F1-Macro de 0.3383 —que sube a 0.3457 con multiplicadores ajustados en validación—, Balanced Accuracy de 0.3503, exactitud de 0.3914 y ROC-AUC de 0.6901, destacando un ROC-AUC de 0.8021 para Alto_valor y 0.7412 para Promocional. La brecha entre entrenamiento y prueba fue de apenas 0.0052, demostrando ausencia de sobreajuste. Además, al comparar contra Regresión Logística y Gradient Boosting, los tres modelos convergen alrededor del mismo techo porque las compras futuras de los siguientes 90 días tienen variabilidad estocástica no observada en el corte histórico. Finalmente, en la evaluación por lotes sobre las 896,611 filas limpias fuera de entrenamiento, el modelo mantuvo un F1-Macro de 0.3348 procesando más de 40,000 registros por segundo.

Mostrar las Figuras 07 (intervalos bootstrap), 08 (matrices de confusión) y 09 (curvas ROC y PR).

### 2:35–3:00 — Ventajas, mejoras futuras y conclusión

> Entre las principales fortalezas destacan el rescate determinista de 60,000 registros, el aislamiento total contra fuga de datos, la exploración completa de 472 configuraciones y la calibración con FrozenEstimator. Como mejoras para producción propongo incorporar variables de tendencia mensual previa al corte y una matriz de costos comerciales asimétricos. Muchas gracias.

Cerrar con la Tabla 10 de ventajas y áreas de mejora.
