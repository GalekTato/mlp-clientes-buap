# Guion para video de presentación — máximo 3 minutos

## Objetivo del video

Explicar de forma clara el problema, el flujo de datos, el modelo, los resultados y las áreas de mejora. La narración debe acompañarse con el PDF, el diagrama del flujo y las gráficas principales.

## Guion sugerido

### 0:00–0:20 — Presentación

> Soy Razo Montañez Gael Askary, de la Facultad de Ciencias de la Computación de la BUAP. En este proyecto desarrollé un sistema de clasificación multiclase de clientes usando una red neuronal MLP para apoyar la segmentación comercial.

Mostrar la portada del reporte con materia, profesor y matrícula.

### 0:20–0:45 — Problema y datos

> El dataset representa clientes de comercio electrónico y contiene variables demográficas, preferencias, actividad de compra, comunicación y satisfacción. La variable objetivo tiene cinco perfiles: Inactivo, Alto_valor, Promocional, Frecuente y Ocasional. El reto principal es el desbalance: las clases mayoritarias pueden ocultar un desempeño deficiente en perfiles minoritarios.

Mostrar la distribución de clases y mencionar que el archivo procesado contiene 1,000,000 de filas, de las cuales 881,034 quedaron disponibles después de la auditoría.

### 0:45–1:10 — Preprocesamiento y muestreo

> Primero audité nulos, dominios, rangos, duplicados y consistencia entre variables. Corregí relaciones deterministas, como el gasto total, y eliminé únicamente registros que no podían repararse de forma defendible. Después calculé la muestra con Cochran y realicé particiones estratificadas de entrenamiento, validación y prueba.

Mostrar el diagrama del flujo y la tabla de auditoría.

### 1:10–1:35 — Ingeniería de variables y modelo

> Para capturar mejor los perfiles Frecuente y Promocional agregué ratio de descuento, frecuencia anualizada, intensidad de recencia, unidades por compra, tasas de abandono, apertura y devoluciones, e índice de valor del cliente. Estas variables se calculan dentro de un pipeline para evitar fuga de información. El modelo combina escalado numérico, codificación one-hot y un MLPClassifier.

Mostrar la tabla de variables derivadas y la curva de pérdida.

### 1:35–2:00 — Balanceo y calibración

> Comparé el esquema base, los pesos de clase y el sobremuestreo dentro de cada fold. El esquema ponderado ofreció el mejor F1 macro medio. En la prueba completa de partición, obtuvo F1 macro 0.3276 y balanced accuracy 0.3574. También comparé calibración sigmoid e isotónica y ajusté multiplicadores usando únicamente validación.

Mostrar la gráfica de impacto del balanceo y la de calibración.

### 2:00–2:30 — Resultados

> El resultado debe interpretarse con varias métricas. El F1 macro ponderado fue 0.3276, la balanced accuracy 0.3574 y el F1 weighted 0.3751. La evaluación por lotes del archivo completo alcanzó F1 macro 0.3219. La matriz de confusión muestra que todavía existe dificultad para capturar Frecuente y Promocional, por lo que no sería correcto afirmar que el problema está completamente resuelto.

Mostrar matriz de confusión, curvas ROC-PR y tabla comparativa.

### 2:30–2:50 — Ventajas y mejoras

> Las ventajas principales son la reproducibilidad, el control de data leakage, la evaluación sensible al desbalance, la calibración de probabilidades y la organización modular. Como mejoras futuras propongo una búsqueda exhaustiva con más recursos, comparación sistemática de SMOTE, ajuste de umbrales por costo comercial y validación con nuevos lotes de datos.

Mostrar la tabla de ventajas y áreas de mejora.

### 2:50–3:00 — Cierre

> En conclusión, el proyecto entrega un pipeline reproducible y técnicamente justificable para clasificación multiclase. La ingeniería de variables y el balanceo mejoran la captura de las clases menos representadas, y el sistema queda preparado para futuras iteraciones con nuevos datos.

Cerrar con la portada o la sección de conclusiones.

## Recomendaciones de grabación

- Usar una resolución mínima de 1080p.
- Mantener visibles las gráficas durante la explicación de cada resultado.
- No leer todas las fórmulas; explicar su propósito.
- Mencionar explícitamente que las métricas corresponden al archivo procesado.
- Mantener el video por debajo de tres minutos.
