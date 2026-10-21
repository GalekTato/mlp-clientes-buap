# Prompt para Gamma AI

```text
Actúa como diseñador editorial académico y crea una presentación profesional en español, de 10 a 12 diapositivas, sobre el proyecto “Clasificación Multiclase de Clientes mediante Redes Neuronales Artificiales”.

Audiencia: profesor universitario de Minería de Datos de la Facultad de Ciencias de la Computación de la BUAP.

Datos de identificación:
- Materia: Minería de Datos
- NRC: 25467
- Clave: ICCS-257
- Profesor: Dr. Iván Olmos Pineda
- Alumno: Razo Montañez Gael Askary
- Matrícula: 202352659
- Periodo: Otoño 2026

Objetivo:
Explicar el desarrollo de un sistema reproducible de clasificación multiclase para cinco perfiles de clientes: Inactivo, Alto_valor, Promocional, Frecuente y Ocasional.

Contenido obligatorio:
1. Portada institucional con BUAP, FCC, materia, alumno y título.
2. Problema comercial y comprensión del dataset.
3. Distribución desbalanceada de las clases y consecuencias de usar solo accuracy.
4. Auditoría y preprocesamiento: nulos, dominios, rangos, relaciones deterministas, correcciones y eliminación trazable.
5. Muestreo de Cochran y particiones estratificadas train/validation/test.
6. Ingeniería de variables: ratio_descuento, frecuencia_anualizada, intensidad_recencia, unidades_por_compra, tasas de abandono/apertura/devolución e índice de valor.
7. Arquitectura del pipeline: transformer de dominio, ColumnTransformer, StandardScaler, OneHotEncoder y MLPClassifier.
8. Validación cruzada estratificada y selección de k=5.
9. Balanceo: comparación del esquema base frente a pesos de clase y sobremuestreo.
10. Calibración probabilística y ajuste de multiplicadores de clase.
11. Resultados cuantitativos:
   - Baseline: F1 macro 0.2715, balanced accuracy 0.2703, F1 weighted 0.3809.
   - Ponderado: F1 macro 0.3240, balanced accuracy 0.3593, F1 weighted 0.3620.
   - Calibración isotónica: F1 macro ajustado en validación 0.3311.
   - Importancia por permutación: descuento_promedio_pct 0.0347, sesiones_365d 0.0205, frecuencia_anualizada 0.0092.
12. Conclusiones, ventajas, limitaciones y mejoras futuras.

Estilo visual:
- Usa una estética institucional académica, limpia y moderna.
- Paleta exacta:
  - seupurple #8E44AD
  - lachaiseblue #2980B9
  - seublue #3498DB
  - seuteal #16A085
  - seuamber #D35400
  - seugray #7F8C8D
  - seubg #F0F8FF
- Asigna los colores de clase así:
  - Alto_valor: seupurple
  - Frecuente: lachaiseblue
  - Ocasional: seublue
  - Promocional: seuteal
  - Inactivo: seuamber
- Usa fondo claro, títulos morados, subtítulos azules, tarjetas con bordes redondeados, tablas legibles y gráficos con etiquetas directas.
- Evita saturar las diapositivas con texto. Usa máximo cinco ideas principales por diapositiva.
- Incluye notas del presentador con una explicación oral de 15 a 25 segundos por diapositiva.
- No inventes métricas, nombres de archivos, enlaces ni resultados que no estén incluidos en este prompt.
- Redacta todo en español académico, con lenguaje claro y adecuado para una exposición de máximo tres minutos.
```
