# Prompt para Gamma AI

```text
Actúa como diseñador editorial académico y crea una presentación en español
de 10 a 12 diapositivas sobre el proyecto:

“Sistema reproducible de clasificación multiclase de clientes mediante una
Red Neuronal Artificial MLP”.

No presentes el trabajo como versiones, iteraciones o mejoras separadas.
Explícalo como un único sistema completo, desde los datos hasta la evaluación.

## Identificación

- Materia: Minería de Datos
- NRC: 25467
- Clave: ICCS-257
- Profesor: Dr. Iván Olmos Pineda
- Alumno: Razo Montañez Gael Askary
- Matrícula: 202352659
- Periodo: Otoño 2026

## Datos y objetivo

El dataset contiene 1,000,000 de registros y 31 columnas. Después de la
auditoría se conservaron 881,034 registros. El objetivo es clasificar cinco
perfiles: Inactivo, Alto_valor, Promocional, Frecuente y Ocasional.

## Estructura obligatoria

1. Portada institucional.
2. Problema comercial y objetivo de clasificación.
3. Comprensión del dataset y distribución desbalanceada.
4. Auditoría: nulos, dominios, rangos, duplicados y reglas relacionales.
5. Muestreo de Cochran y particiones:
   - muestra: 61,712;
   - train: 43,198;
   - validation: 9,257;
   - test: 9,257.
6. Ingeniería de variables:
   ratio_descuento, frecuencia_anualizada, intensidad_recencia,
   unidades_por_compra, tasa_abandono_carrito, tasa_apertura_emails,
   tasa_devoluciones e indice_valor_cliente.
7. Pipeline:
   DomainFeatureTransformer, ColumnTransformer, StandardScaler,
   OneHotEncoder y MLPClassifier.
8. Validación cruzada estratificada:
   explicar la selección de k=5 y la representación de Alto_valor.
9. Balanceo dentro de folds:
   - distribución original: F1 macro 0.2769;
   - ponderación: F1 macro 0.3122;
   - sobremuestreo: F1 macro 0.2614.
10. Resultados del modelo:
   - Train: F1 macro 0.3429, balanced accuracy 0.3760;
   - Validation: F1 macro 0.3185, balanced accuracy 0.3499;
   - Test: F1 macro 0.3276, balanced accuracy 0.3574;
   - Test ROC-AUC OvR: 0.6895.
11. Evaluación del archivo completo:
   - 970,000 filas válidas;
   - 30,000 omitidas por campos no convertibles;
   - throughput: 19,309 registros/s;
   - F1 macro: 0.3219;
   - balanced accuracy: 0.3523;
   - log-loss: 1.4139.
12. Calibración, conclusiones, ventajas y áreas de trabajo futuro.

## Uso obligatorio de gráficas

Incluye y referencia visualmente estas imágenes del proyecto. No inventes
gráficas nuevas si la imagen indicada ya contiene la evidencia solicitada:

- `graficas/01_distribucion_clases_cochran.png`: distribución de clases.
- `graficas/02_analisis_k_crossval.png`: selección de k.
- `graficas/03_evolucion_rejilla_paralela.png`: configuraciones.
- `graficas/04_curvas_aprendizaje_convergencia.png`: convergencia.
- `graficas/05_impacto_balanceo_frecuente_promocional.png`: balanceo.
- `graficas/06_curvas_calibracion_probabilidades.png`: calibración.
- `graficas/07_variacion_estadistica_boxplots.png`: variación.
- `graficas/08_matrices_confusion_test_y_archivo.png`: matrices de confusión.
- `graficas/09_curvas_roc_pr_multiclase.png`: ROC y Precision-Recall.
- `graficas/10_auc_area_multiclase.png`: área bajo la curva ROC-AUC y PR-AUC
  por clase. Destaca esta como la gráfica de AUC/área bajo la curva.

Para cada gráfica incluye una frase breve que explique qué significa, no solo
su título. Usa las gráficas como evidencia y evita llenar las diapositivas con
tablas extensas.

## Estilo visual

Usa una estética institucional, académica, clara y moderna:

- `seupurple`: #8E44AD
- `lachaiseblue`: #2980B9
- `seublue`: #3498DB
- `seuteal`: #16A085
- `seuamber`: #D35400
- `seugray`: #7F8C8D
- `seubg`: #F0F8FF

Colores de clase:

- Alto_valor: seupurple
- Frecuente: lachaiseblue
- Ocasional: seublue
- Promocional: seuteal
- Inactivo: seuamber

Usa fondo claro, títulos morados, subtítulos azules, tarjetas con bordes
redondeados, etiquetas legibles, gráficos grandes y alto contraste. Incluye
notas del presentador de 15 a 25 segundos por diapositiva. La presentación
debe poder exponerse en aproximadamente tres minutos.

No inventes métricas, nombres, enlaces ni resultados. No uses expresiones como
“versión 1”, “versión 2” o “mejora sobre una versión anterior”. Presenta todo
como una implementación única y completa.
```
