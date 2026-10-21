# Plan de mejora continua

## Ciclo de trabajo

1. **Datos:** validar esquema, calidad, distribución y deriva antes de cada
   entrenamiento.
2. **Modelo:** comparar baseline, pesos y sobremuestreo con particiones
   estratificadas y semillas registradas.
3. **Evaluación:** conservar métricas macro, por clase y probabilísticas; nunca
   seleccionar el modelo solo por accuracy.
4. **Reproducibilidad:** registrar parámetros, versiones, tiempos, rutas,
   artefactos y estado de ejecución.
5. **Documentación:** actualizar el reporte únicamente con resultados generados
   por la ejecución vigente.

## Próximas mejoras técnicas

- Ejecutar una rejilla completa con presupuesto de cómputo documentado.
- Comparar sistemáticamente `RandomOverSampler` y `SMOTE` con calibración.
- Añadir validación temporal si se incorporan fechas de operación.
- Incorporar costos comerciales para ajustar umbrales por clase.
- Monitorizar deriva de clases, variables y calibración.
- Añadir pruebas automatizadas para reglas de dominio y contratos de columnas.
- Registrar experimentos con MLflow o una alternativa local reproducible.

## Criterios de promoción

Un modelo se promueve solo si:

- supera el umbral acordado de F1-Macro y balanced accuracy;
- no aumenta de forma material la brecha train-test;
- conserva recall mínimo para `Frecuente`, `Promocional` y `Alto_valor`;
- mantiene calibración aceptable;
- puede reproducirse desde los comandos documentados.
