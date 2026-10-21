# Manifiesto del dataset

## Archivo esperado

El proyecto trabaja con el CSV completo de clientes de comercio electrónico:

- 1,000,000 de registros de clientes.
- 31 columnas.
- 1 identificador (`id_cliente`), 29 variables predictivas y 1 objetivo
  (`clase_cliente`).
- Codificación UTF-8, separador coma y encabezado.

## Clases objetivo

`Inactivo`, `Alto_valor`, `Promocional`, `Frecuente` y `Ocasional`.

## Reglas verificadas

- Rangos de edad, antigüedad, recencia, actividad digital, satisfacción y
  variables monetarias.
- Dominios categóricos cerrados.
- Identificadores únicos.
- Gasto total igual a `round(compras_365d * ticket_promedio_mxn, 2)`.
- Consistencia de ceros cuando no existen compras.
- Límites de devoluciones, descuentos, categorías, sesiones, carritos y correos.

## Política de calidad

Se corrigen únicamente valores derivados cuya reconstrucción es inequívoca.
Los registros con nulos críticos, categorías desconocidas, rangos imposibles o
violaciones relacionales no reparables se excluyen y se contabilizan en
`resultados/auditoria_preprocesamiento.json`.

## Estado de la auditoría actual

La última ejecución completa auditó 1,000,000 filas, conservó 881,034 registros
limpios y produjo la distribución de particiones en
`resultados/distribucion_particiones.csv`.
