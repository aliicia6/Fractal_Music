# Concepto Cerrado De Primera Version

Caso principal:

1. Definir una recurrencia compleja `z[n + 1] = z[n]^2 + c`.
2. Calcular una cantidad configurable de terminos.
3. Detectar si aparece un punto fijo, ciclo, divergencia o comportamiento no resuelto.
4. Proyectar los valores complejos a coordenadas polares.
5. Discretizar el dominio acotado `r` y `theta` mediante subdivisiones uniformes.
6. Visualizar la trayectoria, los atractores y la tabla de transformacion discreta.

La configuracion reproducible del caso principal vive en:

```python
fractal_sequences.default_concept_config()
```

El pipeline completo se ejecuta con:

```python
fractal_sequences.run_exploration(config)
```

No se cierran todavia las estadisticas discretas avanzadas ni la exportacion.
