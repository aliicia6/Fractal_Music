# Concepto Cerrado De Primera Version

Caso principal:

1. Definir una recurrencia compleja `z[n + 1] = z[n]^2 + c`.
2. Calcular una cantidad configurable de terminos.
3. Detectar si aparece un punto fijo, ciclo, divergencia o comportamiento no resuelto.
4. Proyectar los valores complejos a coordenadas polares.
5. Discretizar el dominio acotado `r` y `theta` mediante subdivisiones uniformes.
6. Visualizar la trayectoria, los atractores y la tabla de transformacion discreta.
7. Guardar varios casos y compararlos bajo el mismo sistema de coordenadas.

La configuracion reproducible del caso principal vive en:

```python
fractal_sequences.default_concept_config()
```

El pipeline completo se ejecuta con:

```python
fractal_sequences.run_exploration(config)
```

## Explorador visual

La aplicacion de escritorio se inicia con:

```powershell
python .\explorador_visual.py
```

En el diagrama de bifurcacion y el conjunto de Mandelbrot se selecciona con
clic izquierdo. La rueda del raton, situada sobre el mapa, amplia o reduce la
vista; la rueda situada sobre la tabla o la lista de casos las desplaza.

Un punto logistico `(r, x0)` se vincula automaticamente con la recurrencia
compleja equivalente:

```text
c = r(2-r)/4
z0 = r(1/2-x0)
```

Por tanto, al cambiar a `z^2+c` se muestra y calcula el parametro real de
Mandelbrot correspondiente. La conversion inversa solo existe para puntos del
eje real del plano de parametros; se emplea la rama principal de `r`.

`Guardar caso` conserva el ejemplo actual. Los casos guardados se dibujan a la
vez, junto con sus atractores, en cualquiera de los sistemas de coordenadas.
Los botones de exportacion generan una sesion JSON completa - configuracion,
casos, terminos, atractores y discretizacion - o una tabla CSV de resultados.
`Importar JSON` recupera los casos y recalcula sus resultados con la
configuracion almacenada.

## Reinicio y musica

`Limpiar todo` elimina los casos guardados, sucesiones, atractores, tabla,
valores numericos introducidos y el zoom de los mapas. La sesion queda vacia,
pero el diagrama de bifurcacion sigue disponible para seleccionar un nuevo
punto con el raton.

El menu `Strudel` transforma los atractores discretizados en un comando para
pegar en Strudel. Los criterios disponibles son `melodia`, `armonia`, `ritmo`
y `textura`; el mapa musical se puede basar en:

- `franjas`: indice de la banda espacial mas expresiva (`y`, radio o `x`).
- `posiciones`: identificador lineal de la celda discreta.
- `sectores`: indice angular o fase compleja del punto.

El selector de escala usa mayor, menor, pentatonica o cromatica desde `C4`.
Junto al comando se muestran las celdas y notas de los primeros atractores; el
comando contiene todos los atractores detectados. `Copiar` lo coloca en el
portapapeles. La configuracion musical tambien se conserva al exportar e
importar una sesion JSON.
