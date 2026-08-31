# Comparativa de cadenas de percepción (apartado 6.1)

| Backend | Exactitud | F1 macro | F1 dolor | F1 miedo | Latencia mediana (ms) | Latencia p95 (ms) | No detectados (%) |
|---|---|---|---|---|---|---|---|
| mediapipe (mediapipe) | 0.545 | 0.480 | 0.304 | 0.285 | 16.8 | 20.9 | 10.7 |
| deepface (deepface) | 0.578 | 0.546 | 0.497 | 0.397 | 24.5 | 30.8 | 28.2 |

## Subconjunto común · imágenes detectadas por todos los backends

4921 de 7178 imágenes (68.6 %). Elimina el sesgo de comparar exactitudes calculadas sobre conjuntos distintos.

| Backend | Exactitud | F1 macro | F1 dolor | F1 miedo |
|---|---|---|---|---|
| mediapipe | 0.560 | 0.491 | 0.359 | 0.298 |
| deepface | 0.585 | 0.550 | 0.504 | 0.397 |

---

### Advertencia sobre la clase «dolor»

FER-2013 no contiene una categoría de dolor. Las imágenes etiquetadas aquí como `pain` son la clase **disgust** (asco) del conjunto original, renombrada. La correspondencia se apoya en la proximidad morfológica de ambas configuraciones faciales —arrugamiento nasal y elevación del labio superior— pero **no es una equivalencia**.

En consecuencia, las columnas de dolor de este informe NO acreditan que el sistema detecte dolor. Acreditan que discrimina una expresión empleada como aproximación. Cualquier afirmación clínica exigiría una validación con un conjunto de dolor real (por ejemplo UNBC-McMaster) y con población pediátrica. Véanse los apartados 5.3.3 y 7.1.

## Matriz de confusión · mediapipe

| Real \ Predicho | neutral | joy | sadness | anger | fear | surprise | pain |
|---|---|---|---|---|---|---|---|
| neutral | 632 | 82 | 167 | 128 | 77 | 52 | 24 |
| joy | 99 | 1344 | 41 | 84 | 46 | 57 | 12 |
| sadness | 274 | 66 | 337 | 174 | 100 | 35 | 34 |
| anger | 153 | 52 | 89 | 384 | 52 | 57 | 28 |
| fear | 184 | 67 | 143 | 137 | 203 | 136 | 15 |
| surprise | 46 | 30 | 21 | 41 | 56 | 555 | 7 |
| pain | 10 | 4 | 14 | 20 | 4 | 2 | 38 |

## Matriz de confusión · deepface

| Real \ Predicho | neutral | joy | sadness | anger | fear | surprise | pain |
|---|---|---|---|---|---|---|---|
| neutral | 536 | 77 | 164 | 52 | 108 | 14 | 1 |
| joy | 109 | 1064 | 49 | 19 | 75 | 18 | 4 |
| sadness | 161 | 53 | 296 | 74 | 123 | 16 | 3 |
| anger | 96 | 35 | 128 | 286 | 119 | 19 | 11 |
| fear | 101 | 32 | 123 | 62 | 304 | 72 | 4 |
| surprise | 39 | 36 | 25 | 11 | 91 | 454 | 1 |
| pain | 5 | 2 | 12 | 22 | 12 | 0 | 38 |
