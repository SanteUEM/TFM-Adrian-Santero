# Comparativa de cadenas de percepción (apartado 6.1)

| Backend | Exactitud | F1 macro | F1 dolor | F1 miedo | Latencia mediana (ms) | Latencia p95 (ms) | No detectados (%) |
|---|---|---|---|---|---|---|---|
| mediapipe (mediapipe) | 0.553 | 0.443 | 0.057 | 0.229 | 18.3 | 29.0 | 10.7 |
| deepface (deepface) | 0.578 | 0.546 | 0.497 | 0.397 | 24.6 | 33.7 | 28.2 |

## Matriz de confusión · mediapipe

| Real \ Predicho | neutral | joy | sadness | anger | fear | surprise | pain |
|---|---|---|---|---|---|---|---|
| neutral | 683 | 94 | 224 | 60 | 46 | 51 | 4 |
| joy | 108 | 1427 | 56 | 33 | 22 | 37 | 0 |
| sadness | 299 | 88 | 458 | 83 | 61 | 28 | 3 |
| anger | 164 | 98 | 149 | 302 | 47 | 51 | 4 |
| fear | 206 | 109 | 198 | 98 | 143 | 131 | 0 |
| surprise | 73 | 55 | 22 | 30 | 44 | 532 | 0 |
| pain | 14 | 9 | 35 | 26 | 3 | 2 | 3 |

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
