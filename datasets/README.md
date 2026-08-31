# Conjuntos de datos

Esta carpeta **no se versiona**. El repositorio contiene el código y los
manifiestos de escenario, pero no las imágenes: son un conjunto público de
terceros, con su propia licencia, y redistribuirlo desde aquí no aporta nada y
sí complica la procedencia de los datos.

Todo lo que falta se reconstruye con los scripts de `scripts/`.

---

## 1. Qué debe haber aquí

```
datasets/
├── train/            imágenes de entrenamiento, una subcarpeta por clase
├── test/             imágenes de prueba, una subcarpeta por clase
├── features_train.npz   características extraídas (lo genera train_affect_mlp.py)
├── E1.txt E2.txt E3.txt manifiestos de escenario  ← estos sí se versionan
└── README.md
```

Las siete clases, en el orden canónico de `EMOTIONS`
(`src/pedcom_perception/affect_backend.py`):

```
neutral   joy   sadness   anger   fear   surprise   pain
```

---

## 2. Origen: FER2013 reetiquetado

El conjunto empleado es **FER2013** (Goodfellow et al., 2013), imágenes en
escala de grises de 48 × 48 px, disponible públicamente para investigación —
por ejemplo en Kaggle, *Challenges in Representation Learning: Facial
Expression Recognition Challenge*.

Sobre sus siete clases originales se aplica el siguiente reetiquetado, que
adapta el vocabulario al dominio clínico descrito en el apartado 4.3 de la
memoria:

| FER2013 | Pediatric-Companion |
|---|---|
| `neutral` | `neutral` |
| `happy` | `joy` |
| `sad` | `sadness` |
| `angry` | `anger` |
| `fear` | `fear` |
| `surprise` | `surprise` |
| `disgust` | `pain` |

> El mapeo `disgust → pain` es una **aproximación deliberada y declarada**, no
> una equivalencia: FER2013 no contiene una clase de dolor. Es la limitación
> principal del clasificador y se discute en los apartados 7.1 y 7.4 de la
> memoria (distancia de dominio: adultos en condiciones neutras frente a
> menores hospitalizados). La clase resultante es además la minoritaria: 436
> imágenes en entrenamiento y 111 en prueba.

Estructura resultante esperada:

```
datasets/train/<clase>/*.jpg      ~28 700 imágenes
datasets/test/<clase>/*.jpg        ~7 200 imágenes
```

---

## 3. Reconstruir los derivados

Una vez colocadas las imágenes en `train/` y `test/`:

### 3.1 Entrenar el clasificador afectivo

```bash
python3 scripts/train_affect_mlp.py \
    --dataset /ros2_ws/datasets/train \
    --model   /ros2_ws/src/pedcom_perception/models/face_landmarker.task \
    --out     /ros2_ws/src/pedcom_perception/models/mlp_affect.npz
```

Genera también la caché de características `features_train.npz`. El modelo
entrenado (`mlp_affect.npz`, 24 KB) **sí está versionado**, de modo que el
sistema funciona tras un `git clone` sin necesidad de reentrenar.

Obsérvese que el entrenamiento no ve píxeles: opera sobre los 52 coeficientes
de expresión que produce MediaPipe, y a partir de ahí el dato ya no permite
reconstruir el rostro (apartado 4.5.1 de la memoria).

### 3.2 Regenerar los manifiestos de escenario

```bash
python3 scripts/make_scenario.py --escenario E1
python3 scripts/make_scenario.py --escenario E2
python3 scripts/make_scenario.py --escenario E3
```

El muestreo usa semilla fija (`--semilla 2026`), de modo que el manifiesto
regenerado coincide con el versionado siempre que el conjunto de origen sea el
mismo. Cada manifiesto es un fichero de texto con una línea
`ruta_imagen;etiqueta` por fotograma, en el orden exacto de reproducción.

### 3.3 Reproducir la comparativa de backends

```bash
python3 scripts/eval_backends.py \
  --dataset datasets/test \
  --backends mediapipe deepface \
  --landmarker src/pedcom_perception/models/face_landmarker.task \
  --weights   src/pedcom_perception/models/mlp_affect.npz \
  --out resultados/
```

---

## 4. Aviso ético

No coloques aquí, ni ejecutes ningún script sobre, **imágenes de pacientes
pediátricos reales**. El trabajo se ha desarrollado y evaluado exclusivamente
con conjuntos públicos de acceso académico e imágenes propias del autor
(apartados 2.3 y 4.3.1 de la memoria). Cualquier uso con datos clínicos exige
consentimiento informado, evaluación de impacto conforme al artículo 35 del
RGPD y la aprobación del comité de ética asistencial correspondiente.
