#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entrenamiento del clasificador afectivo sobre coeficientes de expresión.

FICHERO PYTHON (.py) de línea de comandos, NO es un nodo ROS 2. En Ubuntu:

    python3 scripts/train_affect_mlp.py \\
        --dataset /ros2_ws/datasets/train \\
        --model   /ros2_ws/src/pedcom_perception/models/face_landmarker.task \\
        --out     /ros2_ws/src/pedcom_perception/models/mlp_affect.npz

Punto clave del diseño (apartado 3.4.2 de la memoria): el entrenamiento no ve
píxeles crudos, sino los 52 coeficientes de expresión que produce MediaPipe. La
extracción de características se hace una sola vez y a partir de ahí el dato ya
no permite reconstruir el rostro. Además, entrenar sobre 52 dimensiones en
lugar de sobre imágenes hace viable un modelo de menos de 30 000 parámetros.

Estructura esperada del conjunto de datos:
    dataset/neutral/*.jpg  dataset/joy/*.jpg  ...  dataset/pain/*.jpg
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "src", "pedcom_perception"))

EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "pain"]
N_FEATURES = 52


def extraer_caracteristicas(dataset_dir: str, model_path: str):
    """Recorre el conjunto de datos y extrae los coeficientes de expresión."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
    )
    landmarker = vision.FaceLandmarker.create_from_options(opts)

    X, y, descartadas = [], [], 0
    for label_idx, label in enumerate(EMOTIONS):
        patron = os.path.join(dataset_dir, label, "*")
        for path in sorted(glob.glob(patron)):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.face_blendshapes:
                descartadas += 1
                continue
            v = np.array([c.score for c in res.face_blendshapes[0]],
                         dtype=np.float32)
            X.append(v[:N_FEATURES])
            y.append(label_idx)
        print(f"  {label}: {sum(1 for k in y if k == label_idx)} muestras")

    print(f"Imágenes sin rostro detectado (descartadas): {descartadas}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def equilibrar(X, y, semilla=2026):
    """Sobremuestrea las clases minoritarias hasta igualar a la mayoritaria.

    Por qué hace falta
    ------------------
    En FER-2013 la clase que aquí se llama «dolor» tiene 436 imágenes de
    entrenamiento frente a las 7 215 de alegría: una proporción de 1 a 18. Un
    MLP entrenado sin corregir ese desequilibrio minimiza el error global
    ignorando la clase minoritaria, y eso fue exactamente lo que ocurrió: en la
    primera evaluación, el clasificador reconoció 3 de los 92 casos de dolor
    (sensibilidad 0,033; F1 0,057). Para un sistema cuyo objetivo declarado es
    detectar malestar, esa cifra invalida el resultado principal.

    `sklearn.neural_network.MLPClassifier` no admite `class_weight`, así que se
    corrige por remuestreo: se replican con reemplazo las muestras de cada clase
    hasta igualar el tamaño de la mayoritaria. Es la opción más simple que deja
    constancia clara en la memoria; alternativas como SMOTE exigirían una
    dependencia adicional y sintetizarían vectores de expresión que no
    corresponden a ningún rostro observado.

    El remuestreo se aplica SOLO al subconjunto de entrenamiento: el de prueba
    conserva la distribución original, porque las métricas deben reflejar la
    prevalencia real de cada categoría.
    """
    rng = np.random.RandomState(semilla)
    clases, cuentas = np.unique(y, return_counts=True)
    objetivo = int(cuentas.max())
    indices = []
    for c in clases:
        idx_c = np.flatnonzero(y == c)
        indices.append(idx_c)
        faltan = objetivo - len(idx_c)
        if faltan > 0:
            indices.append(rng.choice(idx_c, size=faltan, replace=True))
    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return X[idx], y[idx]


def entrenar(X, y, semilla=2026, balancear=True):
    """Entrena el MLP 52-64-32-7 y devuelve pesos y métricas."""
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=semilla, stratify=y)

    if balancear:
        antes = len(y_tr)
        X_tr, y_tr = equilibrar(X_tr, y_tr, semilla)
        print(f"\nEquilibrado de clases: {antes} -> {len(y_tr)} muestras de "
              f"entrenamiento (el conjunto de prueba NO se toca)")
    else:
        print("\nSIN equilibrado de clases: la clase minoritaria quedará "
              "infrarrepresentada y su sensibilidad será muy baja")

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                        alpha=1e-3, max_iter=800, random_state=semilla,
                        early_stopping=True, n_iter_no_change=25)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    print("\n=== Informe de clasificación (conjunto de prueba) ===")
    print(classification_report(y_te, y_pred, target_names=EMOTIONS,
                                zero_division=0))
    print("=== Matriz de confusión ===")
    print(confusion_matrix(y_te, y_pred))

    # IMPORTANTE para la memoria: evaluación estratificada (mitigación de R3).
    # Si el conjunto incluye metadatos de subgrupo, repítase el informe por
    # subgrupo antes de dar por buena la exactitud global.
    return clf


def exportar(clf, out_path: str) -> None:
    """Guarda los pesos en el formato que espera `affect_backend.MLPClassifier`."""
    Ws = {f"W{i}": np.asarray(w, dtype=np.float32)
          for i, w in enumerate(clf.coefs_)}
    bs = {f"b{i}": np.asarray(b, dtype=np.float32)
          for i, b in enumerate(clf.intercepts_)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **Ws, **bs)
    n = sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)
    print(f"\nModelo guardado en {out_path} ({n} parámetros)")


# Raíz del espacio de trabajo, deducida de la ubicación de este fichero. Permite
# ejecutar el script con el botón «Run» de VS Code, sin argumentos y desde
# cualquier directorio de trabajo.
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_DATASET = os.path.join(WS, "datasets", "train")
DEF_MODEL = os.path.join(WS, "src", "pedcom_perception", "models",
                         "face_landmarker.task")
DEF_OUT = os.path.join(WS, "src", "pedcom_perception", "models",
                       "mlp_affect.npz")
DEF_CACHE = os.path.join(WS, "datasets", "features_train.npz")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEF_DATASET,
                    help="carpeta de entrenamiento (una subcarpeta por clase)")
    ap.add_argument("--model", default=DEF_MODEL,
                    help="ruta a face_landmarker.task")
    ap.add_argument("--out", default=DEF_OUT,
                    help="fichero .npz de salida con los pesos del MLP")
    ap.add_argument("--cache", default=DEF_CACHE,
                    help="fichero .npz con características ya extraídas; "
                         "acelera los reentrenamientos")
    ap.add_argument("--sin-equilibrar", action="store_true",
                    dest="sin_equilibrar",
                    help="no sobremuestrear las clases minoritarias. Reproduce "
                         "el entrenamiento original, en el que la clase de "
                         "dolor obtenía F1 = 0,057")
    a = ap.parse_args()

    if not os.path.isdir(a.dataset):
        sys.exit(f"No existe la carpeta del conjunto de entrenamiento:\n  "
                 f"{a.dataset}\nIndique otra con --dataset.")
    if not os.path.isfile(a.model):
        sys.exit(f"No se encuentra el modelo de MediaPipe:\n  {a.model}\n"
                 f"Descárguelo según el apartado 2.4 del README.")
    print(f"conjunto : {a.dataset}")
    print(f"modelo   : {a.model}")
    print(f"salida   : {a.out}")

    if a.cache and os.path.isfile(a.cache):
        d = np.load(a.cache)
        X, y = d["X"], d["y"]
        print(f"Características cargadas de {a.cache}: {X.shape}")
    else:
        print("Extrayendo coeficientes de expresión...")
        X, y = extraer_caracteristicas(a.dataset, a.model)
        if a.cache:
            np.savez(a.cache, X=X, y=y)

    if len(X) < 50:
        sys.exit("Muestras insuficientes para entrenar (mínimo 50).")

    clf = entrenar(X, y, balancear=not a.sin_equilibrar)
    exportar(clf, a.out)


if __name__ == "__main__":
    main()
