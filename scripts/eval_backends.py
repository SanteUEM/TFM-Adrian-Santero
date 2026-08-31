#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluación comparativa de las cadenas de percepción (objetivo OE3).

FICHERO PYTHON (.py) de línea de comandos, NO es un nodo ROS 2. Puede
ejecutarse sin ROS 2 en marcha, porque los backends son Python puro:

    python3 scripts/eval_backends.py --dataset datasets/test \\
                                     --backends mediapipe deepface \\
                                     --out resultados/

Produce exactamente las tablas del apartado 6.1 de la memoria:

  * `comparativa_backends.csv` -> exactitud, F1 macro, F1 por clase y latencia
    (mediana, p95) de cada backend, medidas sobre las MISMAS imágenes y en el
    MISMO equipo, que es la condición necesaria para que la comparación sea
    válida (apartado 5.3.5);
  * `confusion_<backend>.csv`  -> matriz de confusión de cada backend;
  * `report.md`                -> resumen legible con las tablas anteriores.

Estructura esperada del conjunto de prueba (una carpeta por clase, estilo
FER-2013 / CAFE):

    datasets/test/
    ├── neutral/   imagen001.png ...
    ├── joy/
    ├── sadness/
    ├── anger/
    ├── fear/
    ├── surprise/
    └── pain/

AVISO ÉTICO (apartados 2.3 y 4.3.1 de la memoria): este script no debe
ejecutarse sobre imágenes de pacientes reales. Solo se emplean conjuntos
públicos de acceso académico e imágenes propias del autor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "pedcom_perception"))

from pedcom_perception.affect_backend import (EMOTIONS,  # noqa: E402
                                              create_backend,
                                              distress_from_probabilities)

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff")

# Raíz del espacio de trabajo, deducida de la ubicación de este fichero. Permite
# ejecutar el script con el botón «Run» de VS Code, sin argumentos y desde
# cualquier directorio de trabajo.
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_DATASET = os.path.join(WS, "datasets", "test")
DEF_MODEL = os.path.join(WS, "src", "pedcom_perception", "models",
                         "face_landmarker.task")
DEF_WEIGHTS = os.path.join(WS, "src", "pedcom_perception", "models",
                           "mlp_affect.npz")
DEF_OUT = os.path.join(WS, "resultados")


# --------------------------------------------------------------------- datos
def listar_muestras(root: str) -> List[Tuple[str, int]]:
    """Devuelve [(ruta, índice_de_clase), ...] recorriendo una carpeta por clase."""
    muestras: List[Tuple[str, int]] = []
    for idx, clase in enumerate(EMOTIONS):
        carpeta = os.path.join(root, clase)
        if not os.path.isdir(carpeta):
            print(f"  aviso: no existe la carpeta de la clase '{clase}'")
            continue
        for nombre in sorted(os.listdir(carpeta)):
            if nombre.lower().endswith(IMG_EXT):
                muestras.append((os.path.join(carpeta, nombre), idx))
    return muestras


# ------------------------------------------------------------------ métricas
def matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    n = len(EMOTIONS)
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if p >= 0:                      # -1 = rostro no detectado
            m[t, p] += 1
    return m


def metricas_por_clase(m: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Precisión, sensibilidad y F1 por clase a partir de la matriz de confusión.

    Se calculan a mano para no exigir scikit-learn en el equipo mínimo; los
    valores coinciden con `sklearn.metrics.classification_report`.
    """
    out: Dict[str, Dict[str, float]] = {}
    for i, clase in enumerate(EMOTIONS):
        tp = int(m[i, i])
        fp = int(m[:, i].sum() - tp)
        fn = int(m[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[clase] = {"precision": prec, "recall": rec, "f1": f1,
                      "soporte": int(m[i, :].sum())}
    return out


def resumen(m: np.ndarray, por_clase: Dict[str, Dict[str, float]],
            latencias: List[float], no_detectados: int,
            n_total: int) -> Dict[str, float]:
    total = int(m.sum())
    exactitud = float(np.trace(m)) / total if total else 0.0
    # F1 macro: media NO ponderada, para que las clases minoritarias
    # (que son las clínicamente relevantes) pesen igual (apartado 3.2.6).
    f1_macro = float(np.mean([v["f1"] for v in por_clase.values()]))
    hay_lat = bool(latencias)
    lat = np.array(latencias, dtype=float) if hay_lat else np.array([0.0])
    return {
        "exactitud": exactitud,
        "f1_macro": f1_macro,
        "f1_dolor": por_clase["pain"]["f1"],
        "f1_miedo": por_clase["fear"]["f1"],
        "f1_alegria": por_clase["joy"]["f1"],
        "f1_calma": por_clase["neutral"]["f1"],
        "latencia_mediana_ms": float(np.median(lat)),
        "latencia_p95_ms": float(np.percentile(lat, 95)),
        "fps_equivalente": (float(1000.0 / max(float(np.median(lat)), 1e-6))
                            if hay_lat else 0.0),
        "no_detectados_pct": 100.0 * no_detectados / n_total if n_total else 0.0,
        "n_muestras": n_total,
    }


# --------------------------------------------------------------- evaluación
def comprobar_backends(nombres: List[str], landmarker: str, pesos: str,
                       permitir_degradacion: bool) -> None:
    """Verifica ANTES de evaluar que todos los backends son los solicitados.

    La comprobación va delante del bucle a propósito. Evaluar y abortar a mitad
    de camino desperdicia el trabajo ya hecho y, peor, deja sin escribir los
    resultados de los backends que sí funcionaban. Instanciar cada backend
    cuesta un par de segundos; descubrir el problema al final cuesta la campaña
    entera.
    """
    problemas = []
    for nombre in nombres:
        if nombre not in ("mediapipe", "deepface"):
            continue
        backend = create_backend(nombre, landmarker_model=landmarker,
                                 weights_path=pesos)
        print(f"· {nombre:10s} -> efectivo '{backend.mode}'", end="")
        motivo = getattr(backend, "motivo_degradacion", "") or "causa no registrada"

        if backend.mode != nombre:
            print("  DEGRADADO")
            problemas.append((nombre, backend.mode, motivo))
            continue

        # El backend es el pedido, pero puede estar mutilado. DeepFace sin
        # detector propio recibe el fotograma completo: no detecta, solo
        # clasifica, así que su latencia y su tasa de no detectados dejan de
        # ser comparables con las de MediaPipe. Es un sesgo invisible en las
        # tablas, y por eso se comprueba aquí y no después.
        if not getattr(backend, "detector_propio", True):
            print("  INCOMPLETO")
            problemas.append((nombre, backend.mode + " sin detector", motivo))
            continue

        print("  OK")

    if not problemas:
        return

    lineas = ["", "ERROR: comparativa abortada. No se ha evaluado nada.", ""]
    for nombre, efectivo, motivo in problemas:
        lineas += [f"  '{nombre}' se resolvió como '{efectivo}'",
                   f"      motivo: {motivo}"]
        if "sin detector" in efectivo:
            lineas += ["      solución: apt-get install -y opencv-data"]
        elif nombre == "deepface":
            lineas += ["      solución: instalar DeepFace en un entorno "
                       "aislado (python3 -m venv /opt/venv_deepface)",
                       "                NO en el entorno principal: rompe "
                       "MediaPipe por conflicto de protobuf.",
                       "      alternativa: --backends mediapipe  (retirar esa "
                       "columna del apartado 6.1)"]
        else:
            lineas += ["      solución: ejecutar dentro del contenedor "
                       "(bash docker/run.sh),",
                       "                donde MediaPipe está fijado a 0.10.14."]
    lineas += ["", "  Con --permitir-degradacion se generan igualmente las "
                   "tablas, pero NO",
               "  son comparables y no deben aparecer en la memoria.", ""]
    if not permitir_degradacion:
        raise SystemExit("\n".join(lineas))
    print("\n".join(lineas).replace("ERROR: comparativa abortada. "
                                    "No se ha evaluado nada.",
                                    "AVISO: resultados NO comparables."))


def evaluar(nombre_backend: str, muestras: List[Tuple[str, int]],
            landmarker: str, pesos: str) -> Dict:
    import cv2

    backend = create_backend(nombre_backend,
                             landmarker_model=landmarker,
                             weights_path=pesos)
    print(f"\n· backend solicitado='{nombre_backend}' · efectivo='{backend.mode}'")

    y_true, y_pred, latencias = [], [], []
    no_detectados = 0
    err_d = []                       # error del índice de malestar D(t)
    # Predicción por imagen. Permite recalcular después las métricas sobre el
    # subconjunto que todos los backends detectan, que es la única comparación
    # honesta de calidad de clasificación (véase tabla_subconjunto_comun).
    predicciones = []

    for k, (ruta, clase) in enumerate(muestras, 1):
        img = cv2.imread(ruta, cv2.IMREAD_COLOR)
        if img is None:
            continue
        obs = backend.process(img, None)      # sin filtro de ROI en evaluación
        del img                                # PRIVACIDAD: no se conserva
        y_true.append(clase)
        if not obs.face_present:
            no_detectados += 1
            y_pred.append(-1)
        else:
            y_pred.append(obs.dominant_index)
            latencias.append(obs.inference_ms)
            d_ref = distress_from_probabilities(np.eye(len(EMOTIONS))[clase])
            err_d.append(abs(distress_from_probabilities(obs.probabilities) - d_ref))
        predicciones.append({"ruta": os.path.relpath(ruta),
                             "real": int(clase), "pred": int(y_pred[-1])})
        if k % 200 == 0:
            print(f"  {k}/{len(muestras)} imágenes procesadas")

    m = matriz_confusion(np.array(y_true), np.array(y_pred))
    por_clase = metricas_por_clase(m)
    res = resumen(m, por_clase, latencias, no_detectados, len(y_true))
    res["mae_indice_malestar"] = float(np.mean(err_d)) if err_d else 0.0
    res["backend_efectivo"] = backend.mode
    return {"resumen": res, "por_clase": por_clase, "confusion": m.tolist(),
            "predicciones": predicciones}


# ------------------------------------------------- subconjunto común (sesgo)
def tabla_subconjunto_comun(resultados: Dict[str, Dict]) -> List[str]:
    """Recalcula exactitud y F1 sobre las imágenes que TODOS los backends detectan.

    La exactitud global se calcula solo sobre los rostros detectados, así que
    dos backends con tasas de detección distintas no se están midiendo sobre el
    mismo material. El que menos detecta se queda con las caras fáciles
    —frontales, nítidas— y sale beneficiado sin que la tabla lo muestre.

    Esta segunda tabla elimina ese sesgo: mismo conjunto de imágenes para todos.
    Es la comparación que sostiene una afirmación del tipo «A clasifica mejor
    que B»; la tabla principal sigue siendo válida para la cobertura y la
    latencia, que son propiedades de la cadena completa.
    """
    nombres = [n for n, d in resultados.items() if d.get("predicciones")]
    if len(nombres) < 2:
        return []

    detectadas = None
    for n in nombres:
        propias = {p["ruta"] for p in resultados[n]["predicciones"]
                   if p["pred"] >= 0}
        detectadas = propias if detectadas is None else (detectadas & propias)

    total = len(resultados[nombres[0]]["predicciones"])
    lineas = ["", "## Subconjunto común · imágenes detectadas por todos los backends",
              "",
              f"{len(detectadas)} de {total} imágenes ({100.0 * len(detectadas) / total:.1f} %). "
              "Elimina el sesgo de comparar exactitudes calculadas sobre "
              "conjuntos distintos.", "",
              "| Backend | Exactitud | F1 macro | F1 dolor | F1 miedo |",
              "|---|---|---|---|---|"]

    for n in nombres:
        cm = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=int)
        for p in resultados[n]["predicciones"]:
            if p["ruta"] in detectadas and p["pred"] >= 0:
                cm[p["real"], p["pred"]] += 1
        pc = metricas_por_clase(cm)
        total_cm = cm.sum()
        acc = float(np.trace(cm)) / total_cm if total_cm else 0.0
        f1m = float(np.mean([v["f1"] for v in pc.values()]))
        lineas.append(f"| {n} | {acc:.3f} | {f1m:.3f} | "
                      f"{pc['pain']['f1']:.3f} | {pc['fear']['f1']:.3f} |")
    return lineas


# ------------------------------------------------------------------- salidas
def escribir_csv(ruta: str, cabecera: List[str], filas: List[List]) -> None:
    with open(ruta, "w", encoding="utf8") as fh:
        fh.write(";".join(cabecera) + "\n")
        for fila in filas:
            fh.write(";".join(str(c) for c in fila) + "\n")


def escribir_informe(destino: str, resultados: Dict[str, Dict]) -> None:
    lineas = ["# Comparativa de cadenas de percepción (apartado 6.1)", "",
              "| Backend | Exactitud | F1 macro | F1 dolor | F1 miedo | "
              "Latencia mediana (ms) | Latencia p95 (ms) | No detectados (%) |",
              "|---|---|---|---|---|---|---|---|"]
    for nombre, datos in resultados.items():
        r = datos["resumen"]
        lineas.append(
            f"| {nombre} ({r['backend_efectivo']}) | {r['exactitud']:.3f} | "
            f"{r['f1_macro']:.3f} | {r['f1_dolor']:.3f} | {r['f1_miedo']:.3f} | "
            f"{r['latencia_mediana_ms']:.1f} | {r['latencia_p95_ms']:.1f} | "
            f"{r['no_detectados_pct']:.1f} |")
    lineas += tabla_subconjunto_comun(resultados)

    # La advertencia viaja con el informe, no en la cabeza de quien lo generó.
    # Una tabla titulada «F1 dolor» que en realidad mide asco es una afirmación
    # falsa en cuanto el fichero sale de su contexto.
    lineas += [
        "", "---", "",
        "### Advertencia sobre la clase «dolor»", "",
        "FER-2013 no contiene una categoría de dolor. Las imágenes etiquetadas "
        "aquí como `pain` son la clase **disgust** (asco) del conjunto "
        "original, renombrada. La correspondencia se apoya en la proximidad "
        "morfológica de ambas configuraciones faciales —arrugamiento nasal y "
        "elevación del labio superior— pero **no es una equivalencia**.",
        "",
        "En consecuencia, las columnas de dolor de este informe NO acreditan "
        "que el sistema detecte dolor. Acreditan que discrimina una expresión "
        "empleada como aproximación. Cualquier afirmación clínica exigiría una "
        "validación con un conjunto de dolor real (por ejemplo UNBC-McMaster) "
        "y con población pediátrica. Véanse los apartados 5.3.3 y 7.1.",
    ]

    for nombre, datos in resultados.items():
        lineas += ["", f"## Matriz de confusión · {nombre}", "",
                   "| Real \\ Predicho | " + " | ".join(EMOTIONS) + " |",
                   "|---" * (len(EMOTIONS) + 1) + "|"]
        for i, clase in enumerate(EMOTIONS):
            lineas.append("| " + clase + " | " +
                          " | ".join(str(v) for v in datos["confusion"][i]) + " |")
    with open(destino, "w", encoding="utf8") as fh:
        fh.write("\n".join(lineas) + "\n")


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEF_DATASET,
                    help="carpeta del conjunto de prueba (una subcarpeta por clase)")
    ap.add_argument("--backends", nargs="+", default=["mediapipe", "deepface"],
                    help="backends a comparar")
    ap.add_argument("--out", default=DEF_OUT, help="carpeta de salida")
    ap.add_argument("--landmarker", default=DEF_MODEL,
                    help="ruta del modelo .task")
    ap.add_argument("--weights", default=DEF_WEIGHTS,
                    help="ruta de los pesos .npz del MLP")
    ap.add_argument("--limite", type=int, default=0,
                    help="número máximo de imágenes (0 = todas)")
    ap.add_argument("--permitir-degradacion", action="store_true",
                    dest="permitir_degradacion",
                    help="no abortar si el backend efectivo no es el solicitado "
                         "(solo depuración: los resultados no son comparables)")
    ap.add_argument("--anexar", action="store_true",
                    help="fusionar con el resultados.json ya existente en la "
                         "carpeta de salida, en vez de reemplazarlo. Permite "
                         "evaluar cada backend en su propio entorno de Python "
                         "y componer después la tabla comparativa")
    args = ap.parse_args()

    print(f"conjunto : {args.dataset}")
    print(f"modelo   : {args.landmarker}")
    print(f"pesos    : {args.weights}"
          f"{'' if os.path.isfile(args.weights) else '  (no existe: se usará el clasificador de reglas)'}")
    muestras = listar_muestras(args.dataset)
    if not muestras:
        print("ERROR: no se han encontrado imágenes. Revise la estructura de "
              "carpetas (una por clase).")
        return 1
    if args.limite:
        muestras = muestras[:args.limite]
    print(f"{len(muestras)} imágenes en {len(EMOTIONS)} clases")

    print("\nComprobación previa de los backends:")
    comprobar_backends(args.backends, args.landmarker, args.weights,
                       args.permitir_degradacion)

    os.makedirs(args.out, exist_ok=True)

    # MediaPipe 0.10.14 exige protobuf<5 y TensorFlow (que arrastra DeepFace)
    # exige protobuf>=6.31. Son incompatibles en un mismo intérprete, así que
    # cada backend se evalúa en su propio entorno y las tablas se componen
    # después. Las métricas siguen siendo comparables porque lo que se exige es
    # mismo conjunto, misma máquina y misma instrumentación (apartado 5.3.5),
    # no el mismo proceso de Python.
    resultados: Dict[str, Dict] = {}
    previo = os.path.join(args.out, "resultados.json")
    if args.anexar and os.path.isfile(previo):
        with open(previo, "r", encoding="utf8") as fh:
            resultados = json.load(fh)
        print(f"anexando a resultados previos: {sorted(resultados)}")

    for nombre in args.backends:
        t0 = time.perf_counter()
        resultados[nombre] = evaluar(nombre, muestras, args.landmarker,
                                     args.weights)
        print(f"  completado en {time.perf_counter() - t0:.1f} s")
        escribir_csv(os.path.join(args.out, f"confusion_{nombre}.csv"),
                     ["real\\predicho"] + EMOTIONS,
                     [[EMOTIONS[i]] + fila
                      for i, fila in enumerate(resultados[nombre]["confusion"])])

    campos = ["backend", "backend_efectivo", "exactitud", "f1_macro", "f1_dolor",
              "f1_miedo", "f1_alegria", "f1_calma", "mae_indice_malestar",
              "latencia_mediana_ms", "latencia_p95_ms", "fps_equivalente",
              "no_detectados_pct", "n_muestras"]
    filas = []
    for nombre, datos in resultados.items():
        r = datos["resumen"]
        filas.append([nombre] + [r.get(c, "") for c in campos[1:]])
    escribir_csv(os.path.join(args.out, "comparativa_backends.csv"), campos, filas)
    escribir_informe(os.path.join(args.out, "report.md"), resultados)
    with open(os.path.join(args.out, "resultados.json"), "w", encoding="utf8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False)

    print(f"\nResultados escritos en {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
