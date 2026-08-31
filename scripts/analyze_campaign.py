#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Análisis de la campaña de validación (capítulo 5 de la memoria).

FICHERO PYTHON (.py) de línea de comandos, NO es un nodo ROS 2. En Ubuntu:

    source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash
    python3 scripts/analyze_campaign.py --bag /ros2_ws/bags/E1 --out resultados/

Lee un registro `rosbag2` (formato MCAP o SQLite3), alinea las series
temporales por sello temporal y produce exactamente las tablas 16 a 19 de la
memoria:

  * bloque A: exactitud, F1 macro, precisión y sensibilidad de la clase dolor,
    y matriz de confusión;
  * bloque B: secuencia de estados observada y transiciones espurias;
  * bloque C: latencias (mediana, p95, p99) y frecuencias.

Salida: `metrics.csv`, `confusion_matrix.csv` y `report.md`.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "pain"]


# --------------------------------------------------------------- lectura bag
def read_bag(bag_dir: str) -> Dict[str, List[tuple]]:
    """Devuelve {tema: [(t_ns, mensaje), ...]} usando la API de rosbag2."""
    import rclpy.serialization as ser
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message

    storage = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="")
    converter = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)

    tipos = {t.name: t.type for t in reader.get_all_topics_and_types()}
    datos: Dict[str, List[tuple]] = defaultdict(list)
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        msg = ser.deserialize_message(raw, get_message(tipos[topic]))
        datos[topic].append((t_ns, msg))
    return datos


# ------------------------------------------------------------------ bloque A
def bloque_a(datos) -> dict:
    estados = datos.get("/pedcom/perception/emotional_state", [])
    gt = datos.get("/pedcom/validation/ground_truth", [])
    if not estados or not gt:
        return {}

    gt_t = np.array([t for t, _ in gt], dtype=np.int64)
    gt_v = [m.data for _, m in gt]

    y_true, y_pred = [], []
    for t, m in estados:
        if not m.face_present:
            continue
        i = int(np.searchsorted(gt_t, t))
        i = min(max(i - 1, 0), len(gt_v) - 1)
        y_true.append(gt_v[i])
        y_pred.append(m.dominant_emotion)

    if not y_true:
        return {}

    cm = np.zeros((7, 7), dtype=int)
    idx = {e: i for i, e in enumerate(EMOTIONS)}
    for a, b in zip(y_true, y_pred):
        if a in idx and b in idx:
            cm[idx[a], idx[b]] += 1

    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else 0.0

    f1s = []
    for i in range(7):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)

    p = idx["pain"]
    tp = cm[p, p]
    prec_dolor = tp / cm[:, p].sum() if cm[:, p].sum() else 0.0
    rec_dolor = tp / cm[p, :].sum() if cm[p, :].sum() else 0.0
    no_detectados = sum(1 for _, m in estados if not m.face_present) / len(estados)

    return {"exactitud": acc, "f1_macro": float(np.mean(f1s)),
            "precision_dolor": float(prec_dolor),
            "sensibilidad_dolor": float(rec_dolor),
            "rostros_no_detectados_pct": 100.0 * no_detectados,
            "matriz_confusion": cm}


# ------------------------------------------------------------------ bloque B
def bloque_b(datos) -> dict:
    st = datos.get("/pedcom/status/behaviour_state", [])
    if not st:
        return {}
    secuencia, tiempos = [], []
    for t, m in st:
        if not secuencia or secuencia[-1] != m.behaviour_name:
            secuencia.append(m.behaviour_name)
            tiempos.append(t * 1e-9)

    espurias = 0
    for i in range(2, len(secuencia)):
        if (secuencia[i] == secuencia[i - 2] and secuencia[i] != secuencia[i - 1]
                and (tiempos[i] - tiempos[i - 2]) < 3.0):
            espurias += 1
    duracion_min = ((st[-1][0] - st[0][0]) * 1e-9) / 60.0 or 1.0

    return {"secuencia": " -> ".join(secuencia),
            "n_transiciones": len(secuencia) - 1,
            "transiciones_espurias_min": espurias / duracion_min}


# ------------------------------------------------------------------ bloque C
def bloque_c(datos) -> dict:
    res = {}

    est = datos.get("/pedcom/perception/emotional_state", [])
    if est:
        inf = np.array([m.inference_ms for _, m in est], dtype=float)
        res["latencia_percepcion_ms"] = _stats(inf)
        t = np.array([t * 1e-9 for t, _ in est])
        if len(t) > 2:
            res["frecuencia_percepcion_hz"] = float(1.0 / np.mean(np.diff(t)))

    # Latencia extremo a extremo: captura (header.stamp) -> orden de actuación
    cmds = datos.get("/pedcom/actuation/command", [])
    if est and cmds:
        e2e = []
        est_stamp = [(t, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
                     for t, m in est]
        for t_cmd, _ in cmds:
            anteriores = [s for t, s in est_stamp if t <= t_cmd]
            if anteriores:
                e2e.append((t_cmd * 1e-9 - anteriores[-1]) * 1000.0)
        if e2e:
            res["latencia_e2e_ms"] = _stats(np.array(e2e))
    return res


def _stats(v: np.ndarray) -> dict:
    return {"mediana": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99))}


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", required=True, help="directorio del rosbag2")
    ap.add_argument("--out", default="resultados", help="directorio de salida")
    ap.add_argument("--scenario", default="E1")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    datos = read_bag(a.bag)
    A, B, C = bloque_a(datos), bloque_b(datos), bloque_c(datos)

    if "matriz_confusion" in A:
        cm = A.pop("matriz_confusion")
        with open(os.path.join(a.out, "confusion_matrix.csv"), "w",
                  encoding="utf-8") as fh:
            fh.write("real\\predicho," + ",".join(EMOTIONS) + "\n")
            for i, e in enumerate(EMOTIONS):
                fh.write(e + "," + ",".join(str(x) for x in cm[i]) + "\n")

    lineas = [f"# Resultados de la campaña · escenario {a.scenario}", ""]
    for titulo, bloque in (("Bloque A · percepción", A),
                           ("Bloque B · comportamiento", B),
                           ("Bloque C · rendimiento", C)):
        lineas += [f"## {titulo}", "", "| Métrica | Valor |", "|---|---|"]
        for k, v in bloque.items():
            texto = (", ".join(f"{kk}={vv:.1f}" for kk, vv in v.items())
                     if isinstance(v, dict) else
                     (f"{v:.4f}" if isinstance(v, float) else str(v)))
            lineas.append(f"| {k} | {texto} |")
        lineas.append("")

    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas))
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
