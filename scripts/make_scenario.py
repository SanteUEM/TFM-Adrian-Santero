#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construye los escenarios de la campaña de validación (capítulo 5).

FICHERO PYTHON (.py) de línea de comandos. NO es un nodo ROS 2:

    python3 scripts/make_scenario.py --escenario E1
    python3 scripts/make_scenario.py --escenario E1 --fps 15 --duracion 90

Por qué existe
--------------
`validation.launch.py` espera un directorio `datasets/<escenario>/` con una
subcarpeta por clase. Si no existe, `sequence_injector_node` recurre a un patrón
sintético que ningún detector facial reconoce: la campaña se ejecuta entera, el
bag se graba, y todas las métricas de percepción salen a cero. Es un fallo que
no se distingue de un mal resultado del algoritmo.

Este script arma el escenario a partir de `datasets/test/` con muestreo
reproducible (semilla fija, requisito RNF-06).

El resultado es un MANIFIESTO: un único fichero de texto con una línea
`ruta_imagen;etiqueta` por fotograma, en el orden exacto de reproducción. No se
copia ni una imagen. Frente a crear una carpeta por escenario, el manifiesto
ocupa unos kilobytes en lugar de duplicar miles de ficheros (importante en una
carpeta sincronizada con la nube), se puede versionar junto al código y deja
constancia auditable de qué se inyectó y en qué orden.

`sequence_injector_node` acepta indistintamente un directorio por clases o un
manifiesto `.txt` en su parámetro `dataset_dir`.

Perfiles disponibles
--------------------
`sequence_injector_node` recorre las clases en el orden canónico de `EMOTIONS`
(neutral, joy, sadness, anger, fear, surprise, pain), de modo que el número de
imágenes por clase define directamente el perfil temporal del índice de
malestar D(t):

  E1  línea base: reparto uniforme; recorre las siete categorías.
  E2  escalada progresiva: poca calma y bloque largo de dolor y miedo; sirve
      para provocar la secuencia IDLE -> ENGAGE -> COMFORT -> CALM_DOWN -> ALERT.
  E3  recuperación: bloque de malestar seguido de un bloque largo de calma;
      verifica la histéresis de salida y los tiempos de permanencia.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "pedcom_perception"))

from pedcom_perception.affect_backend import EMOTIONS       # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff")

# Pesos relativos de cada clase dentro del escenario.
PERFILES = {
    "E1": {"neutral": 1, "joy": 1, "sadness": 1, "anger": 1,
           "fear": 1, "surprise": 1, "pain": 1},
    "E2": {"neutral": 1, "joy": 1, "sadness": 2, "anger": 2,
           "fear": 4, "surprise": 1, "pain": 5},
    "E3": {"neutral": 5, "joy": 3, "sadness": 2, "anger": 1,
           "fear": 2, "surprise": 1, "pain": 3},
}


def listar(carpeta: str) -> list:
    if not os.path.isdir(carpeta):
        return []
    return sorted(n for n in os.listdir(carpeta)
                  if n.lower().endswith(IMG_EXT))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--escenario", default="E1", choices=sorted(PERFILES))
    ap.add_argument("--origen", default=os.path.join(WS, "datasets", "test"))
    ap.add_argument("--destino", default="",
                    help="manifiesto de salida (por defecto datasets/<esc>.txt)")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="debe coincidir con target_fps de perception_node")
    ap.add_argument("--duracion", type=float, default=120.0,
                    help="duración objetivo de la secuencia, en segundos")
    ap.add_argument("--semilla", type=int, default=2026)
    a = ap.parse_args()

    destino = a.destino or os.path.join(WS, "datasets", f"{a.escenario}.txt")
    base = os.path.dirname(os.path.abspath(destino))

    perfil = PERFILES[a.escenario]
    total_objetivo = int(round(a.fps * a.duracion))
    suma_pesos = sum(perfil.values())

    rnd = random.Random(a.semilla)
    cabecera = [
        f"# escenario   : {a.escenario}",
        f"# origen      : {os.path.relpath(a.origen, base)}",
        f"# fps         : {a.fps}",
        f"# duración obj: {a.duracion} s",
        f"# semilla     : {a.semilla}  (RNF-06: regenerable byte a byte)",
        "# formato     : <ruta relativa a este fichero>;<etiqueta>",
        "#",
    ]
    lineas, resumen, total_real = [], [], 0

    for clase in EMOTIONS:
        peso = perfil.get(clase, 0)
        cuota = int(round(total_objetivo * peso / suma_pesos))
        carpeta = os.path.join(a.origen, clase)
        disponibles = listar(carpeta)
        if not disponibles or cuota == 0:
            print(f"  aviso: sin muestras para la clase '{clase}'")
            resumen.append(f"# {clase}: 0")
            continue

        if cuota <= len(disponibles):
            elegidas = rnd.sample(disponibles, cuota)
        else:
            # Menos imágenes que fotogramas pedidos: se repiten en orden. Eso
            # prolonga cada expresión y da tiempo a la fusión temporal
            # (tau = 1,5 s) a estabilizarse antes del cambio de clase.
            elegidas = [disponibles[i % len(disponibles)] for i in range(cuota)]

        rel = os.path.relpath(carpeta, base)
        lineas += [f"{os.path.join(rel, n)};{clase}" for n in elegidas]
        total_real += len(elegidas)
        print(f"  {clase:9s}: {len(elegidas):5d} fotogramas "
              f"({len(elegidas) / a.fps:6.1f} s)")
        resumen.append(f"# {clase}: {len(elegidas)} fotogramas "
                       f"({len(elegidas) / a.fps:.1f} s)")

    cabecera += resumen + [f"# TOTAL: {total_real} fotogramas "
                           f"({total_real / a.fps:.1f} s)", "#"]

    os.makedirs(base, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as fh:
        fh.write("\n".join(cabecera + lineas) + "\n")

    print(f"\nManifiesto escrito: {destino}")
    print(f"{total_real} fotogramas · {total_real / a.fps:.1f} s a {a.fps} Hz")
    print("\nLanzamiento de la campaña (dentro del contenedor):\n"
          f"  ros2 launch pedcom_bringup validation.launch.py \\\n"
          f"      scenario:={a.escenario} "
          f"dataset_dir:=/ros2_ws/datasets/{a.escenario}.txt \\\n"
          f"      fps:={a.fps} record:=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
