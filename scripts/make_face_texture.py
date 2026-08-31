#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera la textura facial del paciente simulado (apartado 5.2).

FICHERO PYTHON (.py) de línea de comandos. NO es un nodo ROS 2:

    python3 scripts/make_face_texture.py --origen datasets/test/neutral/XXXX.jpg

Por qué existe
--------------
El modelo `patient_face_plane` del mundo `hospital_room.sdf` era un plano de
color piel sin textura. Ningún detector facial puede encontrar un rostro en una
superficie de color uniforme, de modo que la percepción devolvía siempre
`face_present = false` y la máquina de estados no salía de IDLE: la simulación
parecía funcionar, pero no ejercitaba la cadena de percepción.

Este script produce `worlds/materials/textures/patient_face.png`, un lienzo
cuadrado con el rostro centrado y con margen suficiente para que la ventana de
detección tenga recorrido, y COMPRUEBA que un detector lo reconoce antes de
escribirlo. Si el origen no da lugar a una textura detectable, avisa en lugar de
generar en silencio una textura inservible.

Nota metodológica (apartados 2.3 y 4.3.1 de la memoria)
-------------------------------------------------------
La textura por defecto procede de un conjunto público de acceso académico y
cumple una función instrumental: comprobar que la cadena de percepción responde
dentro del simulador. Las métricas de exactitud del capítulo 6 NO se obtienen
del simulador, sino de `sequence_injector_node`, donde la etiqueta de referencia
de cada fotograma es conocida. No debe emplearse aquí ninguna imagen de un menor
identificable.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_DESTINO = os.path.join(WS, "src", "pedcom_bringup", "worlds", "materials",
                           "textures", "patient_face.png")


def elegir_origen(origen: str) -> str:
    if origen:
        return origen
    candidatos = sorted(glob.glob(os.path.join(WS, "datasets", "test",
                                               "neutral", "*.jpg")))
    if not candidatos:
        raise SystemExit("No hay imágenes en datasets/test/neutral y no se ha "
                         "indicado --origen.")
    return candidatos[0]


def construir(origen: str, lado: int, ocupacion: float) -> np.ndarray:
    img = cv2.imread(origen, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"No se ha podido leer la imagen {origen!r}")

    objetivo = int(lado * ocupacion)
    h0, w0 = img.shape[:2]
    escala = objetivo / float(max(h0, w0))
    red = cv2.resize(img, (int(round(w0 * escala)), int(round(h0 * escala))),
                     interpolation=cv2.INTER_CUBIC)

    # Realce suave: las imágenes de 48x48 quedan muy blandas al ampliarlas y el
    # detector se apoya en los gradientes de ojos y boca.
    suave = cv2.GaussianBlur(red, (0, 0), 3.0)
    red = cv2.addWeighted(red, 1.5, suave, -0.5, 0)

    lienzo = np.full((lado, lado, 3), 218, dtype=np.uint8)     # almohada clara
    y0 = (lado - red.shape[0]) // 2
    x0 = (lado - red.shape[1]) // 2
    lienzo[y0:y0 + red.shape[0], x0:x0 + red.shape[1]] = red
    return lienzo


def comprobar(textura: np.ndarray) -> int:
    """Devuelve el número de rostros detectados sobre la textura generada."""
    casc = cv2.CascadeClassifier(os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    gris = cv2.equalizeHist(cv2.cvtColor(textura, cv2.COLOR_BGR2GRAY))
    return len(casc.detectMultiScale(gris, 1.1, 5, minSize=(48, 48)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", default="",
                    help="imagen de partida (por defecto, la primera de "
                         "datasets/test/neutral)")
    ap.add_argument("--destino", default=DEF_DESTINO)
    ap.add_argument("--lado", type=int, default=1024)
    ap.add_argument("--ocupacion", type=float, default=0.62,
                    help="fracción del lado que ocupa el rostro")
    ap.add_argument("--sin-comprobar", action="store_true", dest="sin_comprobar")
    a = ap.parse_args()

    origen = elegir_origen(a.origen)
    print(f"origen  : {origen}")
    textura = construir(origen, a.lado, a.ocupacion)

    if not a.sin_comprobar:
        n = comprobar(textura)
        print(f"detector: {n} rostro(s) encontrado(s) en la textura")
        if n == 0:
            print("\nAVISO: el detector de referencia no encuentra ningún "
                  "rostro en esta textura.\nPruebe con otra imagen de origen o "
                  "ajuste --ocupacion. No se escribe el fichero.\n"
                  "(Use --sin-comprobar para forzar la escritura.)")
            return 1

    os.makedirs(os.path.dirname(a.destino), exist_ok=True)
    cv2.imwrite(a.destino, textura)
    print(f"escrito : {a.destino}  ({a.lado}x{a.lado})")
    print("\nRecuerde reconstruir para que la textura llegue a share/:\n"
          "  colcon build --packages-select pedcom_bringup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
