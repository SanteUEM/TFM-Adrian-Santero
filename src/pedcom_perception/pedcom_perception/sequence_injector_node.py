#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inyector de secuencias faciales para la campaña de validación (capítulo 5).

FICHERO PYTHON (.py):
    ros2 run pedcom_perception sequence_injector_node --ros-args \\
        -p dataset_dir:=/ros2_ws/datasets/E1 -p fps:=30.0

Publica en el mismo tema que la cámara simulada de Gazebo, de modo que ningún
otro nodo necesita cambiar entre la ejecución en simulación y la campaña
experimental. Publica además la etiqueta de referencia en un tema aparte, lo
que permite calcular exactitud y F1 comparando series temporales alineadas.

Estructura esperada del directorio (una carpeta por categoría):
    dataset_dir/
      neutral/*.png   joy/*.png   sadness/*.png   anger/*.png
      fear/*.png      surprise/*.png              pain/*.png

Si el directorio no existe, el nodo genera un patrón sintético. Esto permite
verificar toda la cadena sin descargar ningún conjunto de datos y sin manejar
imágenes de menores durante el desarrollo.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import Image
from std_msgs.msg import String

from pedcom_perception.affect_backend import EMOTIONS

QOS_SENSOR = QoSProfile(depth=5,
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST,
                        durability=DurabilityPolicy.VOLATILE)


class SequenceInjectorNode(Node):

    def __init__(self) -> None:
        super().__init__("sequence_injector_node")
        self.declare_parameter("dataset_dir", "")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        # Al terminar la secuencia, cerrar el nodo. El lanzamiento de la campaña
        # usa `on_exit=Shutdown()` para detener también la grabación: así el bag
        # cubre la secuencia y no 170 s adicionales de reposo.
        self.declare_parameter("shutdown_on_end", True)
        self.shutdown_on_end = bool(self.get_parameter("shutdown_on_end").value)

        self.fps = float(self.get_parameter("fps").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.w = int(self.get_parameter("width").value)
        self.h = int(self.get_parameter("height").value)

        self.pub_img = self.create_publisher(
            Image, "/pedcom/camera/image_raw", QOS_SENSOR)
        self.pub_gt = self.create_publisher(
            String, "/pedcom/validation/ground_truth", 10)

        self.samples = self._load(self.get_parameter("dataset_dir").value)
        self.idx = 0
        self.timer = self.create_timer(1.0 / self.fps, self.on_timer)
        self.get_logger().info(
            f"sequence_injector_node: {len(self.samples)} muestras a {self.fps} Hz")

    # ------------------------------------------------------------------ datos
    def _load(self, root: str):
        samples = []
        if root and os.path.isfile(root) and root.lower().endswith(".txt"):
            samples = self._load_manifiesto(root)
        elif root and os.path.isdir(root):
            try:
                import cv2
                for label in EMOTIONS:
                    for path in sorted(glob.glob(os.path.join(root, label, "*"))):
                        img = cv2.imread(path, cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        samples.append((self._encuadrar(img, cv2), label))
            except ImportError:
                self.get_logger().error("OpenCV no disponible: se usa patrón sintético")
        elif root:
            # Una ruta mal escrita no debe degradarse en silencio: la
            # campaña seguiría adelante inyectando un patrón sintético y
            # produciría un bag sin un solo rostro, indistinguible a simple
            # vista de un mal resultado de la percepción.
            self.get_logger().error(
                f"La ruta indicada no existe: {root!r}. Debe ser un directorio "
                "con una subcarpeta por clase, o un manifiesto .txt. "
                "Revise el argumento dataset_dir del lanzamiento.")

        if not samples:
            self.get_logger().warn(
                "SECUENCIA SINTÉTICA: el patrón geométrico NO contiene un "
                "rostro y ningún detector facial lo reconocerá. Sirve para "
                "verificar el grafo y las latencias, NO para medir exactitud.")
            samples = [(self._synthetic(i), EMOTIONS[i % len(EMOTIONS)])
                       for i in range(len(EMOTIONS) * 30)]
        return samples

    def _load_manifiesto(self, ruta: str):
        """Carga la secuencia desde un manifiesto `ruta_imagen;etiqueta`.

        Un escenario de campaña son miles de fotogramas. Copiarlos a una carpeta
        por escenario duplicaría el conjunto de datos en disco (y, sobre una
        carpeta sincronizada, provocaría una subida masiva). El manifiesto
        describe la secuencia en un único fichero de texto: ocupa unos kilobytes,
        es legible, se versiona y deja constancia exacta del orden de
        reproducción, que es lo que exige la repetibilidad del RNF-06.

        Las líneas que empiezan por `#` son comentarios. Las rutas relativas se
        resuelven respecto al directorio del propio manifiesto.
        """
        import cv2
        base = os.path.dirname(os.path.abspath(ruta))
        muestras, descartadas = [], 0
        with open(ruta, "r", encoding="utf-8") as fh:
            for linea in fh:
                linea = linea.strip()
                if not linea or linea.startswith("#"):
                    continue
                trozos = linea.split(";")
                if len(trozos) != 2:
                    descartadas += 1
                    continue
                rel, etiqueta = trozos[0].strip(), trozos[1].strip()
                if etiqueta not in EMOTIONS:
                    descartadas += 1
                    continue
                path = rel if os.path.isabs(rel) else os.path.join(base, rel)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    descartadas += 1
                    continue
                muestras.append((self._encuadrar(img, cv2), etiqueta))
        if descartadas:
            self.get_logger().warn(
                f"manifiesto {os.path.basename(ruta)}: {descartadas} líneas "
                f"descartadas (ruta inexistente o etiqueta desconocida)")
        self.get_logger().info(
            f"manifiesto cargado: {len(muestras)} fotogramas desde {ruta}")
        return muestras

    def _encuadrar(self, img, cv2):
        """Coloca el recorte facial centrado en un lienzo de w x h.

        Se conserva la proporción y se deja margen alrededor. Estirar un
        recorte de 48x48 hasta 640x480 deformaba el rostro y lo dejaba pegado a
        los bordes, dos condiciones que penalizan a cualquier detector.
        """
        h0, w0 = img.shape[:2]
        objetivo = int(min(self.w, self.h) * 0.55)      # el rostro ocupa ~55 %
        escala = objetivo / float(max(h0, w0))
        red = cv2.resize(img, (max(1, int(round(w0 * escala))),
                               max(1, int(round(h0 * escala)))),
                         interpolation=cv2.INTER_CUBIC)
        lienzo = np.full((self.h, self.w, 3), 96, dtype=np.uint8)
        y0 = (self.h - red.shape[0]) // 2
        x0 = (self.w - red.shape[1]) // 2
        lienzo[y0:y0 + red.shape[0], x0:x0 + red.shape[1]] = red
        return lienzo

    def _synthetic(self, i: int) -> np.ndarray:
        """Patrón con una elipse clara sobre fondo oscuro (rostro esquemático)."""
        img = np.full((self.h, self.w, 3), 30, dtype=np.uint8)
        cy, cx = self.h // 2, self.w // 2
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        mask = (((xx - cx) / 90.0) ** 2 + ((yy - cy) / 120.0) ** 2) <= 1.0
        img[mask] = (190, 205, 225)
        phase = (i % 30) / 30.0
        img[cy - 40:cy - 30, cx - 45:cx - 15] = int(40 + 60 * phase)
        img[cy - 40:cy - 30, cx + 15:cx + 45] = int(40 + 60 * phase)
        return img

    # --------------------------------------------------------------- publicar
    def on_timer(self) -> None:
        if self.idx >= len(self.samples):
            if not self.loop:
                self.get_logger().info(
                    f"Secuencia terminada: {len(self.samples)} muestras "
                    f"inyectadas a {self.fps} Hz "
                    f"({len(self.samples) / max(self.fps, 1e-6):.1f} s)")
                self.timer.cancel()
                if self.shutdown_on_end:
                    raise SystemExit(0)
                return
            self.idx = 0

        img, label = self.samples[self.idx]
        self.idx += 1

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        self.pub_img.publish(msg)
        self.pub_gt.publish(String(data=label))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SequenceInjectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
