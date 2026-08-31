#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nodo de percepción de Pediatric-Companion (rclpy).

FICHERO PYTHON (.py) — se ejecuta como nodo ROS 2:
    ros2 run pedcom_perception perception_node

Por qué un único nodo y no tres
-------------------------------
La memoria describe la percepción como tres componentes (captura, detección y
estimación) cargados en un mismo contenedor con comunicación intraproceso. En
C++ eso se consigue con `rclcpp_components`; en Python la comunicación
intraproceso con copia cero no está disponible, de modo que publicar la imagen
recortada entre nodos la convertiría en tráfico DDS observable — exactamente lo
que el requisito RNF-02 prohíbe.

La solución adoptada mantiene la garantía de privacidad: los tres pasos se
ejecutan dentro de ESTE proceso y el fotograma nunca se publica. Hacia el
exterior solo salen `FaceRegion` (geometría, sin píxeles) y `EmotionalState`
(siete probabilidades). El fotograma se libera al terminar cada ciclo.

Temas publicados
    /pedcom/perception/face_roi          pedcom_interfaces/FaceRegion
    /pedcom/perception/emotional_state   pedcom_interfaces/EmotionalState
Tema suscrito
    /pedcom/camera/image_raw             sensor_msgs/Image   (best effort)
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool

from pedcom_interfaces.msg import EmotionalState, FaceRegion

from pedcom_perception.affect_backend import EMOTIONS, create_backend

# Perfiles de calidad de servicio (apartado 4.6.3 de la memoria)
QOS_SENSOR = QoSProfile(depth=5,
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST,
                        durability=DurabilityPolicy.VOLATILE)
QOS_STATE = QoSProfile(depth=10,
                       reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST,
                       durability=DurabilityPolicy.VOLATILE)


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """Conversión sensor_msgs/Image -> ndarray BGR sin depender de cv_bridge.

    Evitar cv_bridge reduce dependencias y hace el nodo más fácil de probar.
    """
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if enc == "rgb8" else img.copy()
    if enc in ("bgra8", "rgba8"):
        img = buf.reshape(msg.height, msg.width, 4)[:, :, :3]
        return img[:, :, ::-1].copy() if enc == "rgba8" else img.copy()
    if enc in ("mono8", "8uc1"):
        gray = buf.reshape(msg.height, msg.width)
        return np.repeat(gray[:, :, None], 3, axis=2)
    raise ValueError(f"Codificación de imagen no soportada: {msg.encoding}")


class PerceptionNode(Node):

    def __init__(self) -> None:
        super().__init__("perception_node")

        # --- parámetros declarados (configurables desde config/perception.yaml)
        self.declare_parameter("backend", "auto")   # mediapipe | deepface | auto
        self.declare_parameter("landmarker_model", "")
        self.declare_parameter("classifier_weights", "")
        self.declare_parameter("min_detection_confidence", 0.5)
        self.declare_parameter("target_fps", 15.0)
        self.declare_parameter("patient_roi", [-1.0, -1.0, 1.0, 1.0])
        self.declare_parameter("enabled_on_startup", False)   # RNF-07

        gp = self.get_parameter
        self.backend = create_backend(
            gp("backend").value,
            landmarker_model=gp("landmarker_model").value,
            weights_path=gp("classifier_weights").value,
            min_detection_confidence=gp("min_detection_confidence").value,
        )
        self.min_period = 1.0 / max(float(gp("target_fps").value), 1.0)
        self.roi = tuple(float(v) for v in gp("patient_roi").value)
        self.enabled = bool(gp("enabled_on_startup").value)

        # --- comunicaciones
        # RNF-02/RNF-07: la suscripción al vídeo NO se crea aquí. Solo existe
        # mientras la percepción está habilitada; véase _apply_enabled(). Si se
        # creara siempre y se descartaran los fotogramas en el callback, las
        # imágenes seguirían viajando por DDS hasta este proceso y «percepción
        # desactivada» sería una convención interna, no una garantía.
        self.sub = None
        self.pub_state = self.create_publisher(
            EmotionalState, "/pedcom/perception/emotional_state", QOS_STATE)
        self.pub_roi = self.create_publisher(
            FaceRegion, "/pedcom/perception/face_roi", QOS_STATE)
        self.srv = self.create_service(
            SetBool, "/pedcom/perception/enable", self.on_enable)

        self._last_proc = 0.0
        self._apply_enabled(self.enabled)
        self.get_logger().info(
            f"perception_node listo · backend={self.backend.mode} · "
            f"clasificador={'MLP entrenado' if self.backend.clf.ready else 'reglas'} · "
            f"percepción {'ACTIVA' if self.enabled else 'DESACTIVADA (RNF-07)'}")

    # ------------------------------------------------------------- servicios
    def _apply_enabled(self, enabled: bool) -> None:
        """Crea o destruye la suscripción al vídeo según el consentimiento.

        Es el punto en el que el permiso del personal clínico se traduce en una
        modificación real del grafo ROS 2. Comprobable desde otra terminal:

            ros2 topic info /pedcom/camera/image_raw --verbose
        """
        self.enabled = bool(enabled)
        if self.enabled and self.sub is None:
            self.sub = self.create_subscription(
                Image, "/pedcom/camera/image_raw", self.on_image, QOS_SENSOR)
        elif not self.enabled and self.sub is not None:
            self.destroy_subscription(self.sub)
            self.sub = None

    def on_enable(self, request: SetBool.Request,
                  response: SetBool.Response) -> SetBool.Response:
        self._apply_enabled(request.data)
        response.success = True
        response.message = ("percepción activada" if self.enabled
                            else "percepción desactivada")
        self.get_logger().warn(f"[control clínico] {response.message}")
        return response

    # ------------------------------------------------------------ callbacks
    def on_image(self, msg: Image) -> None:
        if not self.enabled:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        # Tolerancia del 10 % sobre el periodo objetivo. Sin ella, cuando la
        # cámara publica exactamente a la frecuencia objetivo (15 Hz frente a
        # 15 Hz), el más mínimo adelanto de un fotograma lo hace caer por
        # debajo del umbral y se descarta; el siguiente llega ya con un periodo
        # doble. El resultado es un aliasing que reduce la tasa efectiva a la
        # mitad o menos sin que nada lo indique.
        if now - self._last_proc < self.min_period * 0.9:
            return                      # limitación de tasa (control de CPU)
        self._last_proc = now

        try:
            frame = imgmsg_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return

        obs = self.backend.process(frame, self.roi)

        # PRIVACIDAD: el fotograma se libera aquí y no se publica ni se guarda.
        del frame

        roi_msg = FaceRegion()
        roi_msg.header = msg.header          # sello temporal DE CAPTURA
        roi_msg.face_present = obs.face_present
        roi_msg.x, roi_msg.y, roi_msg.width, roi_msg.height = obs.bbox
        roi_msg.confidence = float(obs.confidence)
        roi_msg.center_u, roi_msg.center_v = (float(obs.center_uv[0]),
                                              float(obs.center_uv[1]))
        self.pub_roi.publish(roi_msg)

        st = EmotionalState()
        st.header = msg.header
        st.face_present = obs.face_present
        st.probabilities = [float(x) for x in obs.probabilities]
        st.dominant_index = int(obs.dominant_index)
        st.dominant_emotion = EMOTIONS[st.dominant_index]
        st.confidence = float(obs.confidence)
        st.inference_ms = float(obs.inference_ms)
        self.pub_state.publish(st)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
