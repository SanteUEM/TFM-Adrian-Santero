#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nodo de fusión temporal e índice de malestar (apartado 4.3.3).

FICHERO PYTHON (.py):
    ros2 run pedcom_behaviour affect_fusion_node

Convierte la señal categórica y ruidosa que produce la percepción en una
variable de control continua y estable, D(t) en [0,1]:

  1. promediado exponencial del vector de probabilidades, ponderado por la
     confianza del detector;
  2. proyección lineal con pesos por categoría;
  3. decaimiento exponencial hacia cero cuando no hay observación reciente,
     de modo que la ausencia de información no se confunda con persistencia
     del malestar.

Suscribe  /pedcom/perception/emotional_state
Publica   /pedcom/affect/distress
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from pedcom_interfaces.msg import DistressLevel, EmotionalState

QOS_STATE = QoSProfile(depth=10,
                       reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST)


class AffectFusionNode(Node):

    def __init__(self) -> None:
        super().__init__("affect_fusion_node")

        self.declare_parameter("tau_s", 1.5)             # constante de suavizado
        self.declare_parameter("decay_tau_s", 3.0)       # decaimiento sin rostro
        self.declare_parameter("min_confidence", 0.55)
        self.declare_parameter("stale_after_s", 2.0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter(
            "weights", [0.0, -0.5, 0.6, 0.55, 0.85, 0.0, 1.0])

        gp = self.get_parameter
        self.tau = float(gp("tau_s").value)
        self.decay_tau = float(gp("decay_tau_s").value)
        self.min_conf = float(gp("min_confidence").value)
        self.stale_after = float(gp("stale_after_s").value)
        self.W = np.array(gp("weights").value, dtype=np.float32)
        rate = float(gp("publish_rate_hz").value)

        self.p_hat = np.zeros(7, dtype=np.float32)
        self.conf_hat = 0.0
        self.t_last_face: float | None = None

        self.sub = self.create_subscription(
            EmotionalState, "/pedcom/perception/emotional_state",
            self.on_state, QOS_STATE)
        self.pub = self.create_publisher(
            DistressLevel, "/pedcom/affect/distress", QOS_STATE)
        self.dt = 1.0 / rate
        self.timer = self.create_timer(self.dt, self.on_timer)
        self.get_logger().info(
            f"affect_fusion_node listo (tau={self.tau}s, umbral conf={self.min_conf})")

    # ------------------------------------------------------------- callbacks
    def on_state(self, msg: EmotionalState) -> None:
        if not msg.face_present or msg.confidence < self.min_conf:
            return
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        dt = 0.0 if self.t_last_face is None else max(t - self.t_last_face, 0.0)
        alpha = 1.0 - math.exp(-dt / self.tau) if dt > 0.0 else 1.0
        w = float(np.clip(alpha * msg.confidence, 0.0, 1.0))
        self.p_hat = (1.0 - w) * self.p_hat + w * np.array(msg.probabilities,
                                                           dtype=np.float32)
        self.conf_hat = (1.0 - w) * self.conf_hat + w * float(msg.confidence)
        self.t_last_face = t

    def on_timer(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        since = (float("inf") if self.t_last_face is None
                 else now - self.t_last_face)
        stale = since > self.stale_after

        if stale:
            self.p_hat *= math.exp(-self.dt / self.decay_tau)
            self.conf_hat *= math.exp(-self.dt / self.decay_tau)

        d = float(np.clip(float(self.W @ self.p_hat), 0.0, 1.0))

        msg = DistressLevel()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "pedcom"
        msg.distress_index = d
        msg.smoothed_confidence = float(self.conf_hat)
        msg.valid = not stale
        msg.seconds_since_face = float(min(since, 1e6))
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AffectFusionNode()
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
