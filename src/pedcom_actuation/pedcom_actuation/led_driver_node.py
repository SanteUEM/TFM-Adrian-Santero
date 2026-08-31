#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlador de iluminación LED (apartado 4.5.1).

FICHERO PYTHON (.py):
    ros2 run pedcom_actuation led_driver_node

Dos responsabilidades:

1. Interpolar suavemente entre el color actual y el solicitado (1 s), para que
   los cambios de estado no se perciban como saltos bruscos.
2. Aplicar de forma REDUNDANTE el límite de seguridad de 3 Hz de parpadeo
   (RNF-04). Aunque el gestor de comportamiento ya lo limita, un actuador que
   confía en su emisor no es un actuador seguro: el límite se comprueba aquí
   otra vez, en el punto donde se genera el estímulo físico.

En simulación publica el color en `/pedcom/actuation/led_color` (visualizable en
RViz mediante un marcador). En el prototipo físico este nodo se sustituye por
uno que escriba en el bus del anillo de diodos, sin cambiar la interfaz.
"""

from __future__ import annotations

import colorsys
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from pedcom_interfaces.msg import MultisensoryCommand

QOS_LATCHED = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST)


class LedDriverNode(Node):

    ABS_MAX_BLINK_HZ = 3.0        # límite duro no configurable (seguridad)

    def __init__(self) -> None:
        super().__init__("led_driver_node")

        self.declare_parameter("update_rate_hz", 30.0)
        self.declare_parameter("transition_s", 1.0)
        self.declare_parameter("max_intensity", 0.85)
        self.declare_parameter("publish_marker", True)

        self.rate = float(self.get_parameter("update_rate_hz").value)
        self.transition = float(self.get_parameter("transition_s").value)
        self.max_i = float(self.get_parameter("max_intensity").value)

        self.cur = dict(hue=40.0, sat=0.2, inten=0.0)
        self.tgt = dict(self.cur)
        self.blink_hz = 0.0
        self.t = 0.0

        self.sub = self.create_subscription(
            MultisensoryCommand, "/pedcom/actuation/command",
            self.on_command, QOS_LATCHED)
        self.pub_color = self.create_publisher(
            ColorRGBA, "/pedcom/actuation/led_color", 10)
        self.pub_marker = self.create_publisher(
            Marker, "/pedcom/actuation/led_marker", 10)
        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)
        self.get_logger().info(
            f"led_driver_node listo · límite de parpadeo {self.ABS_MAX_BLINK_HZ} Hz")

    # ------------------------------------------------------------- callbacks
    def on_command(self, msg: MultisensoryCommand) -> None:
        self.tgt = dict(hue=float(msg.hue),
                        sat=float(min(max(msg.saturation, 0.0), 1.0)),
                        inten=float(min(max(msg.intensity, 0.0), self.max_i)))
        solicitado = float(msg.blink_hz)
        if solicitado >= self.ABS_MAX_BLINK_HZ:
            self.get_logger().error(
                f"parpadeo solicitado de {solicitado:.2f} Hz RECHAZADO "
                f"(límite de seguridad {self.ABS_MAX_BLINK_HZ} Hz)")
            solicitado = 0.0
        self.blink_hz = max(0.0, solicitado)

    def on_timer(self) -> None:
        dt = 1.0 / self.rate
        self.t += dt

        # Interpolación exponencial hacia el objetivo
        k = min(1.0, dt / max(self.transition, 1e-3))
        for key in ("hue", "sat", "inten"):
            self.cur[key] += k * (self.tgt[key] - self.cur[key])

        # Modulación de intensidad (respiración luminosa), nunca por debajo del 35 %
        mod = 1.0
        if self.blink_hz > 0.0:
            mod = 0.675 + 0.325 * math.sin(2.0 * math.pi * self.blink_hz * self.t)

        r, g, b = colorsys.hsv_to_rgb((self.cur["hue"] % 360.0) / 360.0,
                                      self.cur["sat"],
                                      self.cur["inten"] * mod)
        color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)
        self.pub_color.publish(color)

        if self.get_parameter("publish_marker").value:
            m = Marker()
            m.header.frame_id = "led_ring_link"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = "pedcom_led", 0, Marker.SPHERE, Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.10
            m.color = color
            self.pub_marker.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LedDriverNode()
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
