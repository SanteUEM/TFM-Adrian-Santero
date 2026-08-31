#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guardián de privacidad: auditoría activa del grafo (apartado 4.7).

FICHERO PYTHON (.py):
    ros2 run pedcom_behaviour privacy_guard_node

Comprueba periódicamente, mediante la API de introspección de ROS 2, que nadie
ajeno se haya suscrito al tema de imagen. Si detecta un suscriptor no
autorizado —una herramienta de grabación, un nodo de depuración olvidado o un
proceso malicioso— solicita la parada segura y lo registra.

Es la diferencia entre una política de uso («no grabéis imágenes») y una
garantía técnica verificable, que es lo que exige el artículo 25 del RGPD.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

from pedcom_interfaces.srv import SetBehaviour

SAFE_STOP = 5


class PrivacyGuardNode(Node):

    def __init__(self) -> None:
        super().__init__("privacy_guard_node")

        self.declare_parameter("image_topic", "/pedcom/camera/image_raw")
        self.declare_parameter(
            "allowed_subscribers",
            ["perception_node", "privacy_guard_node"])
        self.declare_parameter("audit_period_s", 2.0)
        self.declare_parameter("audit_log", "/ros2_ws/logs/privacy_audit.jsonl")
        self.declare_parameter("enforce", True)

        gp = self.get_parameter
        self.topic = gp("image_topic").value
        self.allowed = set(gp("allowed_subscribers").value)
        self.log_path = gp("audit_log").value
        self.enforce = bool(gp("enforce").value)

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.cli = self.create_client(SetBehaviour, "/pedcom/set_behaviour")
        # Segunda vía de corte, independiente del gestor: al desactivar la
        # percepción, perception_node destruye su suscripción y, como el puente
        # de Gazebo es `lazy`, deja de entrar vídeo en el grafo ROS 2. Cortar la
        # fuente es más eficaz que pedir a los actuadores que se apaguen.
        self.cli_perc = self.create_client(SetBool,
                                           "/pedcom/perception/enable")
        self.timer = self.create_timer(float(gp("audit_period_s").value),
                                       self.audit)
        self._reported: set[str] = set()
        self.get_logger().info(
            f"privacy_guard_node vigilando {self.topic} · "
            f"autorizados={sorted(self.allowed)}")

    # ------------------------------------------------------------- auditoría
    def audit(self) -> None:
        try:
            info = self.get_subscriptions_info_by_topic(self.topic)
        except Exception as exc:                       # noqa: BLE001
            self.get_logger().debug(f"introspección no disponible: {exc}")
            return

        current = {i.node_name for i in info}
        intruders = sorted(current - self.allowed)

        if not intruders:
            self._reported.clear()
            return

        nuevos = [n for n in intruders if n not in self._reported]
        if not nuevos:
            return
        self._reported.update(nuevos)

        self.get_logger().error(
            f"SUSCRIPTOR NO AUTORIZADO al flujo de imagen: {nuevos}")
        self._write_event("UNAUTHORIZED_SUBSCRIBER", nuevos)

        if self.enforce:
            self._disable_perception()
            self._request_safe_stop()

    def _disable_perception(self) -> None:
        """Corta la fuente: sin suscriptor no hay vídeo en el grafo."""
        if not self.cli_perc.wait_for_service(timeout_sec=0.5):
            self.get_logger().error(
                "no se pudo contactar con /pedcom/perception/enable")
            return
        req = SetBool.Request()
        req.data = False
        self.cli_perc.call_async(req)
        self._write_event("PERCEPTION_DISABLED", [])
        self.get_logger().error(
            "PERCEPCIÓN CORTADA por motivo de privacidad")

    def _request_safe_stop(self) -> None:
        if not self.cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().error(
                "no se pudo contactar con /pedcom/set_behaviour")
            return
        req = SetBehaviour.Request()
        req.requested_state = SAFE_STOP
        req.force = True
        req.disable_perception = True
        req.operator_id = "privacy_guard"
        self.cli.call_async(req)
        self._write_event("SAFE_STOP_REQUESTED", [])
        self.get_logger().error("solicitada PARADA SEGURA por motivo de privacidad")

    def _write_event(self, event: str, nodes) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "topic": self.topic,
            "nodes": nodes,
        }
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PrivacyGuardNode()
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
