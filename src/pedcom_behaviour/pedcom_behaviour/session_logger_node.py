#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registro de sesión seudonimizado (apartado 4.7, requisito RF-08).

FICHERO PYTHON (.py):
    ros2 run pedcom_behaviour session_logger_node

Escribe un CSV con exactamente cuatro tipos de dato: identificador aleatorio de
sesión, tiempo relativo, índice de malestar y estado de comportamiento. No
existe ninguna ruta de código en este nodo capaz de escribir píxeles: ni
siquiera se suscribe al tema de imagen.

Incluye además la política de conservación (30 días por defecto), que se aplica
al arrancar borrando los ficheros caducados.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timedelta, timezone

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from pedcom_interfaces.msg import BehaviourStatus

QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_LAST)

CAMPOS = ["session_id", "t_rel_ms", "distress_index", "behaviour_state",
          "transition_reason", "alert_active"]


class SessionLoggerNode(Node):

    def __init__(self) -> None:
        super().__init__("session_logger_node")

        self.declare_parameter("log_dir", "/ros2_ws/logs/sessions")
        self.declare_parameter("retention_days", 30)
        self.declare_parameter("sample_period_s", 0.5)

        self.log_dir = self.get_parameter("log_dir").value
        self.retention = int(self.get_parameter("retention_days").value)
        self.period = float(self.get_parameter("sample_period_s").value)
        os.makedirs(self.log_dir, exist_ok=True)
        self._apply_retention()

        self.declare_parameter("session_switch_grace_s", 5.0)
        self.grace = float(self.get_parameter("session_switch_grace_s").value)

        self._fh = None
        self._writer = None
        self._session = ""
        self._t0 = 0.0
        self._last_write = 0.0
        self._t_session_start = 0.0
        self._intrusos: set[str] = set()
        self._last: BehaviourStatus | None = None

        self.sub = self.create_subscription(
            BehaviourStatus, "/pedcom/status/behaviour_state", self.on_status, QOS)
        self.get_logger().info(
            f"session_logger_node listo · destino={self.log_dir} · "
            f"conservación={self.retention} días · SIN datos de imagen")

    # -------------------------------------------------------------- retención
    def _apply_retention(self) -> None:
        limite = datetime.now(timezone.utc) - timedelta(days=self.retention)
        borrados = 0
        for name in os.listdir(self.log_dir):
            path = os.path.join(self.log_dir, name)
            if not os.path.isfile(path):
                continue
            mtime = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
            if mtime < limite:
                os.remove(path)
                borrados += 1
        if borrados:
            self.get_logger().warn(
                f"política de conservación: {borrados} registros eliminados")

    # -------------------------------------------------------------- callbacks
    def on_status(self, msg: BehaviourStatus) -> None:
        if not msg.session_id:
            self._close()
            return

        now = time.monotonic()

        if msg.session_id != self._session:
            # Un cambio de sesión legítimo ocurre tras cerrar la anterior o al
            # cabo de un tiempo razonable. Un cambio a los pocos milisegundos
            # significa que hay VARIOS behaviour_manager_node vivos en el mismo
            # ROS_DOMAIN_ID (lanzamientos anteriores sin cerrar): cada uno
            # publica su propio identificador y el registro se fragmentaba en
            # un fichero por mensaje. Se conserva la primera sesión observada y
            # se descarta el resto, dejando constancia una sola vez.
            if (self._fh is not None
                    and now - self._t_session_start < self.grace):
                if msg.session_id not in self._intrusos:
                    self._intrusos.add(msg.session_id)
                    self.get_logger().error(
                        f"identificador de sesión inesperado "
                        f"({msg.session_id[:8]}) mientras {self._session[:8]} "
                        f"sigue activa: hay más de un gestor de comportamiento "
                        f"publicando. Se ignora. Compruebe 'ros2 node list' y "
                        f"cierre los lanzamientos huérfanos.")
                return
            self._open(msg.session_id)

        if now - self._last_write < self.period:
            return
        self._last_write = now

        self._writer.writerow({
            "session_id": msg.session_id,
            "t_rel_ms": int((now - self._t0) * 1000),
            "distress_index": round(float(msg.distress_index), 4),
            "behaviour_state": msg.behaviour_name,
            "transition_reason": msg.transition_reason,
            "alert_active": int(msg.alert_active),
        })
        self._fh.flush()

    # ---------------------------------------------------------------- ficheros
    def _open(self, session_id: str) -> None:
        self._close()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(self.log_dir, f"session_{stamp}_{session_id[:8]}.csv")
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CAMPOS)
        self._writer.writeheader()
        self._session = session_id
        self._t0 = time.monotonic()
        self._t_session_start = self._t0
        self._intrusos.clear()
        self.get_logger().info(f"nueva sesión registrada en {path}")

    def _close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
            self._session = ""

    def destroy_node(self) -> bool:
        self._close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SessionLoggerNode()
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
