#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproductor de audio de acompañamiento (apartado 4.5.2).

FICHERO PYTHON (.py):
    ros2 run pedcom_actuation audio_player_node

Reproduce pistas pregrabadas mediante `paplay`/`aplay` si hay salida de audio
disponible; en simulación sin tarjeta de sonido se limita a publicar el estado
de reproducción, que es lo que necesitan las pruebas de integración.

Seguridad (RNF-05): la ganancia se limita al valor equivalente a 60 dB(A) en la
posición nominal del paciente. Como en el controlador de iluminación, el límite
se comprueba aquí de nuevo y no se delega en el emisor.

El sistema NO sintetiza voz ni mantiene diálogo: es una decisión de alcance
(apartado 2.3), no una limitación técnica.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import String

from pedcom_interfaces.msg import MultisensoryCommand

QOS_LATCHED = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST)


class AudioPlayerNode(Node):

    ABS_MAX_GAIN = 0.60           # equivalente a 60 dB(A) a 1,2 m

    def __init__(self) -> None:
        super().__init__("audio_player_node")

        self.declare_parameter("audio_dir", "/ros2_ws/src/pedcom_actuation/audio")
        self.declare_parameter("enable_playback", True)

        self.audio_dir = self.get_parameter("audio_dir").value
        self.enable = bool(self.get_parameter("enable_playback").value)
        self.player = (shutil.which("paplay") or shutil.which("aplay")
                       if self.enable else None)
        self.proc: subprocess.Popen | None = None
        self.current = ""

        self.sub = self.create_subscription(
            MultisensoryCommand, "/pedcom/actuation/command",
            self.on_command, QOS_LATCHED)
        self.pub = self.create_publisher(
            String, "/pedcom/actuation/audio_status", 10)

        modo = self.player if self.player else "simulado (sin salida de audio)"
        self.get_logger().info(
            f"audio_player_node listo · reproductor={modo} · "
            f"ganancia máxima {self.ABS_MAX_GAIN}")

    # ------------------------------------------------------------- callbacks
    def on_command(self, msg: MultisensoryCommand) -> None:
        gain = float(msg.audio_gain)
        if gain > self.ABS_MAX_GAIN:
            self.get_logger().error(
                f"ganancia solicitada {gain:.2f} RECORTADA al límite de seguridad "
                f"{self.ABS_MAX_GAIN:.2f} (RNF-05)")
            gain = self.ABS_MAX_GAIN

        track = msg.audio_track.strip()
        if track == self.current:
            return
        self._stop()
        self.current = track

        if not track:
            self.pub.publish(String(data="stopped"))
            return

        path = os.path.join(self.audio_dir, track)
        if self.player and os.path.isfile(path):
            vol = int(65536 * gain)
            cmd = ([self.player, "--volume", str(vol), path]
                   if self.player.endswith("paplay") else [self.player, path])
            try:
                self.proc = subprocess.Popen(cmd,
                                             stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
                self.pub.publish(String(data=f"playing:{track}:{gain:.2f}"))
                return
            except OSError as exc:
                self.get_logger().warn(f"no se pudo reproducir {track}: {exc}")

        # Modo simulado: se anuncia la pista sin reproducirla
        self.pub.publish(String(data=f"simulated:{track}:{gain:.2f}"))
        self.get_logger().info(f"[simulado] reproduciría {track} a {gain:.2f}")

    def _stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None

    def destroy_node(self) -> bool:
        self._stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AudioPlayerNode()
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
