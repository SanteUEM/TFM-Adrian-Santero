#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control del cuello: seguimiento del rostro y micromovimientos (apartado 4.5.3).

FICHERO PYTHON (.py):
    ros2 run pedcom_actuation head_motion_node

Combina dos aportaciones sobre las mismas dos articulaciones:

* seguimiento suave del rostro, con la posición del centro de la región de
  interés convertida en consignas angulares y filtrada paso bajo;
* micromovimientos expresivos de baja amplitud, superpuestos, que evitan la
  apariencia de inmovilidad sin distraer al paciente.

La velocidad se limita a la mitad del máximo mecánico para que el movimiento
resulte pausado y previsible, criterio de aceptabilidad habitual en robótica
social con niños.

Publica en `/pedcom/neck_controller/joint_trajectory`, que consume
`joint_trajectory_controller` de `ros2_control` en la simulación de Gazebo.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from pedcom_interfaces.msg import FaceRegion, MultisensoryCommand

QOS_LATCHED = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST)
QOS_RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class HeadMotionNode(Node):

    def __init__(self) -> None:
        super().__init__("head_motion_node")

        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("yaw_limit_deg", 90.0)
        self.declare_parameter("pitch_min_deg", -30.0)
        self.declare_parameter("pitch_max_deg", 45.0)
        self.declare_parameter("max_speed_deg_s", 30.0)   # mitad del máximo
        self.declare_parameter("fov_h_deg", 62.0)
        self.declare_parameter("fov_v_deg", 48.0)
        self.declare_parameter("tracking_gain", 0.35)
        self.declare_parameter("joint_names", ["neck_yaw_joint", "neck_pitch_joint"])

        gp = self.get_parameter
        self.rate = float(gp("rate_hz").value)
        self.yaw_lim = float(gp("yaw_limit_deg").value)
        self.pitch_min = float(gp("pitch_min_deg").value)
        self.pitch_max = float(gp("pitch_max_deg").value)
        self.max_speed = float(gp("max_speed_deg_s").value)
        self.fov_h = float(gp("fov_h_deg").value)
        self.fov_v = float(gp("fov_v_deg").value)
        self.kp = float(gp("tracking_gain").value)
        self.joints = list(gp("joint_names").value)

        self.yaw = 0.0
        self.pitch = 0.0
        self.tgt_yaw = 0.0
        self.tgt_pitch = 0.0
        self.amp = 0.0
        self.period = 0.0
        self.tracking = False
        self.t = 0.0

        self.sub_cmd = self.create_subscription(
            MultisensoryCommand, "/pedcom/actuation/command",
            self.on_command, QOS_LATCHED)
        self.sub_face = self.create_subscription(
            FaceRegion, "/pedcom/perception/face_roi", self.on_face, QOS_RELIABLE)
        self.pub = self.create_publisher(
            JointTrajectory, "/pedcom/neck_controller/joint_trajectory", 10)
        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)
        self.get_logger().info("head_motion_node listo")

    # ------------------------------------------------------------- callbacks
    def on_command(self, msg: MultisensoryCommand) -> None:
        self.amp = float(msg.motion_amplitude_deg)
        self.period = float(msg.motion_period_s)
        # En reposo y parada segura el cuello vuelve al centro
        if msg.behaviour_state in (MultisensoryCommand.IDLE,
                                   MultisensoryCommand.SAFE_STOP):
            self.tracking = False
            self.tgt_yaw = self.tgt_pitch = 0.0
        else:
            self.tracking = True

    def on_face(self, msg: FaceRegion) -> None:
        if not (self.tracking and msg.face_present):
            return
        # center_u / center_v en [-1,1] -> error angular respecto del eje óptico
        err_yaw = -msg.center_u * (self.fov_h / 2.0)
        err_pitch = -msg.center_v * (self.fov_v / 2.0)
        self.tgt_yaw = clamp(self.yaw + self.kp * err_yaw,
                             -self.yaw_lim, self.yaw_lim)
        self.tgt_pitch = clamp(self.pitch + self.kp * err_pitch,
                               self.pitch_min, self.pitch_max)

    def on_timer(self) -> None:
        dt = 1.0 / self.rate
        self.t += dt
        step = self.max_speed * dt

        self.yaw += clamp(self.tgt_yaw - self.yaw, -step, step)
        self.pitch += clamp(self.tgt_pitch - self.pitch, -step, step)

        # Micromovimiento expresivo superpuesto (no altera la consigna base)
        micro = 0.0
        if self.amp > 0.0 and self.period > 0.0:
            micro = self.amp * math.sin(2.0 * math.pi * self.t / self.period)

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.joints
        pt = JointTrajectoryPoint()
        pt.positions = [math.radians(self.yaw),
                        math.radians(clamp(self.pitch + micro,
                                           self.pitch_min, self.pitch_max))]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(dt * 1e9)
        traj.points = [pt]
        self.pub.publish(traj)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HeadMotionNode()
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
