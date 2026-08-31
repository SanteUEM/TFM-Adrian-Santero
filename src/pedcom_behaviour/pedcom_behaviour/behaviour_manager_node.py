#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gestor de comportamiento: nodo de ciclo de vida gestionado (apartado 4.4).

FICHERO PYTHON (.py):
    ros2 run pedcom_behaviour behaviour_manager_node

Es un `LifecycleNode` y eso no es un detalle técnico: en el estado
`unconfigured` no existe ninguna suscripción a los temas de percepción, de modo
que «sistema desactivado» es un estado real del grafo y no una convención.
Así se materializa el requisito RNF-07 exigido por el RGPD.

Ciclo de vida (comandos de Ubuntu, en el contenedor):
    ros2 lifecycle set /behaviour_manager_node configure
    ros2 lifecycle set /behaviour_manager_node activate
    ros2 lifecycle set /behaviour_manager_node deactivate

Suscribe  /pedcom/affect/distress, /pedcom/perception/face_roi
Publica   /pedcom/actuation/command, /pedcom/status/behaviour_state
Servicios /pedcom/set_behaviour, /pedcom/start_session
"""

from __future__ import annotations

import uuid
from typing import Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from pedcom_interfaces.msg import (BehaviourStatus, DistressLevel, FaceRegion,
                                   MultisensoryCommand)
from pedcom_interfaces.srv import SetBehaviour, StartSession

from pedcom_behaviour.fsm import (ALERT, CALM_DOWN, COMFORT, ENGAGE, IDLE,
                                  SAFE_STOP, STATE_NAMES, BehaviourFSM,
                                  ContextBuilder, Thresholds, Timers)

QOS_RELIABLE = QoSProfile(depth=10,
                          reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST)
# Los actuadores y el panel deben recibir la última orden vigente al conectarse.
QOS_LATCHED = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST)

# Correspondencia estado -> respuesta multisensorial (tabla 10 de la memoria).
# hue, saturación, intensidad, parpadeo (Hz), pista de audio, ganancia,
# amplitud de movimiento (grados), período (s)
RESPONSES = {
    IDLE:      dict(hue=40.0,  sat=0.20, inten=0.15, blink=0.0,
                    track="",                  gain=0.0, amp=0.0, per=0.0),
    ENGAGE:    dict(hue=40.0,  sat=0.25, inten=0.60, blink=0.0,
                    track="greeting.wav",      gain=0.45, amp=0.0, per=0.0),
    COMFORT:   dict(hue=180.0, sat=0.45, inten=0.50, blink=0.2,
                    track="ambient_loop.wav",  gain=0.45, amp=5.0, per=8.0),
    CALM_DOWN: dict(hue=35.0,  sat=0.70, inten=0.70, blink=0.1,
                    track="breathing_6rpm.wav", gain=0.50, amp=8.0, per=10.0),
    ALERT:     dict(hue=35.0,  sat=0.70, inten=0.70, blink=0.1,
                    track="breathing_6rpm.wav", gain=0.50, amp=8.0, per=10.0),
    SAFE_STOP: dict(hue=0.0,   sat=0.0,  inten=0.0,  blink=0.0,
                    track="",                  gain=0.0, amp=0.0, per=0.0),
}


class BehaviourManagerNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("behaviour_manager_node")

        # Umbrales y temporizadores configurables desde config/behaviour.yaml
        self.declare_parameter("threshold_engage", 0.30)
        self.declare_parameter("threshold_comfort_exit", 0.25)
        self.declare_parameter("threshold_calm", 0.60)
        self.declare_parameter("threshold_calm_exit", 0.50)
        self.declare_parameter("threshold_alert", 0.85)
        self.declare_parameter("dwell_calm_s", 3.0)
        self.declare_parameter("dwell_alert_s", 10.0)
        self.declare_parameter("dwell_exit_s", 5.0)
        self.declare_parameter("min_dwell_s", 2.0)
        self.declare_parameter("watchdog_s", 1.0)
        self.declare_parameter("tick_hz", 20.0)
        self.declare_parameter("max_blink_hz", 3.0)     # RNF-04
        self.declare_parameter("max_audio_gain", 0.6)   # RNF-05

        self.fsm: Optional[BehaviourFSM] = None
        self.ctxb: Optional[ContextBuilder] = None
        self.session_id = ""
        self.distress = 0.0
        self.valid = False
        self.face_present = False
        self.t_last_msg: Optional[float] = None
        self.ack = False
        self.forced: Optional[int] = None
        self.fault = False
        self._subs = []
        self._timer = None

        # Los servicios se crean AQUÍ, no en on_configure. La parada segura debe
        # poder solicitarse en cualquier estado del ciclo de vida: si el único
        # camino para exigirla apareciera al configurar, privacy_guard_node
        # detectaría una fuga en `unconfigured` y no tendría a quién pedir que
        # se corte. Un mecanismo de seguridad que solo existe cuando el sistema
        # ya está en marcha no es un mecanismo de seguridad.
        self.srv_set = self.create_service(
            SetBehaviour, "/pedcom/set_behaviour", self.on_set_behaviour)
        self.srv_session = self.create_service(
            StartSession, "/pedcom/start_session", self.on_start_session)

    # =================================================== transiciones de vida
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        gp = self.get_parameter
        th = Thresholds(engage=gp("threshold_engage").value,
                        comfort_exit=gp("threshold_comfort_exit").value,
                        calm=gp("threshold_calm").value,
                        calm_exit=gp("threshold_calm_exit").value,
                        alert=gp("threshold_alert").value)
        tm = Timers(dwell_calm=gp("dwell_calm_s").value,
                    dwell_alert=gp("dwell_alert_s").value,
                    dwell_exit=gp("dwell_exit_s").value,
                    min_dwell=gp("min_dwell_s").value,
                    watchdog=gp("watchdog_s").value)
        now = self._now()
        self.fsm = BehaviourFSM(th, tm, t0=now)
        self.ctxb = ContextBuilder(th)

        self.pub_cmd = self.create_lifecycle_publisher(
            MultisensoryCommand, "/pedcom/actuation/command", QOS_LATCHED)
        self.pub_status = self.create_lifecycle_publisher(
            BehaviourStatus, "/pedcom/status/behaviour_state", QOS_LATCHED)

        self.get_logger().info("configurado: la percepción sigue sin consumirse")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        # Solo aquí se crean las suscripciones: antes no hay tratamiento de datos
        self._subs = [
            self.create_subscription(DistressLevel, "/pedcom/affect/distress",
                                     self.on_distress, QOS_RELIABLE),
            self.create_subscription(FaceRegion, "/pedcom/perception/face_roi",
                                     self.on_face, QOS_RELIABLE),
        ]
        # El vigilante se arma AQUÍ, no en el constructor. Si `t_last_msg`
        # siguiera valiendo None al primer tick, `seconds_since_msg` sería
        # infinito y la FSM entraría en SAFE_STOP a los ~50 ms de activar,
        # antes de que la fusión afectiva haya podido publicar su primer
        # mensaje. Como SAFE_STOP solo se abandona por rearme manual, el
        # sistema quedaba bloqueado y el resultado dependía del orden de
        # arranque de los nodos. Armar el vigilante en el instante de la
        # activación concede exactamente un periodo de gracia igual a
        # `watchdog_s`: si en ese plazo no llega ningún dato, salta igual.
        self.t_last_msg = self._now()
        hz = float(self.get_parameter("tick_hz").value)
        self._timer = self.create_timer(1.0 / hz, self.on_tick)
        self.get_logger().warn("ACTIVADO: comienza el acompañamiento")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._teardown()
        self._publish_command(SAFE_STOP)     # apaga actuadores al desactivar
        self.get_logger().warn("DESACTIVADO: cesa el tratamiento de datos")
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def _teardown(self) -> None:
        for s in self._subs:
            self.destroy_subscription(s)
        self._subs = []
        if self._timer is not None:
            self.destroy_timer(self._timer)
            self._timer = None

    # ============================================================= callbacks
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_distress(self, msg: DistressLevel) -> None:
        self.distress = float(msg.distress_index)
        self.valid = bool(msg.valid)
        self.t_last_msg = self._now()

    def on_face(self, msg: FaceRegion) -> None:
        self.face_present = bool(msg.face_present)

    def on_set_behaviour(self, req: SetBehaviour.Request,
                         res: SetBehaviour.Response) -> SetBehaviour.Response:
        if self.fsm is None:
            # Sin configurar no hay FSM que forzar, pero una parada segura debe
            # aceptarse igualmente: el gestor ya está inerte y así lo confirma.
            if int(req.requested_state) == SAFE_STOP:
                res.accepted = True
                res.message = ("parada segura registrada; el gestor estaba "
                               "sin configurar y no trata dato alguno")
                self.get_logger().warn(
                    f"[parada segura:{req.operator_id}] {res.message}")
            else:
                res.accepted = False
                res.message = "el nodo no está configurado"
            return res
        if req.requested_state == ALERT and not req.force:
            res.accepted = False
            res.message = "ALERT no puede forzarse manualmente"
            return res
        self.forced = int(req.requested_state)
        self.ack = (req.requested_state == CALM_DOWN)   # confirmación de aviso
        res.accepted = True
        res.message = f"estado forzado a {STATE_NAMES[self.forced]}"
        self.get_logger().warn(
            f"[control clínico:{req.operator_id}] {res.message}")
        return res

    def on_start_session(self, req: StartSession.Request,
                         res: StartSession.Response) -> StartSession.Response:
        if req.start:
            # Identificador aleatorio, sin vínculo con la historia clínica (4.7)
            self.session_id = str(uuid.uuid4())
            res.message = "sesión iniciada"
        else:
            self.session_id = ""
            res.message = "sesión cerrada"
        res.accepted = True
        res.session_id = self.session_id
        return res

    # ================================================================== ciclo
    def on_tick(self) -> None:
        if self.fsm is None or self.ctxb is None:
            return
        now = self._now()
        since = (float("inf") if self.t_last_msg is None
                 else now - self.t_last_msg)

        ctx = self.ctxb.update(
            now=now, distress=self.distress, valid=self.valid,
            face_present=self.face_present, seconds_since_msg=since,
            current_state=self.fsm.state, ack=self.ack, fault=self.fault,
            forced=self.forced)

        previous = self.fsm.state
        state = self.fsm.step(ctx)
        self.forced = None
        self.ack = False

        if state != previous:
            self.get_logger().info(
                f"{STATE_NAMES[previous]} -> {STATE_NAMES[state]} "
                f"({self.fsm.reason}, D={self.distress:.2f})")
            self._publish_command(state)

        st = BehaviourStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.behaviour_state = state
        st.behaviour_name = STATE_NAMES[state]
        st.transition_reason = self.fsm.reason
        st.time_in_state_s = float(self.fsm.time_in_state(now))
        st.distress_index = float(self.distress)
        st.alert_active = (state == ALERT)
        st.session_id = self.session_id
        self.pub_status.publish(st)

    def _publish_command(self, state: int) -> None:
        r = RESPONSES[state]
        max_blink = float(self.get_parameter("max_blink_hz").value)
        max_gain = float(self.get_parameter("max_audio_gain").value)

        cmd = MultisensoryCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.behaviour_state = state
        cmd.hue = float(r["hue"])
        cmd.saturation = float(r["sat"])
        cmd.intensity = float(r["inten"])
        # Doble comprobación de seguridad: también se verifica en el actuador
        cmd.blink_hz = float(min(r["blink"], max_blink - 0.1))
        cmd.audio_track = str(r["track"])
        cmd.audio_gain = float(min(r["gain"], max_gain))
        cmd.motion_amplitude_deg = float(r["amp"])
        cmd.motion_period_s = float(r["per"])
        self.pub_cmd.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BehaviourManagerNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
