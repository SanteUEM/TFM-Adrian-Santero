# -*- coding: utf-8 -*-
"""Máquina de estados del gestor de comportamiento (apartado 4.4).

FICHERO PYTHON PURO: sin ROS 2, sin tiempo real, sin efectos laterales. Esta
separación es deliberada y tiene una consecuencia práctica importante: permite
verificar el requisito RNF-06 (determinismo) con `pytest` en milisegundos, sin
levantar el sistema.

La lógica está escrita como guardas puras evaluadas sobre un contexto inmutable.
Dada la misma secuencia de contextos, la secuencia de estados es siempre la
misma.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# Códigos de estado (coinciden con MultisensoryCommand.msg)
IDLE, ENGAGE, COMFORT, CALM_DOWN, ALERT, SAFE_STOP = range(6)

STATE_NAMES: Dict[int, str] = {
    IDLE: "IDLE", ENGAGE: "ENGAGE", COMFORT: "COMFORT",
    CALM_DOWN: "CALM_DOWN", ALERT: "ALERT", SAFE_STOP: "SAFE_STOP",
}


@dataclass(frozen=True)
class Thresholds:
    """Umbrales con histéresis: la entrada y la salida no usan el mismo valor."""

    engage: float = 0.30        # ENGAGE  -> COMFORT
    comfort_exit: float = 0.25  # COMFORT -> ENGAGE
    calm: float = 0.60          # COMFORT -> CALM_DOWN
    calm_exit: float = 0.50     # CALM_DOWN -> COMFORT
    alert: float = 0.85         # CALM_DOWN -> ALERT


@dataclass(frozen=True)
class Timers:
    """Temporizadores de permanencia, en segundos."""

    face_to_engage: float = 1.0
    no_face_to_idle: float = 5.0
    dwell_calm: float = 3.0
    dwell_alert: float = 10.0
    dwell_exit: float = 5.0
    min_dwell: float = 2.0       # limitación de tasa entre transiciones
    watchdog: float = 1.0        # sin datos -> SAFE_STOP


@dataclass(frozen=True)
class Context:
    """Fotografía de las entradas de la máquina en un instante dado."""

    now: float                    # tiempo monótono en segundos
    distress: float               # D(t)
    valid: bool                   # hay observación reciente
    face_present: bool
    seconds_since_msg: float      # antigüedad del último mensaje recibido
    face_for: float = 0.0         # tiempo con rostro continuo
    no_face_for: float = 0.0      # tiempo sin rostro
    above_calm_for: float = 0.0   # tiempo con D >= umbral de calma
    above_alert_for: float = 0.0
    below_exit_for: float = 0.0   # tiempo con D por debajo del umbral de salida
    ack_by_staff: bool = False
    fault: bool = False
    forced_state: Optional[int] = None


@dataclass
class Transition:
    target: int
    guard: Callable[[Context], bool]
    reason: str


class BehaviourFSM:
    """Máquina de estados con histéresis, temporizadores y fallo seguro."""

    def __init__(self,
                 thresholds: Optional[Thresholds] = None,
                 timers: Optional[Timers] = None,
                 t0: float = 0.0) -> None:
        self.th = thresholds or Thresholds()
        self.tm = timers or Timers()
        self.state: int = IDLE
        self.reason: str = "inicio"
        self.t_entered: float = t0
        self.t_last_transition: float = t0
        self.history: List[Tuple[float, int, str]] = [(t0, IDLE, "inicio")]
        self._table = self._build_table()

    # ------------------------------------------------------------------ tabla
    def _build_table(self) -> Dict[int, List[Transition]]:
        th, tm = self.th, self.tm
        return {
            IDLE: [
                Transition(ENGAGE,
                           lambda c: c.face_present and c.face_for >= tm.face_to_engage,
                           "rostro detectado"),
            ],
            ENGAGE: [
                Transition(COMFORT,
                           lambda c: c.valid and c.distress >= th.engage,
                           "D >= umbral de acompanamiento"),
                Transition(IDLE,
                           lambda c: c.no_face_for >= tm.no_face_to_idle,
                           "ausencia de rostro"),
            ],
            COMFORT: [
                Transition(CALM_DOWN,
                           lambda c: (c.valid and c.distress >= th.calm
                                      and c.above_calm_for >= tm.dwell_calm),
                           "D >= umbral de calma sostenido"),
                Transition(ENGAGE,
                           lambda c: (c.distress < th.comfort_exit
                                      and c.below_exit_for >= tm.dwell_exit),
                           "D por debajo del umbral de salida"),
                Transition(IDLE,
                           lambda c: c.no_face_for >= tm.no_face_to_idle,
                           "ausencia de rostro"),
            ],
            CALM_DOWN: [
                Transition(ALERT,
                           lambda c: (c.valid and c.distress >= th.alert
                                      and c.above_alert_for >= tm.dwell_alert),
                           "malestar muy elevado y sostenido"),
                Transition(COMFORT,
                           lambda c: (c.distress < th.calm_exit
                                      and c.below_exit_for >= tm.dwell_exit),
                           "D por debajo del umbral de salida"),
            ],
            ALERT: [
                Transition(CALM_DOWN,
                           lambda c: c.ack_by_staff,
                           "confirmacion del personal"),
            ],
            SAFE_STOP: [],       # solo se sale por rearme manual explícito
        }

    # ------------------------------------------------------------------ ciclo
    def step(self, ctx: Context) -> int:
        """Evalúa un ciclo de la máquina y devuelve el estado resultante."""
        # 1. Prioridad absoluta: fallo seguro (vigilante o error declarado)
        if ctx.fault or ctx.seconds_since_msg > self.tm.watchdog:
            if self.state != SAFE_STOP:
                motivo = "fallo declarado" if ctx.fault else "vigilante vencido"
                return self._enter(SAFE_STOP, motivo, ctx.now)
            return self.state

        # 2. Orden manual del personal clínico (RF-07)
        if ctx.forced_state is not None and ctx.forced_state != self.state:
            return self._enter(ctx.forced_state, "orden del personal", ctx.now)

        # 3. Limitación de tasa: nunca más de una transición cada min_dwell
        if ctx.now - self.t_last_transition < self.tm.min_dwell:
            return self.state

        # 4. Evaluación de guardas en orden de prioridad
        for tr in self._table.get(self.state, []):
            if tr.guard(ctx):
                return self._enter(tr.target, tr.reason, ctx.now)
        return self.state

    def _enter(self, target: int, reason: str, now: float) -> int:
        self.state = target
        self.reason = reason
        self.t_entered = now
        self.t_last_transition = now
        self.history.append((now, target, reason))
        return target

    # --------------------------------------------------------------- utilidad
    def rearm(self, now: float) -> int:
        """Rearme manual desde SAFE_STOP (requiere acción humana)."""
        if self.state == SAFE_STOP:
            return self._enter(IDLE, "rearme manual", now)
        return self.state

    def time_in_state(self, now: float) -> float:
        return max(0.0, now - self.t_entered)

    @property
    def name(self) -> str:
        return STATE_NAMES[self.state]

    def spurious_transitions(self, window_s: float = 3.0) -> int:
        """Transiciones que revierten en menos de `window_s` (métrica del bloque B)."""
        n = 0
        for i in range(2, len(self.history)):
            t0, s0, _ = self.history[i - 2]
            _, s1, _ = self.history[i - 1]
            t2, s2, _ = self.history[i]
            if s0 == s2 and s1 != s0 and (t2 - t0) < window_s:
                n += 1
        return n


class ContextBuilder:
    """Acumula los temporizadores necesarios para construir un `Context`.

    Se mantiene fuera de la FSM para que esta siga siendo puramente funcional.
    """

    def __init__(self, thresholds: Optional[Thresholds] = None) -> None:
        self.th = thresholds or Thresholds()
        self.face_for = 0.0
        self.no_face_for = 0.0
        self.above_calm_for = 0.0
        self.above_alert_for = 0.0
        self.below_exit_for = 0.0
        self._t_prev: Optional[float] = None

    def update(self, now: float, distress: float, valid: bool,
               face_present: bool, seconds_since_msg: float,
               current_state: int, ack: bool = False,
               fault: bool = False, forced: Optional[int] = None) -> Context:
        dt = 0.0 if self._t_prev is None else max(0.0, now - self._t_prev)
        self._t_prev = now

        if face_present:
            self.face_for += dt
            self.no_face_for = 0.0
        else:
            self.no_face_for += dt
            self.face_for = 0.0

        self.above_calm_for = (self.above_calm_for + dt
                               if distress >= self.th.calm else 0.0)
        self.above_alert_for = (self.above_alert_for + dt
                                if distress >= self.th.alert else 0.0)

        # El umbral de salida depende del estado actual (histéresis asimétrica)
        exit_th = (self.th.calm_exit if current_state == CALM_DOWN
                   else self.th.comfort_exit)
        self.below_exit_for = (self.below_exit_for + dt
                               if distress < exit_th else 0.0)

        return Context(
            now=now, distress=distress, valid=valid, face_present=face_present,
            seconds_since_msg=seconds_since_msg,
            face_for=self.face_for, no_face_for=self.no_face_for,
            above_calm_for=self.above_calm_for,
            above_alert_for=self.above_alert_for,
            below_exit_for=self.below_exit_for,
            ack_by_staff=ack, fault=fault, forced_state=forced,
        )
