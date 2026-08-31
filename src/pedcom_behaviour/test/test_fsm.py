# -*- coding: utf-8 -*-
"""Pruebas unitarias de la máquina de estados (bloque B del capítulo 5).

FICHERO PYTHON de pruebas. Se ejecuta en Ubuntu con:

    cd /ros2_ws && colcon test --packages-select pedcom_behaviour
    colcon test-result --verbose

o directamente, sin ROS 2:

    python3 -m pytest src/pedcom_behaviour/test/test_fsm.py -v

Estas pruebas verifican los escenarios E6 a E9 y, sobre todo, el requisito
RNF-06 (determinismo), que es el que permite defender ante el tribunal que la
lógica de comportamiento es auditable.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pedcom_behaviour.fsm import (ALERT, CALM_DOWN, COMFORT, ENGAGE, IDLE,  # noqa: E402
                                  SAFE_STOP, BehaviourFSM, ContextBuilder,
                                  Thresholds, Timers)

DT = 0.05          # 20 Hz, igual que el nodo real


def simular(perfil, timers=None, thresholds=None, faces=None):
    """Ejecuta la FSM sobre un perfil de D(t) y devuelve (fsm, estados)."""
    th = thresholds or Thresholds()
    tm = timers or Timers()
    fsm = BehaviourFSM(th, tm, t0=0.0)
    cb = ContextBuilder(th)
    estados = []
    for i, d in enumerate(perfil):
        t = i * DT
        face = True if faces is None else faces[i]
        ctx = cb.update(now=t, distress=d, valid=True, face_present=face,
                        seconds_since_msg=0.0, current_state=fsm.state)
        estados.append(fsm.step(ctx))
    return fsm, estados


def rampa(v0, v1, segundos):
    n = int(segundos / DT)
    return [v0 + (v1 - v0) * i / max(n - 1, 1) for i in range(n)]


def constante(v, segundos):
    return [v] * int(segundos / DT)


# ------------------------------------------------------------------ E6 rampas
def test_e6_rampa_ascendente_recorre_todos_los_estados():
    perfil = constante(0.0, 3) + rampa(0.0, 1.0, 40) + constante(1.0, 20)
    fsm, estados = simular(perfil)
    visitados = []
    for s in estados:
        if not visitados or visitados[-1] != s:
            visitados.append(s)
    assert visitados == [IDLE, ENGAGE, COMFORT, CALM_DOWN, ALERT]


def test_e6_rampa_descendente_regresa_ordenadamente():
    """Sin llegar a ALERT, el descenso debe recorrer CALM_DOWN -> COMFORT ->
    ENGAGE y quedarse ahí, porque el rostro sigue presente."""
    perfil = (constante(0.0, 3) + rampa(0.0, 0.75, 30) + constante(0.75, 10)
              + rampa(0.75, 0.0, 30) + constante(0.0, 20))
    fsm, estados = simular(perfil)
    visitados = []
    for s in estados:
        if not visitados or visitados[-1] != s:
            visitados.append(s)
    assert visitados == [IDLE, ENGAGE, COMFORT, CALM_DOWN, COMFORT, ENGAGE]
    assert estados[-1] == ENGAGE


def test_alert_solo_se_abandona_con_confirmacion_del_personal():
    """Requisito de diseño: aunque el malestar desaparezca, el aviso a
    enfermería permanece activo hasta que alguien lo confirma."""
    perfil = (constante(0.0, 3) + rampa(0.0, 1.0, 40) + constante(1.0, 20)
              + constante(0.0, 40))
    fsm, estados = simular(perfil)
    assert estados[-1] == ALERT

    # Con la confirmación del personal, la máquina vuelve al acompañamiento
    cb = ContextBuilder()
    t = len(perfil) * DT + 1.0
    ctx = cb.update(now=t, distress=0.0, valid=True, face_present=True,
                    seconds_since_msg=0.0, current_state=fsm.state, ack=True)
    assert fsm.step(ctx) == CALM_DOWN


# -------------------------------------------------------------- E7 histéresis
def test_e7_oscilacion_en_umbral_no_produce_transiciones_espurias():
    """El requisito clave: oscilar en torno a 0,60 no debe hacer parpadear el
    comportamiento entre COMFORT y CALM_DOWN."""
    perfil = constante(0.0, 2) + constante(0.45, 10)
    for _ in range(30):
        perfil += constante(0.63, 0.5) + constante(0.57, 0.5)
    fsm, estados = simular(perfil)
    assert fsm.spurious_transitions(window_s=3.0) == 0


def test_histeresis_asimetrica_umbral_de_salida_menor_que_el_de_entrada():
    th = Thresholds()
    assert th.comfort_exit < th.engage
    assert th.calm_exit < th.calm


# ------------------------------------------------------ E8 pérdida de rostro
def test_e8_perdida_de_rostro_devuelve_a_idle():
    n_con = int(20 / DT)
    n_sin = int(20 / DT)
    perfil = constante(0.45, 20) + constante(0.0, 20)
    faces = [True] * n_con + [False] * n_sin
    fsm, estados = simular(perfil, faces=faces)
    assert estados[-1] == IDLE


# --------------------------------------------------------- E9 fallo inducido
def test_e9_watchdog_lleva_a_safe_stop():
    th, tm = Thresholds(), Timers()
    fsm = BehaviourFSM(th, tm, t0=0.0)
    cb = ContextBuilder(th)
    for i in range(int(10 / DT)):
        ctx = cb.update(now=i * DT, distress=0.4, valid=True, face_present=True,
                        seconds_since_msg=0.0, current_state=fsm.state)
        fsm.step(ctx)
    assert fsm.state in (ENGAGE, COMFORT)

    # El nodo de percepción deja de publicar
    ctx = cb.update(now=11.0, distress=0.4, valid=False, face_present=True,
                    seconds_since_msg=2.0, current_state=fsm.state)
    assert fsm.step(ctx) == SAFE_STOP


def test_safe_stop_solo_se_abandona_con_rearme_manual():
    fsm = BehaviourFSM(t0=0.0)
    cb = ContextBuilder()
    fsm.step(cb.update(now=1.0, distress=0.0, valid=False, face_present=False,
                       seconds_since_msg=5.0, current_state=fsm.state))
    assert fsm.state == SAFE_STOP
    for i in range(60):
        t = 2.0 + i * DT
        fsm.step(cb.update(now=t, distress=0.9, valid=True, face_present=True,
                           seconds_since_msg=0.0, current_state=fsm.state))
    assert fsm.state == SAFE_STOP
    assert fsm.rearm(now=10.0) == IDLE


# ------------------------------------------------------- RNF-06 determinismo
def test_rnf06_determinismo_diez_repeticiones_identicas():
    perfil = constante(0.0, 2) + rampa(0.0, 0.95, 30) + rampa(0.95, 0.1, 30)
    referencia = simular(perfil)[1]
    for _ in range(9):
        assert simular(perfil)[1] == referencia


# ---------------------------------------------------- limitación de la tasa
def test_limitacion_de_tasa_entre_transiciones():
    perfil = constante(0.0, 2) + rampa(0.0, 1.0, 10) + constante(1.0, 30)
    fsm, _ = simular(perfil)
    tiempos = [t for t, _, _ in fsm.history[1:]]
    for t0, t1 in zip(tiempos[:-1], tiempos[1:]):
        assert (t1 - t0) >= Timers().min_dwell - 1e-9


def test_vigilante_concede_periodo_de_gracia_al_activar():
    """Regresión: activar el gestor no debe provocar SAFE_STOP inmediato.

    El nodo arma el vigilante en `on_activate` (`t_last_msg = now`). Aquí se
    reproduce ese contrato: durante el primer periodo de vigilancia, aunque
    todavía no haya llegado ningún `DistressLevel`, la FSM debe permanecer en
    IDLE. Sin este comportamiento, el sistema quedaba bloqueado en SAFE_STOP
    a los ~50 ms de activar, según qué nodo arrancase antes.
    """
    tm = Timers()
    fsm = BehaviourFSM(t0=0.0)
    cb = ContextBuilder()

    # 0,95 s desde la activación sin un solo mensaje: aún dentro del plazo.
    t = 0.0
    while t < tm.watchdog - DT:
        t += DT
        fsm.step(cb.update(now=t, distress=0.0, valid=False, face_present=False,
                           seconds_since_msg=t, current_state=fsm.state))
    assert fsm.state == IDLE, "SAFE_STOP prematuro: el vigilante no da margen"

    # Superado el plazo sin datos, el fallo seguro sí debe dispararse.
    t += 2 * DT
    fsm.step(cb.update(now=t, distress=0.0, valid=False, face_present=False,
                       seconds_since_msg=t, current_state=fsm.state))
    assert fsm.state == SAFE_STOP, "el vigilante ha dejado de proteger"


def test_alerta_requiere_permanencia_prolongada():
    """Un pico breve por encima de 0,85 no debe disparar el aviso a enfermería."""
    perfil = (constante(0.0, 2) + constante(0.7, 15)
              + constante(0.95, 5) + constante(0.7, 10))
    fsm, estados = simular(perfil)
    assert ALERT not in estados


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
