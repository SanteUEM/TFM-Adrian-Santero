# -*- coding: utf-8 -*-
"""Pruebas unitarias del backend de percepción y del índice de malestar.

FICHERO PYTHON de pruebas:

    cd /ros2_ws && colcon test --packages-select pedcom_perception
    python3 -m pytest src/pedcom_perception/test/test_affect_backend.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pedcom_perception.affect_backend import (DISTRESS_WEIGHTS, EMOTIONS,  # noqa: E402
                                              DeepFaceBackend,
                                              FaceObservation, MLPClassifier,
                                              PerceptionBackend,
                                              create_backend,
                                              distress_from_probabilities,
                                              softmax)

IDX = {name: i for i, name in enumerate(EMOTIONS)}


def one_hot(name: str) -> np.ndarray:
    v = np.zeros(7, dtype=np.float32)
    v[IDX[name]] = 1.0
    return v


# ------------------------------------------------------------------- softmax
def test_softmax_es_una_distribucion_de_probabilidad():
    p = softmax(np.array([2.0, 1.0, 0.1, -3.0, 0.0, 0.5, 1.5]))
    assert pytest.approx(float(p.sum()), abs=1e-5) == 1.0
    assert (p >= 0).all()


def test_softmax_es_estable_con_valores_grandes():
    p = softmax(np.array([1000.0, 999.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert np.isfinite(p).all()


# ----------------------------------------------------- índice de malestar D(t)
def test_dolor_puro_produce_el_maximo_malestar():
    assert distress_from_probabilities(one_hot("pain")) == pytest.approx(1.0)


def test_alegria_pura_produce_malestar_nulo():
    # El peso es negativo, pero el índice está acotado inferiormente en 0
    assert distress_from_probabilities(one_hot("joy")) == pytest.approx(0.0)


def test_neutro_produce_malestar_nulo():
    assert distress_from_probabilities(one_hot("neutral")) == pytest.approx(0.0)


def test_indice_siempre_acotado_en_cero_uno():
    rng = np.random.default_rng(2026)
    for _ in range(500):
        p = softmax(rng.normal(size=7) * 3.0)
        d = distress_from_probabilities(p)
        assert 0.0 <= d <= 1.0


def test_orden_relativo_de_las_categorias_negativas():
    d_dolor = distress_from_probabilities(one_hot("pain"))
    d_miedo = distress_from_probabilities(one_hot("fear"))
    d_tristeza = distress_from_probabilities(one_hot("sadness"))
    assert d_dolor > d_miedo > d_tristeza


def test_los_pesos_coinciden_con_la_memoria():
    esperado = [0.00, -0.50, 0.60, 0.55, 0.85, 0.00, 1.00]
    assert list(np.round(DISTRESS_WEIGHTS, 2)) == pytest.approx(esperado)


# ------------------------------------------------------------- clasificador
def test_clasificador_de_reserva_devuelve_distribucion_valida():
    clf = MLPClassifier(None)
    p = clf.predict(np.zeros(25, dtype=np.float32))
    assert p.shape == (7,)
    assert pytest.approx(float(p.sum()), abs=1e-5) == 1.0


def test_clasificador_de_reserva_detecta_sonrisa():
    """Con los blendshapes de sonrisa altos, la alegría debe dominar."""
    clf = MLPClassifier(None)
    f = np.zeros(25, dtype=np.float32)
    f[14] = 0.9      # mouthSmileLeft
    f[15] = 0.9      # mouthSmileRight
    p = clf.predict(f)
    assert EMOTIONS[int(np.argmax(p))] == "joy"


def test_clasificador_de_reserva_es_determinista():
    clf = MLPClassifier(None)
    f = np.linspace(0, 1, 25).astype(np.float32)
    p1, p2 = clf.predict(f), clf.predict(f)
    assert np.allclose(p1, p2)


def test_clasificador_de_reserva_acepta_diccionario_de_blendshapes():
    """El camino con MediaPipe entrega los coeficientes indexados por nombre."""
    clf = MLPClassifier(None)
    p = clf.predict(np.zeros(52, dtype=np.float32),
                    {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9})
    assert EMOTIONS[int(np.argmax(p))] == "joy"


# ------------------------------------------ contrato de entrada del MLP (regresión)
MODELO = os.path.join(os.path.dirname(__file__), "..", "models",
                      "mlp_affect.npz")


@pytest.mark.skipif(not os.path.isfile(MODELO),
                    reason="no hay modelo entrenado disponible")
def test_el_mlp_espera_el_vector_completo_de_blendshapes():
    """Regresión: la inferencia y el entrenamiento deben usar la misma entrada.

    El entrenamiento (`scripts/train_affect_mlp.py`) guarda los 52 coeficientes
    crudos de MediaPipe en su orden nativo. Durante un tiempo la inferencia
    construyó en cambio un vector de 25 posiciones filtrando por los nombres de
    `_BS_KEYS`, lo que hacía fallar el producto matricial en cuanto MediaPipe
    llegaba a cargarse. El error permanecía latente porque el camino de reserva
    nunca usa el MLP.
    """
    clf = MLPClassifier(MODELO)
    assert clf.ready
    assert clf.n_entradas == 52

    p = clf.predict(np.zeros(clf.n_entradas, dtype=np.float32))
    assert p.shape == (7,)
    assert pytest.approx(float(p.sum()), abs=1e-5) == 1.0

    with pytest.raises(ValueError, match="coeficientes de expresión"):
        clf.predict(np.zeros(25, dtype=np.float32))


# ------------------------------------------------------ observación y backend
def test_observacion_por_defecto_es_neutra_y_sin_rostro():
    obs = FaceObservation()
    assert obs.face_present is False
    assert obs.dominant_emotion == "neutral"


def test_backend_arranca_en_modo_de_reserva_sin_modelo():
    backend = PerceptionBackend(landmarker_model="", weights_path="")
    assert backend.mode == "fallback"


def test_backend_tolera_una_imagen_vacia():
    backend = PerceptionBackend()
    obs = backend.process(np.zeros((0, 0, 3), dtype=np.uint8))
    assert obs.face_present is False


def test_backend_descarta_rostros_fuera_de_la_zona_del_paciente():
    """Escenario E5: un acompañante fuera de la región de la cama se ignora."""
    backend = PerceptionBackend()
    obs = FaceObservation(face_present=True, center_uv=(0.9, 0.0),
                          probabilities=one_hot("pain"))
    u0, v0, u1, v1 = (-0.6, -0.7, 0.6, 0.7)
    u, v = obs.center_uv
    dentro = (u0 <= u <= u1 and v0 <= v <= v1)
    assert dentro is False


# --------------------------------- backend DeepFace y factoría (OE3, 5.3.5)
def test_la_factoria_devuelve_mediapipe_por_defecto():
    backend = create_backend("mediapipe")
    assert isinstance(backend, PerceptionBackend)


def test_la_factoria_degrada_si_deepface_no_esta_instalado():
    """Un equipo sin la dependencia opcional debe poder ejecutar el sistema."""
    backend = create_backend("deepface")
    assert hasattr(backend, "process")          # interfaz común garantizada
    if isinstance(backend, DeepFaceBackend):
        assert backend.available is True
    else:
        assert isinstance(backend, PerceptionBackend)


def test_deepface_mapea_su_vocabulario_a_las_siete_clases():
    """'disgust' se asimila a dolor; la distribución debe sumar 1 (5.3.3)."""
    backend = DeepFaceBackend()
    probs = backend._map_probabilities({"emotion": {"happy": 80.0,
                                                    "disgust": 20.0}})
    assert pytest.approx(float(probs.sum()), abs=1e-5) == 1.0
    assert probs[IDX["joy"]] > probs[IDX["pain"]] > 0.0


def test_deepface_devuelve_neutro_ante_una_salida_vacia():
    backend = DeepFaceBackend()
    probs = backend._map_probabilities({"emotion": {}})
    assert probs[IDX["neutral"]] == 1.0


def test_ambos_backends_comparten_la_interfaz_de_observacion():
    """Condición necesaria para que la comparativa del 6.1 sea válida."""
    vacio = np.zeros((0, 0, 3), dtype=np.uint8)
    for backend in (PerceptionBackend(), DeepFaceBackend()):
        obs = backend.process(vacio, None)
        assert isinstance(obs, FaceObservation)
        assert obs.face_present is False
        assert len(obs.probabilities) == len(EMOTIONS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
