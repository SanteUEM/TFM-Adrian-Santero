#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prueba de integración de la cadena completa (launch_testing).

FICHERO PYTHON de prueba de integración. Se ejecuta en Ubuntu con:

    cd /ros2_ws && colcon test --packages-select pedcom_bringup
    colcon test-result --verbose

o de forma aislada:

    launch_test src/pedcom_bringup/test/test_pipeline_integration.py

Levanta el sistema SIN simulador (inyector sintético + percepción + fusión +
gestor + actuación) y comprueba tres cosas:

1. que la cadena percepción -> fusión -> decisión -> actuación produce
   mensajes de extremo a extremo;
2. que el gestor de comportamiento respeta el ciclo de vida (no publica nada
   antes de ser activado): verificación práctica del requisito RNF-07;
3. que en ningún momento se publica una imagen fuera del proceso de percepción
   (requisito RNF-02).
"""

import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from pedcom_interfaces.msg import (BehaviourStatus, DistressLevel,
                                   EmotionalState, MultisensoryCommand)

QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_LAST)


@pytest.mark.launch_test
def generate_test_description():
    injector = launch_ros.actions.Node(
        package="pedcom_perception", executable="sequence_injector_node",
        name="sequence_injector_node", output="screen",
        parameters=[{"dataset_dir": "", "fps": 30.0, "loop": True}],
    )
    perception = launch_ros.actions.Node(
        package="pedcom_perception", executable="perception_node",
        name="perception_node", output="screen",
        parameters=[{"enabled_on_startup": True,
                     "landmarker_model": "",
                     "classifier_weights": "",
                     "target_fps": 15.0,
                     "patient_roi": [-1.0, -1.0, 1.0, 1.0]}],
    )
    fusion = launch_ros.actions.Node(
        package="pedcom_behaviour", executable="affect_fusion_node",
        name="affect_fusion_node", output="screen",
        parameters=[{"min_confidence": 0.3}],
    )
    manager = launch_ros.actions.Node(
        package="pedcom_behaviour", executable="behaviour_manager_node",
        name="behaviour_manager_node", output="screen",
    )
    led = launch_ros.actions.Node(
        package="pedcom_actuation", executable="led_driver_node",
        name="led_driver_node", output="log",
    )
    return launch.LaunchDescription([
        injector, perception, fusion, manager, led,
        launch_testing.actions.ReadyToTest(),
    ])


class TestCadenaCompleta(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("test_observer")

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _recoger(self, msg_type, topic, segundos, qos=QOS):
        recibidos = []
        sub = self.node.create_subscription(
            msg_type, topic, lambda m: recibidos.append(m), qos)
        fin = time.time() + segundos
        while time.time() < fin:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.node.destroy_subscription(sub)
        return recibidos

    # ---------------------------------------------------------------- pruebas
    def test_01_la_percepcion_publica_estado_afectivo(self):
        msgs = self._recoger(EmotionalState,
                             "/pedcom/perception/emotional_state", 8.0)
        self.assertGreater(len(msgs), 5,
                           "el nodo de percepción no publica estimaciones")
        for m in msgs:
            self.assertAlmostEqual(sum(m.probabilities), 1.0, places=3)

    def test_02_la_fusion_publica_indice_acotado(self):
        msgs = self._recoger(DistressLevel, "/pedcom/affect/distress", 6.0)
        self.assertGreater(len(msgs), 10)
        for m in msgs:
            self.assertGreaterEqual(m.distress_index, 0.0)
            self.assertLessEqual(m.distress_index, 1.0)

    def test_03_rnf07_el_gestor_no_publica_antes_de_activarse(self):
        """Sin `configure`+`activate` no debe existir ningún estado publicado."""
        msgs = self._recoger(BehaviourStatus,
                             "/pedcom/status/behaviour_state", 4.0)
        self.assertEqual(len(msgs), 0,
                         "el gestor publica sin haber sido activado (RNF-07)")

    def test_04_rnf02_ningun_tema_de_imagen_ajeno_al_inyector(self):
        """Solo puede existir el tema de la cámara, y con un único publicador."""
        time.sleep(2.0)
        temas = dict(self.node.get_topic_names_and_types())
        de_imagen = [t for t, tipos in temas.items()
                     if any("sensor_msgs/msg/Image" in x for x in tipos)]
        self.assertEqual(sorted(de_imagen), ["/pedcom/camera/image_raw"],
                         f"temas de imagen inesperados: {de_imagen}")
        pubs = self.node.get_publishers_info_by_topic("/pedcom/camera/image_raw")
        nombres = sorted(p.node_name for p in pubs)
        self.assertEqual(nombres, ["sequence_injector_node"])

    def test_05_la_actuacion_recibe_ordenes_tras_activar(self):
        import subprocess
        for cmd in ("configure", "activate"):
            subprocess.run(
                ["ros2", "lifecycle", "set", "/behaviour_manager_node", cmd],
                check=False, capture_output=True, timeout=20)
            time.sleep(1.0)

        estados = self._recoger(BehaviourStatus,
                                "/pedcom/status/behaviour_state", 6.0)
        self.assertGreater(len(estados), 5,
                           "el gestor no publica tras ser activado")

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        ordenes = self._recoger(MultisensoryCommand,
                                "/pedcom/actuation/command", 4.0, latched)
        for o in ordenes:
            self.assertLess(o.blink_hz, 3.0, "parpadeo por encima del límite")
            self.assertLessEqual(o.audio_gain, 0.6, "ganancia por encima del límite")


@launch_testing.post_shutdown_test()
class TestSalidaLimpia(unittest.TestCase):

    def test_los_nodos_terminan_sin_error(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
