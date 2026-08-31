#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanzamiento de la campaña de validación SIN simulador (capítulo 5).

FICHERO PYTHON (.py) de tipo *launch*. En Ubuntu:

    ros2 launch pedcom_bringup validation.launch.py \\
        scenario:=E1 dataset_dir:=/ros2_ws/datasets/E1 record:=true

Sustituye la cámara de Gazebo por el inyector de secuencias, lo que permite:

* conocer la etiqueta de referencia exacta de cada fotograma;
* repetir el experimento con entrada idéntica (verificación del RNF-06);
* ejecutar la campaña sin coste de simulación gráfica.

La grabación con `rosbag2` incluye una lista EXPLÍCITA de temas que excluye
cualquier tema de imagen: la campaña no produce ningún fichero con píxeles
(requisito RNF-02).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            RegisterEventHandler, Shutdown, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Temas grabados. NINGUNO contiene imagen (RNF-02).
TOPICS_SIN_IMAGEN = [
    "/pedcom/perception/emotional_state",
    "/pedcom/perception/face_roi",
    "/pedcom/affect/distress",
    "/pedcom/actuation/command",
    "/pedcom/status/behaviour_state",
    "/pedcom/validation/ground_truth",
    "/pedcom/actuation/audio_status",
    "/pedcom/neck_controller/joint_trajectory",
]


def generate_launch_description() -> LaunchDescription:
    pkg_perception = get_package_share_directory("pedcom_perception")
    pkg_behaviour = get_package_share_directory("pedcom_behaviour")
    pkg_actuation = get_package_share_directory("pedcom_actuation")

    args = [
        DeclareLaunchArgument("scenario", default_value="E1"),
        DeclareLaunchArgument("dataset_dir", default_value=""),
        # 15 Hz, igual que `target_fps` de perception_node. Con el inyector a 30
        # Hz la percepción descartaba uno de cada dos fotogramas y la mitad de
        # las etiquetas de referencia se quedaba sin predicción emparejada.
        DeclareLaunchArgument("fps", default_value="15.0"),
        DeclareLaunchArgument("record", default_value="true"),
        DeclareLaunchArgument("bag_dir", default_value="/ros2_ws/bags"),
        # Resolución de inyección. A 640x480 cada mensaje son 921 KB, que DDS
        # fragmenta en más de una decena de datagramas UDP; con QoS BEST_EFFORT
        # la pérdida bajo carga es masiva (se procesaba el 13 % de los
        # fotogramas). Las imágenes de origen son recortes de 48x48, así que la
        # resolución alta no aportaba información: solo coste de transporte.
        # En Gazebo la cámara sigue siendo de 640x480; allí no hay este cuello
        # de botella porque el puente entrega los fotogramas en el proceso.
        DeclareLaunchArgument("width", default_value="320"),
        DeclareLaunchArgument("height", default_value="240"),
    ]

    injector = Node(
        package="pedcom_perception", executable="sequence_injector_node",
        name="sequence_injector_node", output="screen",
        parameters=[{"dataset_dir": LaunchConfiguration("dataset_dir"),
                     "fps": LaunchConfiguration("fps"),
                     "width": LaunchConfiguration("width"),
                     "height": LaunchConfiguration("height"),
                     "loop": False}],
    )
    perception = Node(
        package="pedcom_perception", executable="perception_node",
        name="perception_node", output="screen",
        parameters=[os.path.join(pkg_perception, "config", "perception.yaml"),
                    {"enabled_on_startup": True}],   # la campaña sí arranca activa
    )
    fusion = Node(
        package="pedcom_behaviour", executable="affect_fusion_node",
        name="affect_fusion_node", output="screen",
        parameters=[os.path.join(pkg_behaviour, "config", "behaviour.yaml")],
    )
    manager = Node(
        package="pedcom_behaviour", executable="behaviour_manager_node",
        name="behaviour_manager_node", output="screen",
        parameters=[os.path.join(pkg_behaviour, "config", "behaviour.yaml")],
    )
    logger = Node(
        package="pedcom_behaviour", executable="session_logger_node",
        name="session_logger_node", output="screen",
        parameters=[os.path.join(pkg_behaviour, "config", "behaviour.yaml")],
    )
    actuation = [
        Node(package="pedcom_actuation", executable=exe, name=exe, output="log",
             parameters=[os.path.join(pkg_actuation, "config",
                                      "actuation.yaml")])
        for exe in ("led_driver_node", "audio_player_node", "head_motion_node")
    ]

    activar = TimerAction(
        period=3.0,
        actions=[ExecuteProcess(
            cmd=["bash", "-c",
                 "ros2 lifecycle set /behaviour_manager_node configure && "
                 "sleep 1 && "
                 "ros2 lifecycle set /behaviour_manager_node activate && "
                 "ros2 service call /pedcom/start_session "
                 "pedcom_interfaces/srv/StartSession "
                 "'{start: true, ward: validacion}'"],
            output="screen")],
    )

    # `ros2 bag record` se niega a escribir sobre un directorio existente y muere
    # nada más arrancar. El resto del lanzamiento continúa tan campante, así que
    # la campaña se ejecuta entera y no graba nada: al analizar después se lee el
    # registro de la ejecución ANTERIOR sin que nada lo indique. Aquí el bag
    # previo se aparta con marca temporal —no se borra— antes de grabar.
    record = ExecuteProcess(
        cmd=["bash", "-c",
             ['DEST="', LaunchConfiguration("bag_dir"), "/",
              LaunchConfiguration("scenario"), '"; '
              'if [ -e "$DEST" ]; then '
              '  PREVIO="${DEST}_previo_$(date +%Y%m%dT%H%M%S)"; '
              '  mv "$DEST" "$PREVIO"; '
              '  echo "AVISO: el registro anterior se ha movido a $PREVIO"; '
              'fi; '
              'exec ros2 bag record -o "$DEST" '
              + " ".join(TOPICS_SIN_IMAGEN)]],
        condition=IfCondition(LaunchConfiguration("record")),
        output="screen",
    )

    # Cuando el inyector agota la secuencia se detiene todo el lanzamiento, de
    # modo que `ros2 bag record` cierre el fichero y escriba metadata.yaml. Sin
    # esto, la grabación seguía viva indefinidamente y el registro se llenaba de
    # reposo posterior a la secuencia (en el bag del 9-ago, 7 s de secuencia
    # frente a 180 s grabados).
    fin_de_secuencia = RegisterEventHandler(
        OnProcessExit(target_action=injector, on_exit=[Shutdown(
            reason="secuencia de validación completada")]))

    return LaunchDescription(
        args + [injector, perception, fusion, manager, logger]
        + actuation + [activar, record, fin_de_secuencia])
