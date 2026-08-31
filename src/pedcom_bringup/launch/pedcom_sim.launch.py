#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanzamiento completo en simulación (Gazebo Harmonic + sistema PedCom).

FICHERO PYTHON (.py) de tipo *launch*. Se ejecuta en Ubuntu con:

    ros2 launch pedcom_bringup pedcom_sim.launch.py

Argumentos disponibles:
    gui:=true|false            interfaz gráfica de Gazebo
    rviz:=true|false           visualización en RViz 2
    autostart:=true|false      configurar y activar el gestor automáticamente
    params_perception:=<ruta>  fichero de parámetros alternativo

Nota sobre el ciclo de vida: si `autostart` es false (valor por defecto y
recomendado), el gestor de comportamiento queda en `unconfigured` y NO hay
suscripciones a los temas de percepción. Es la materialización del requisito
RNF-07: el sistema arranca desactivado y requiere una acción humana explícita.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            ExecuteProcess, IncludeLaunchDescription,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_bringup = get_package_share_directory("pedcom_bringup")
    pkg_perception = get_package_share_directory("pedcom_perception")
    pkg_behaviour = get_package_share_directory("pedcom_behaviour")
    pkg_actuation = get_package_share_directory("pedcom_actuation")

    world = os.path.join(pkg_bringup, "worlds", "hospital_room.sdf")
    xacro_file = os.path.join(pkg_bringup, "urdf", "pedcom.urdf.xacro")
    bridge_cfg = os.path.join(pkg_bringup, "config", "bridge.yaml")
    rviz_cfg = os.path.join(pkg_bringup, "config", "pedcom.rviz")

    args = [
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("autostart", default_value="false"),
        DeclareLaunchArgument(
            "params_perception",
            default_value=os.path.join(pkg_perception, "config",
                                       "perception.yaml")),
        DeclareLaunchArgument(
            "params_behaviour",
            default_value=os.path.join(pkg_behaviour, "config",
                                       "behaviour.yaml")),
        DeclareLaunchArgument(
            "params_actuation",
            default_value=os.path.join(pkg_actuation, "config",
                                       "actuation.yaml")),
    ]

    # Sin esto, Gazebo no resuelve `materials/textures/patient_face.png` y el
    # plano del paciente se dibuja en blanco: la percepción no detectaría rostro
    # alguno y la máquina de estados se quedaría en IDLE toda la sesión.
    recursos = AppendEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH", os.path.join(pkg_bringup, "worlds"))

    # ------------------------------------------------------------- simulador
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])),
        launch_arguments={
            "gz_args": [
                world, " -r ",
                PythonExpression(
                    ["'' if '", LaunchConfiguration("gui"), "' == 'true' else '-s'"]),
            ],
        }.items(),
    )

    robot_description = ParameterValue(
        Command(["xacro ", xacro_file, " use_sim:=true"]), value_type=str)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description,
                     "use_sim_time": True}],
        output="screen",
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-topic", "robot_description",
                   "-name", "pedcom",
                   "-x", "0.65", "-y", "0.0", "-z", "0.90"],
        output="screen",
    )

    # Puente Gazebo <-> ROS 2. Solo se puentea lo estrictamente necesario:
    # la imagen y el reloj. Ningún otro proceso ve el flujo de vídeo.
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{"config_file": bridge_cfg, "use_sim_time": True}],
        output="screen",
    )

    # switch-timeout ampliado: con render por software (sin GPU en Docker/WSL2)
    # el bucle de simulación va lento y la activación por defecto (5 s) expira.
    controllers = [
        Node(package="controller_manager", executable="spawner",
             arguments=["joint_state_broadcaster",
                        "--controller-manager", "/controller_manager",
                        "--switch-timeout", "60",
                        "--service-call-timeout", "70"],
             output="screen"),
        Node(package="controller_manager", executable="spawner",
             arguments=["neck_controller",
                        "--controller-manager", "/controller_manager",
                        "--switch-timeout", "60",
                        "--service-call-timeout", "70"],
             output="screen"),
    ]

    # -------------------------------------------------------------- sistema
    perception = Node(
        package="pedcom_perception", executable="perception_node",
        name="perception_node", output="screen",
        parameters=[LaunchConfiguration("params_perception"),
                    {"use_sim_time": True}],
    )
    fusion = Node(
        package="pedcom_behaviour", executable="affect_fusion_node",
        name="affect_fusion_node", output="screen",
        parameters=[LaunchConfiguration("params_behaviour"),
                    {"use_sim_time": True}],
    )
    manager = Node(
        package="pedcom_behaviour", executable="behaviour_manager_node",
        name="behaviour_manager_node", output="screen",
        parameters=[LaunchConfiguration("params_behaviour"),
                    {"use_sim_time": True}],
    )
    guard = Node(
        package="pedcom_behaviour", executable="privacy_guard_node",
        name="privacy_guard_node", output="screen",
        parameters=[LaunchConfiguration("params_behaviour"),
                    {"use_sim_time": True}],
    )
    logger = Node(
        package="pedcom_behaviour", executable="session_logger_node",
        name="session_logger_node", output="screen",
        parameters=[LaunchConfiguration("params_behaviour"),
                    {"use_sim_time": True}],
    )
    actuation = [
        Node(package="pedcom_actuation", executable=exe, name=exe,
             output="screen",
             parameters=[LaunchConfiguration("params_actuation"),
                         {"use_sim_time": True}])
        for exe in ("led_driver_node", "audio_player_node", "head_motion_node")
    ]

    rviz = Node(
        package="rviz2", executable="rviz2", arguments=["-d", rviz_cfg],
        condition=IfCondition(LaunchConfiguration("rviz")),
        parameters=[{"use_sim_time": True}], output="log",
    )

    # Arranque automático del ciclo de vida (solo para desarrollo y pruebas)
    autostart = TimerAction(
        period=6.0,
        actions=[ExecuteProcess(
            cmd=["bash", "-c",
                 "ros2 lifecycle set /behaviour_manager_node configure && "
                 "sleep 1 && "
                 "ros2 lifecycle set /behaviour_manager_node activate && "
                 "ros2 service call /pedcom/perception/enable "
                 "std_srvs/srv/SetBool '{data: true}'"],
            output="screen")],
        condition=IfCondition(LaunchConfiguration("autostart")),
    )

    return LaunchDescription(
        args + [recursos, gz, robot_state_publisher, spawn, bridge] + controllers
        + [perception, fusion, manager, guard, logger] + actuation
        + [rviz, autostart]
    )
