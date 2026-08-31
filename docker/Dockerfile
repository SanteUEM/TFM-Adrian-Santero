# ============================================================================
# Imagen de desarrollo de Pediatric-Companion
# FICHERO Dockerfile. Se construye en Ubuntu (o en Windows con Docker Desktop):
#     docker build -t ros2-jazzy-pedcom -f docker/Dockerfile .
# ============================================================================
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# --- Dependencias del sistema ------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-venv python3-numpy python3-opencv \
      opencv-data \
      python3-colcon-common-extensions python3-rosdep \
      ros-jazzy-rviz2 \
      ros-jazzy-xacro \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-joint-state-publisher \
      ros-jazzy-ros-gz-sim \
      ros-jazzy-ros-gz-bridge \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-ros2-control \
      ros-jazzy-ros2-controllers \
      ros-jazzy-controller-manager \
      ros-jazzy-joint-trajectory-controller \
      ros-jazzy-joint-state-broadcaster \
      ros-jazzy-rosbag2-storage-mcap \
      ros-jazzy-launch-testing ros-jazzy-launch-testing-ros \
      ros-jazzy-launch-testing-ament-cmake \
      pulseaudio-utils alsa-utils wget git nano \
    && rm -rf /var/lib/apt/lists/*

# --- Entorno PRINCIPAL: cadena de percepción del sistema (MediaPipe) ---------
# mediapipe 0.10.14 exige protobuf<5. No instale aquí nada que arrastre
# TensorFlow: lo dejaría inservible.
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed \
      mediapipe==0.10.14 \
      "protobuf>=4.25.3,<5" \
      scikit-learn pandas matplotlib psutil

# --- Entorno AISLADO: línea base comparativa del OE3 (DeepFace) --------------
# DeepFace arrastra TensorFlow, que exige protobuf>=6.31: es incompatible con
# MediaPipe en un mismo intérprete. Se instala en un entorno virtual propio y
# la comparativa se compone con la opción --anexar de scripts/eval_backends.py.
# La comparabilidad (apartado 5.3.5) exige mismo conjunto, misma máquina y misma
# instrumentación, no el mismo proceso de Python.
# OpenCV se fija a la serie 4: en OpenCV 5 el clasificador en cascada ya no está
# expuesto como `cv2.CascadeClassifier`, y sin él DeepFace se queda sin etapa de
# detección propia, lo que invalidaría la comparación de latencias del OE3.
RUN python3 -m venv /opt/venv_deepface \
    && /opt/venv_deepface/bin/pip install --no-cache-dir \
         deepface tf-keras numpy "opencv-python-headless<5"

# --- Espacio de trabajo ------------------------------------------------------
WORKDIR /ros2_ws
RUN mkdir -p /ros2_ws/src /ros2_ws/logs /ros2_ws/bags /ros2_ws/datasets

# Cargar automáticamente el entorno de ROS 2 y del propio workspace
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc && \
    echo '[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash' \
      >> /root/.bashrc && \
    echo 'export ROS_LOCALHOST_ONLY=1' >> /root/.bashrc && \
    echo 'export ROS_DOMAIN_ID=42' >> /root/.bashrc

CMD ["/bin/bash"]
