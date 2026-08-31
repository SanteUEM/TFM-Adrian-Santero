#!/usr/bin/env bash
# ============================================================================
# Arranque del contenedor de desarrollo.
# FICHERO BASH (Ubuntu). Ejecutar desde la raíz del workspace:
#     bash docker/run.sh
# ============================================================================
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-ros2-jazzy-pedcom}"
NAME="pedcom"

# Copia local (fuera de OneDrive/WSL9p) de datasets grandes, para evitar la
# lentitud de leer miles de ficheros desde /mnt/d. Opcional: si la carpeta no
# existe, el volumen queda vacío y no pasa nada.
FAST_DATASETS="${FAST_DATASETS:-$HOME/pedcom_datasets}"
mkdir -p "${FAST_DATASETS}"

# Carpetas de salida que deben sobrevivir al contenedor. El contenedor se lanza
# con --rm, de modo que cualquier ruta NO montada aquí se pierde al salir. Falto
# `resultados` durante un tiempo y el efecto fue desconcertante: la comparativa
# se ejecutaba y decía haber escrito en /ros2_ws/resultados, pero en el anfitrión
# seguían viéndose las tablas de una ejecución anterior.
mkdir -p "${WS}/resultados" "${WS}/logs" "${WS}/bags"

# Permitir que el contenedor abra ventanas (Gazebo y RViz) en el X del anfitrión
xhost +local:docker >/dev/null 2>&1 || true

# GPU: en WSL2, Docker Desktop pasa la GPU vía --gpus (no existe /dev/dri aquí,
# WSL2 usa /dev/dxg internamente y Docker Desktop lo gestiona por su cuenta).
GPU_ARGS=()
if docker info 2>/dev/null | grep -qi nvidia; then
  GPU_ARGS=(--gpus all --env NVIDIA_DRIVER_CAPABILITIES=all --env NVIDIA_VISIBLE_DEVICES=all)
fi

docker run -it --rm \
  --name "${NAME}" \
  --hostname pedcom \
  --network bridge \
  --workdir /ros2_ws \
  --env DISPLAY="${DISPLAY:-:0}" \
  --env QT_X11_NO_MITSHM=1 \
  --env ROS_DOMAIN_ID=42 \
  --env ROS_LOCALHOST_ONLY=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "${WS}/src:/ros2_ws/src:rw" \
  --volume "${WS}/logs:/ros2_ws/logs:rw" \
  --volume "${WS}/bags:/ros2_ws/bags:rw" \
  --volume "${WS}/resultados:/ros2_ws/resultados:rw" \
  --volume "${WS}/datasets:/ros2_ws/datasets:rw" \
  --volume "${WS}/scripts:/ros2_ws/scripts:rw" \
  --volume "${WS}/build:/ros2_ws/build:rw" \
  --volume "${WS}/install:/ros2_ws/install:rw" \
  --volume "${WS}/log:/ros2_ws/log:rw" \
  --volume "${FAST_DATASETS}:/ros2_ws/datasets_fast:rw" \
  "${GPU_ARGS[@]}" \
  "${IMAGE}" \
  /bin/bash
