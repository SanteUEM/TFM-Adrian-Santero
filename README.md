# Pediatric-Companion — código ROS 2

Implementación del sistema descrito en el TFM *«Pediatric-Companion: desarrollo
de un sistema de interacción basado en IA para el acompañamiento emocional y la
monitorización de niños en entornos hospitalarios»*.

**Autor:** Adrián Santero Alonso · **Directora:** Noelia Fernández Talavera
**Máster Universitario en Robótica y Automatización — Universidad Europea de Madrid**

---

## 0. Cómo leer esta guía

En el TFM conviven dos tipos de "código" y conviene no mezclarlos. En esta guía
se distinguen siempre así:

| Marca | Significado | Dónde se ejecuta |
|---|---|---|
| 🖥️ **Ubuntu** | Comandos de terminal (bash). Se **escriben y se ejecutan**, no se guardan como parte del sistema | Terminal de Ubuntu, dentro del contenedor Docker |
| 🐍 **Python (.py)** | Ficheros fuente del sistema. Se **editan** en VS Code y los ejecuta ROS 2 | Se cargan como nodos ROS 2 |
| 📄 **Configuración** | YAML, XML, XACRO, SDF. Describen parámetros, el robot y el mundo | Los leen los nodos y Gazebo |

Regla práctica: si aparece `ros2 run`, `colcon`, `apt`, `docker` o `source`, es
🖥️ **Ubuntu**. Si aparece `class ... (Node)`, es 🐍 **Python**.

---

## 1. Estructura del workspace

```
pediatric_companion_ws/
├── docker/                     📄 entorno reproducible
│   ├── Dockerfile              📄 imagen ROS 2 Jazzy + Gazebo + MediaPipe
│   └── run.sh                  🖥️ arranque del contenedor
├── .devcontainer/              📄 integración con VS Code (Dev Containers)
├── .vscode/                    📄 tareas, depuración y rutas de Python
├── scripts/
│   ├── train_affect_mlp.py     🐍 entrenamiento del clasificador afectivo
│   ├── eval_backends.py        🐍 comparativa DeepFace vs MediaPipe (6.1)
│   └── analyze_campaign.py     🐍 análisis de la campaña (capítulo 6)
├── src/
│   ├── pedcom_interfaces/      📄 mensajes, servicios y acción (C++/CMake)
│   ├── pedcom_perception/      🐍 percepción facial y afectiva
│   ├── pedcom_behaviour/       🐍 fusión temporal, FSM y gobernanza
│   ├── pedcom_actuation/       🐍 iluminación, audio y movimiento
│   └── pedcom_bringup/         📄 URDF, mundo Gazebo y lanzamiento
├── datasets/                   ⛔ no versionado (ver datasets/README.md)
├── logs/                       resultados agregados de la campaña
├── resultados/                 salidas de eval_backends.py (apartado 6.1)
└── bags/                       ⛔ no versionado (grabaciones rosbag2)
```

Qué **no** viaja en el repositorio y por qué: las imágenes de `datasets/` son un
conjunto público de terceros y se reconstruyen siguiendo `datasets/README.md`;
`build/`, `install/` y `log/` son artefactos de `colcon`; `bags/` y las sesiones
crudas de `logs/sessions/` se regeneran ejecutando la campaña. Sí viajan el
clasificador entrenado (`mlp_affect.npz`, 24 KB) y los manifiestos de escenario
`datasets/E*.txt`, de modo que el sistema arranca tras un `git clone` y la
campaña es trazable.

Correspondencia con la memoria:

| Paquete | Apartado de la memoria | Objetivo |
|---|---|---|
| `pedcom_perception` | 5.3 | OE3 |
| `pedcom_behaviour` | 5.4, 5.5, 5.7 | OE4, OE5 |
| `pedcom_actuation` | 5.5.4 | OE4 |
| `pedcom_interfaces` | 5.6 | OE2 |
| `pedcom_bringup` | 5.1, 5.2 | OE2, OE6 |

> Los objetivos específicos son los del capítulo 2 de la memoria: OE1 estado del
> arte, OE2 arquitectura, OE3 percepción emocional, OE4 decisión y reacción,
> OE5 análisis GDPR, OE6 evaluación del sistema integrado.

---

## 2. Puesta en marcha

### 2.0 Obtener el código — 🖥️ Ubuntu

```bash
git clone https://github.com/SANTEUEM/TFM-Adrian-Santero.git
cd TFM-Adrian-Santero
```

Antes de la campaña de validación hay que colocar el conjunto de datos: no se
distribuye con el repositorio. Las instrucciones están en
[`datasets/README.md`](datasets/README.md). Sin él, el sistema arranca y la
simulación funciona; lo que no se puede reproducir es la evaluación del
capítulo 6.

### 2.1 Construir la imagen — 🖥️ Ubuntu

Si ya tienes tu imagen `ros2-jazzy`, puedes usarla, pero le faltarán Gazebo,
MediaPipe y `ros2_control`. Lo recomendable es construir la imagen del proyecto:

```bash
cd pediatric_companion_ws
docker build -t ros2-jazzy-pedcom -f docker/Dockerfile .
```

### 2.2 Arrancar el contenedor — 🖥️ Ubuntu

```bash
bash docker/run.sh
```

El script monta `src/`, `logs/`, `bags/` y `datasets/` dentro de `/ros2_ws`, y
comparte el servidor gráfico para que Gazebo y RViz puedan abrir ventanas.

Si prefieres tu propio `docker run`, el equivalente mínimo es:

```bash
xhost +local:docker
docker run -it --rm --name pedcom --hostname pedcom \
  --network=bridge --workdir=/ros2_ws \
  --env DISPLAY=$DISPLAY --env ROS_DOMAIN_ID=42 --env ROS_LOCALHOST_ONLY=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "$PWD/src:/ros2_ws/src:rw" \
  --volume "$PWD/logs:/ros2_ws/logs:rw" \
  --volume "$PWD/bags:/ros2_ws/bags:rw" \
  ros2-jazzy-pedcom /bin/bash
```

> `ROS_LOCALHOST_ONLY=1` no es cosmético: confina el tráfico DDS a la máquina
> local y es uno de los mecanismos del requisito **RNF-03** (apartado 4.7).

### 2.3 Compilar — 🖥️ Ubuntu (dentro del contenedor)

```bash
source /opt/ros/jazzy/setup.bash
cd /ros2_ws
rosdep install --from-paths src --ignore-src -r -y     # dependencias
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` enlaza los ficheros Python en lugar de copiarlos: al editar
un `.py` no hace falta recompilar, solo reiniciar el nodo.

### 2.4 Descargar el modelo de MediaPipe — 🖥️ Ubuntu

```bash
cd /ros2_ws/src/pedcom_perception/models
wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Si no lo descargas, el sistema arranca igualmente en **modo de reserva** (OpenCV
+ clasificador de reglas) y lo indica en el registro. Es útil para depurar la
lógica de comportamiento sin depender de la visión.

### 2.5 Instalar DeepFace (opcional, línea base comparativa) — 🖥️ Ubuntu

```bash
pip install deepface tf-keras
```

DeepFace es la **línea base** con la que se compara la cadena de percepción del
sistema (objetivo OE3, apartado 6.1 de la memoria). Es una dependencia
*opcional*: si no está instalada, `create_backend("deepface")` degrada a
MediaPipe y lo indica en el registro, de modo que el sistema arranca igualmente.

Para seleccionar el backend, edita `backend` en
`src/pedcom_perception/config/perception.yaml` (`mediapipe`, `deepface` o
`auto`) o pásalo por línea de órdenes:

```bash
ros2 run pedcom_perception perception_node --ros-args -p backend:=deepface
```

### 2.6 Comparar los dos backends — 🖥️ Ubuntu

```bash
python3 scripts/eval_backends.py \
  --dataset datasets/test \
  --backends mediapipe deepface \
  --landmarker src/pedcom_perception/models/face_landmarker.task \
  --weights   src/pedcom_perception/models/mlp_affect.npz \
  --out resultados/
```

El conjunto de prueba debe tener una subcarpeta por clase (`neutral`, `joy`,
`sadness`, `anger`, `fear`, `surprise`, `pain`). El script genera
`comparativa_backends.csv`, una matriz de confusión por backend y un
`report.md`, que son exactamente las tablas del apartado 6.1 de la memoria.

> ⚠️ **Aviso ético:** no ejecutes este script sobre imágenes de pacientes
> reales. Solo conjuntos públicos de acceso académico o imágenes propias
> (apartados 2.3 y 4.3.1 de la memoria).

---

## 3. Ejecución

### 3.1 Simulación completa — 🖥️ Ubuntu

```bash
source /ros2_ws/install/setup.bash
ros2 launch pedcom_bringup pedcom_sim.launch.py rviz:=true
```

Esto levanta Gazebo con la habitación pediátrica, genera el robot, arranca el
puente y todos los nodos del sistema.

**El sistema arranca desactivado a propósito.** En otra terminal:

```bash
# 1. Configurar y activar el gestor de comportamiento (ciclo de vida)
ros2 lifecycle set /behaviour_manager_node configure
ros2 lifecycle set /behaviour_manager_node activate

# 2. Habilitar la percepción (equivale al consentimiento del personal)
ros2 service call /pedcom/perception/enable std_srvs/srv/SetBool "{data: true}"

# 3. Abrir sesión (genera un identificador aleatorio, sin vínculo clínico)
ros2 service call /pedcom/start_session pedcom_interfaces/srv/StartSession \
  "{start: true, ward: 'pediatria'}"
```

Para saltarse los tres pasos durante el desarrollo: `autostart:=true`.

### 3.2 Observar el sistema en marcha — 🖥️ Ubuntu

```bash
ros2 topic echo /pedcom/affect/distress          # índice de malestar D(t)
ros2 topic echo /pedcom/status/behaviour_state   # estado de la FSM
ros2 topic hz   /pedcom/perception/emotional_state
ros2 node list
rqt_graph                                        # grafo de nodos (figura 10)
```

### 3.3 Control manual por el personal clínico — 🖥️ Ubuntu

```bash
# Forzar la parada segura
ros2 service call /pedcom/set_behaviour pedcom_interfaces/srv/SetBehaviour \
  "{requested_state: 5, force: true, disable_perception: true, operator_id: 'enf01'}"

# Desactivar por completo el tratamiento de datos
ros2 lifecycle set /behaviour_manager_node deactivate
```

### 3.4 Campaña de validación — 🖥️ Ubuntu

```bash
ros2 launch pedcom_bringup validation.launch.py \
    scenario:=E1 dataset_dir:=/ros2_ws/datasets/E1 record:=true

# Al terminar, análisis de métricas
python3 /ros2_ws/scripts/analyze_campaign.py \
    --bag /ros2_ws/bags/E1 --scenario E1 --out /ros2_ws/logs/resultados_E1
```

Si `dataset_dir` está vacío, el inyector genera una secuencia sintética: sirve
para verificar la cadena sin manejar imágenes de menores.

---

## 4. Los nodos, uno a uno

| Nodo | Fichero 🐍 | Entradas | Salidas |
|---|---|---|---|
| `perception_node` | `pedcom_perception/perception_node.py` | `/pedcom/camera/image_raw` | `emotional_state`, `face_roi` |
| `sequence_injector_node` | `pedcom_perception/sequence_injector_node.py` | conjunto de datos | `image_raw`, `ground_truth` |
| `affect_fusion_node` | `pedcom_behaviour/affect_fusion_node.py` | `emotional_state` | `affect/distress` |
| `behaviour_manager_node` | `pedcom_behaviour/behaviour_manager_node.py` | `distress`, `face_roi` | `actuation/command`, `behaviour_state` |
| `privacy_guard_node` | `pedcom_behaviour/privacy_guard_node.py` | grafo ROS 2 | auditoría, parada segura |
| `session_logger_node` | `pedcom_behaviour/session_logger_node.py` | `behaviour_state` | CSV seudonimizado |
| `led_driver_node` | `pedcom_actuation/led_driver_node.py` | `actuation/command` | `led_color`, marcador RViz |
| `audio_player_node` | `pedcom_actuation/audio_player_node.py` | `actuation/command` | `audio_status` |
| `head_motion_node` | `pedcom_actuation/head_motion_node.py` | `command`, `face_roi` | trayectoria del cuello |

Ficheros de lógica pura (sin ROS 2, y por eso comprobables con `pytest`):

- `pedcom_perception/affect_backend.py` — cadena de visión y clasificador.
- `pedcom_behaviour/fsm.py` — máquina de estados con histéresis.

---

## 5. Decisiones de implementación que conviene poder defender

### 5.1 Por qué la percepción es un solo nodo y no tres

La memoria describe tres componentes en un contenedor con comunicación
intraproceso. En **C++** eso se consigue con `rclcpp_components` y transferencia
por puntero. En **Python no existe** comunicación intraproceso con copia cero:
publicar la imagen recortada entre nodos la convertiría en tráfico DDS
observable por cualquier suscriptor, que es justo lo que prohíbe el requisito
**RNF-02**.

La solución adoptada mantiene intacta la garantía: los tres pasos se ejecutan
dentro del proceso de `perception_node` y hacia el exterior solo salen la
geometría del rostro y siete probabilidades. Si en el futuro se porta la
percepción a C++, la interfaz publicada no cambia.

### 5.2 Por qué el gestor es un `LifecycleNode`

En estado `unconfigured` **no existen suscripciones** a los temas de percepción.
«Sistema desactivado» pasa a ser un estado real y verificable del grafo en lugar
de una convención. Es la materialización técnica del requisito **RNF-07** (el
sistema arranca desactivado) exigido por el RGPD.

Comprobación — 🖥️ Ubuntu:

```bash
ros2 lifecycle get /behaviour_manager_node
ros2 topic info /pedcom/affect/distress --verbose   # 0 suscriptores antes de activar
```

### 5.3 Por qué los límites de seguridad se comprueban dos veces

El parpadeo (< 3 Hz, **RNF-04**) y la ganancia de audio (**RNF-05**) se limitan
en el gestor de comportamiento *y* otra vez en cada actuador. Un actuador que
confía en su emisor no es un actuador seguro: si un fallo lógico pidiera un
parpadeo de 10 Hz, `led_driver_node` lo rechaza y lo registra como error.

### 5.4 Por qué la FSM está separada de ROS 2

`fsm.py` es Python puro y determinista. Eso permite verificar el requisito
**RNF-06** con diez repeticiones en milisegundos, sin levantar el sistema, y es
lo que hace defendible la afirmación de que la lógica es auditable.

---

## 6. Pruebas

### 6.1 Pruebas puras, sin ROS 2 — 🖥️ Ubuntu

```bash
cd /ros2_ws
python3 -m pytest src/pedcom_behaviour/test src/pedcom_perception/test -v
```

Verifican los escenarios E6–E9, la ausencia de transiciones espurias (E7), el
determinismo (RNF-06) y las propiedades del índice de malestar.

### 6.2 Suite completa, con integración — 🖥️ Ubuntu

```bash
cd /ros2_ws
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

La prueba de integración `test_pipeline_integration.py` levanta la cadena
completa y comprueba, entre otras cosas, que **no existe ningún tema de imagen
con publicadores ajenos al inyector** (RNF-02) y que el gestor no publica nada
antes de ser activado (RNF-07).

---

## 7. Uso desde Visual Studio Code

1. Instala las extensiones **Dev Containers**, **Python** y **ROS**.
2. Abre la carpeta `pediatric_companion_ws`.
3. `F1` → *Dev Containers: Reopen in Container*. VS Code construye la imagen del
   `Dockerfile` y abre el editor dentro del contenedor.
4. Tareas disponibles con `Ctrl+Shift+B` y `F1` → *Tasks: Run Task*:
   - `colcon: build`
   - `colcon: test`
   - `pytest: pruebas puras (sin ROS)`
   - `ros2 launch: simulacion`
5. `F5` depura `affect_fusion_node` o `behaviour_manager_node` con puntos de
   interrupción.

En Windows con WSL 2, abre la carpeta desde WSL (`\\wsl$\...`) y no desde
`C:\` o `D:\`: el rendimiento del sistema de ficheros es órdenes de magnitud
mejor y los enlaces simbólicos de `--symlink-install` funcionan.

---

## 8. Diagnóstico de problemas frecuentes

| Síntoma | Causa probable | Solución 🖥️ Ubuntu |
|---|---|---|
| `Package 'pedcom_interfaces' not found` | no se ha recargado el entorno | `source /ros2_ws/install/setup.bash` |
| `perception_node` dice `backend=fallback` | falta `face_landmarker.task` | descargarlo (apartado 2.4) |
| El robot no se mueve | no se cargaron los controladores | `ros2 control list_controllers` |
| No hay imagen en RViz | política de fiabilidad incorrecta | poner *Best Effort* en el display de imagen |
| `distress` siempre vale 0 | percepción desactivada | `ros2 service call /pedcom/perception/enable std_srvs/srv/SetBool "{data: true}"` |
| El gestor no reacciona | está en `unconfigured` | `ros2 lifecycle set /behaviour_manager_node activate` |
| Gazebo no abre ventana | falta permiso del servidor X | `xhost +local:docker` en el anfitrión |
| Los nodos no se ven entre sí | dominios distintos | mismo `ROS_DOMAIN_ID` en todas las terminales |

---

## 9. Nota sobre datos personales

Este código **no debe ejecutarse con vídeo real de menores** fuera de un marco
con consentimiento informado, evaluación de impacto (RGPD art. 35) y aprobación
del comité de ética correspondiente. El desarrollo y la campaña de validación
están diseñados para funcionar con secuencias sintéticas o con conjuntos de
datos de acceso abierto ya publicados con fines de investigación.

---

## 10. Licencia, cita y alcance

**Licencia:** Apache-2.0. Texto completo en [`LICENSE`](LICENSE).

**Cita:** los metadatos están en [`CITATION.cff`](CITATION.cff); GitHub genera
la referencia desde el botón *Cite this repository*. El trabajo de referencia es
el Trabajo Fin de Máster de Adrián Santero Alonso, dirigido por Noelia Fernández
Talavera, Máster Universitario en Robótica y Automatización, Universidad Europea
de Madrid, septiembre de 2026.

**Componentes de terceros incluidos en `src/pedcom_perception/models/`:**

| Fichero | Origen | Licencia |
|---|---|---|
| `face_landmarker.task` | MediaPipe Face Landmarker (Google) | Apache-2.0 |
| `haarcascade_frontalface_default.xml` | OpenCV | Apache-2.0 |
| `mlp_affect.npz` | Entrenado en este trabajo | Apache-2.0 |

**Alcance:** este software es un prototipo de investigación validado en
simulación. **No es un producto sanitario**, no ha sido evaluado clínicamente y
no debe emplearse para diagnóstico ni para ninguna decisión asistencial. La
salida del clasificador es una estimación de expresión facial, no una medida del
estado interno del paciente; el sistema la utiliza únicamente para modular una
respuesta de acompañamiento. Cualquier despliegue con pacientes reales exige
consentimiento informado, evaluación de impacto conforme al artículo 35 del RGPD
y la aprobación del comité de ética asistencial correspondiente.
