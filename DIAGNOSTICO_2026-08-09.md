# Diagnóstico y correcciones · 9 ago 2026

Revisión del workspace tras la campaña. Resumen: **la infraestructura ROS 2
funciona (nodos, temas, ciclo de vida, grabación), pero ningún resultado
numérico obtenido hasta ahora es publicable**, porque la cadena de percepción
no ha detectado un solo rostro en ninguna de las dos vías (campaña y
comparativa de backends).

Las secciones 1 a 6 explican cada fallo y su causa. La sección 0 resume lo que
ya está corregido en el código y la sección 7 indica qué tienes que ejecutar tú.

---

## 0. Cambios ya aplicados

| Fichero | Cambio |
|---|---|
| `pedcom_perception/affect_backend.py` | Preprocesado `_preparar()`: reescalado y margen antes de detectar. La reserva Haar pasa de **0 % a 78,9 %** de detección sobre FER-2013. `minSize` 48→24 y ecualización de histograma. |
| `pedcom_perception/affect_backend.py` | La excepción de MediaPipe deja de tragarse: se guarda en `motivo_degradacion` y se emite un `RuntimeWarning`. Igual para DeepFace en `create_backend()`. |
| `scripts/eval_backends.py` | **Aborta** si el backend efectivo no es el solicitado, en vez de generar tablas de ceros. Opción `--permitir-degradacion` para depurar. |
| `pedcom_behaviour/behaviour_manager_node.py` | `t_last_msg` se arma en `on_activate`: se acaba el `SAFE_STOP` inmediato. |
| `pedcom_behaviour/test/test_fsm.py` | Prueba de regresión del periodo de gracia del vigilante. 12/12 pruebas pasan. |
| `pedcom_behaviour/session_logger_node.py` | Ignora identificadores de sesión intrusos dentro de un margen configurable y avisa una sola vez, en lugar de abrir un CSV por mensaje. |
| `pedcom_perception/sequence_injector_node.py` | Encuadre proporcional con margen; error explícito si la ruta no existe; aviso de que el patrón sintético no contiene rostro; cierre limpio al terminar la secuencia; **carga desde manifiesto `.txt`**. |
| `pedcom_bringup/launch/validation.launch.py` | `fps` por defecto 30→**15 Hz** (igual que la percepción) y `Shutdown` al agotarse la secuencia, para que el bag no siga grabando reposo. |
| `pedcom_bringup/worlds/hospital_room.sdf` | `patient_face_plane` pasa a tener **textura facial** real. |
| `pedcom_bringup/launch/pedcom_sim.launch.py` | `GZ_SIM_RESOURCE_PATH` apunta a `worlds/` para que Gazebo resuelva la textura. |
| `scripts/make_face_texture.py` | **Nuevo.** Genera la textura y comprueba que un detector la reconoce antes de escribirla. |
| `scripts/make_scenario.py` | **Nuevo.** Genera los manifiestos de escenario `E1`, `E2` y `E3` con semilla fija. |
| `datasets/E1.txt`, `E2.txt`, `E3.txt` | **Nuevos.** 1 800 fotogramas cada uno (120 s a 15 Hz). Manifiestos de 75 KB en vez de duplicar 5 400 imágenes. |
| `logs/sessions/` | Los 332 CSV contaminados se han movido a `logs/archivo_2026-08-09_previo_a_correcciones/` con un `LEEME.txt` que explica por qué no son válidos. |
| `.gitignore` | Añadidos `__pycache__/`, `.venv/`, bags y CSV de sesión. |

Lo que **no** se ha tocado: el modelo `mlp_affect.npz` y `features_train.npz`
son correctos (25 602 muestras válidas) y la ruta de MediaPipe no se ha alterado,
para no invalidar el entrenamiento ya hecho.

---

## 1. La comparativa de backends (`resultados/`) no es válida

`resultados/resultados.json` declara, para los dos backends:

```
backend_efectivo: "fallback"      no_detectados_pct: 100.0      n_muestras: 7178
```

Es decir: ni MediaPipe ni DeepFace llegaron a ejecutarse.

**Causa.** En `affect_backend.py`, `PerceptionBackend._init_mediapipe()` captura
la excepción y la descarta en silencio (`except Exception: self._landmarker = None`).
Al fallar, `create_backend()` cae al detector Haar de OpenCV, que se invoca así:

```python
faces = self._cascade.detectMultiScale(gray, 1.15, 5, minSize=(48, 48))
```

Las imágenes de FER-2013 son de **48×48 px**. Pedir una ventana mínima de 48×48
sobre una imagen de 48×48 es imposible de satisfacer: el detector devuelve 0
rostros siempre. De ahí el 100,0 % exacto.

**Comprobación empírica** (194 imágenes del conjunto de prueba):

| Preprocesado | Detecciones |
|---|---|
| tal cual, `minSize=(48,48)` (código actual) | **0 / 194** |
| reescalado a 192×192 | 111 / 194 |
| reescalado a 192×192 + margen replicado de 48 px | **151 / 194** |

**MediaPipe sí funciona con estas imágenes.** `datasets/features_train.npz`
contiene 25 602 vectores de 52 blendshapes no nulos extraídos de las 28 709
imágenes de `datasets/train` → 89 % de detección sobre las mismas imágenes de
48×48. El modelo `mlp_affect.npz` (52-64-32-7) está entrenado y es correcto.

**Conclusión.** El fallo está en el *entorno* donde se ejecutó `eval_backends.py`,
no en el algoritmo. El entrenamiento se hizo en el contenedor (MediaPipe 0.10.14,
fijado en el `Dockerfile`); la evaluación parece haberse lanzado desde el
`.venv` de Windows, que tiene **mediapipe 1.0.0** y no crea el `FaceLandmarker`.

### Acciones

1. Quitar el `except` mudo para ver el error real:
   ```python
   except Exception as exc:
       self._landmarker = None
       print(f"[affect_backend] MediaPipe no disponible: {exc!r}")
   ```
2. Ejecutar la comparativa **dentro del contenedor**, no en el `.venv`:
   ```bash
   bash docker/run.sh
   python3 scripts/eval_backends.py --dataset datasets/test --out resultados/
   ```
   La primera línea debe decir `efectivo='mediapipe'`. Si dice `fallback`, parar.
3. Instalar DeepFace en la imagen (`pip3 install deepface tf-keras`) o retirar
   esa columna de la memoria: hoy la fila «deepface» es en realidad Haar.
4. Corregir el camino de reserva aunque no se use: `minSize=(24,24)` y
   reescalado previo si `min(h,w) < 96`. Un fallback que nunca detecta nada no
   es una degradación elegante, es un fallo silencioso.
5. **No incluir `resultados/report.md` en la memoria** hasta rehacerlo.

---

## 2. La campaña E1 no ha inyectado rostros reales

`validation.launch.py` recibe `dataset_dir`, pero en `datasets/` solo existen
`train/`, `test/` y `features_train.npz`: **no hay carpeta `E1`**. El inyector
aplica su reserva:

```
Sin conjunto de datos: se genera una secuencia sintética
```

El patrón sintético es una elipse clara sobre fondo oscuro. Ni MediaPipe ni Haar
detectan eso como rostro, así que `face_present` es falso siempre → `D(t)=0` →
la FSM se queda en `IDLE` sin una sola transición. Eso explica el bloque A vacío
de `logs/resultados_E1/report.md`.

Además, en el último registro (`bags/E1`, 179,6 s):

| Tema | Mensajes | Comentario |
|---|---|---|
| `/pedcom/validation/ground_truth` | 210 | 7 clases × 30 = **7 s de inyección** |
| `/pedcom/perception/emotional_state` | 70 | percepción a 15 Hz vs inyector a 30 Hz |
| `/pedcom/perception/face_roi` | 70 | |
| `/pedcom/affect/distress` | **10 869** | ≈ 3× lo esperado a 20 Hz (véase §3) |
| `/pedcom/status/behaviour_state` | 3 354 | |
| `/pedcom/actuation/command` | 0 | nunca hubo cambio de estado |
| `/pedcom/neck_controller/joint_trajectory` | 3 580 | |

Con `loop:=false` la secuencia se agota a los 7 s, pero la grabación siguió
otros 173 s registrando reposo. El 96 % del bag es ruido.

### Acciones

1. Crear `datasets/E1/{neutral,joy,sadness,anger,fear,surprise,pain}/` con las
   imágenes del escenario (subconjunto de `datasets/test`, o secuencias propias),
   y pasar `dataset_dir:=/ros2_ws/datasets/E1`.
2. Alinear frecuencias: `fps:=15.0` en el inyector, igual que `target_fps` de
   percepción; si no, se descarta la mitad de las etiquetas de referencia.
3. Dimensionar la secuencia a la duración prevista del escenario, o `loop:=true`
   con parada explícita.
4. El inyector escala cada imagen a 640×480 con `cv2.resize`; conviene añadir
   margen (`copyMakeBorder`) para que el rostro no llegue pegado al borde.

---

## 3. Hay varias instancias del sistema vivas a la vez

Dos evidencias independientes:

* `logs/sessions/` contiene **330 ficheros**, la mayoría de 83–158 bytes. En un
  mismo segundo (`105056Z`) aparecen cuatro `session_id` distintos:
  `07e7c352`, `0a14be89`, `25f57193`, `8c0b833c`.
* `/pedcom/affect/distress` registra 10 869 mensajes en 179,6 s ≈ **60 Hz**,
  exactamente 3× la tasa configurada de 20 Hz.

Varios `behaviour_manager_node` / `affect_fusion_node` conviven en
`ROS_DOMAIN_ID=42` (contenedores o lanzamientos anteriores sin cerrar). Como
`session_logger_node` abre un fichero nuevo cada vez que cambia el `session_id`,
el trasiego entre gestores genera un CSV por mensaje.

### Acciones

1. Antes de cada campaña: `docker ps` y cerrar contenedores huérfanos;
   `pkill -f behaviour_manager_node` dentro del contenedor.
2. Verificar con `ros2 node list | sort | uniq -d` (no debe repetirse ninguno)
   y `ros2 topic hz /pedcom/affect/distress` (debe dar ≈20 Hz, no 60).
3. Blindar `session_logger_node`: fijar el `session_id` en la primera muestra e
   ignorar mensajes de otro `session_id` (o rechazarlos con un aviso), en lugar
   de reabrir fichero.
4. Archivar los 330 CSV actuales: contaminan la carpeta de resultados.

---

## 4. Condición de carrera al activar: `SAFE_STOP` inmediato

En `behaviour_manager_node.on_tick()`:

```python
since = float("inf") if self.t_last_msg is None else now - self.t_last_msg
```

El temporizador arranca a 20 Hz en `on_activate`. Si el primer `DistressLevel`
aún no ha llegado, `since = inf > watchdog (1 s)` y la FSM entra en `SAFE_STOP`
en el primer tick (~50 ms). Y `SAFE_STOP` es terminal: solo se sale por rearme
manual.

Es exactamente lo que registran las sesiones de las 10:50–12:00:

```
session_id,t_rel_ms,distress_index,behaviour_state,transition_reason,alert_active
25f57193-...,99,0.0,SAFE_STOP,vigilante vencido,0
```

La sesión de las 12:09 sí arrancó bien (`IDLE`, `inicio`) porque la fusión ya
estaba publicando. Es decir: el comportamiento depende del orden de arranque.
Un fallo intermitente así no puede quedar en un TFM sin explicar.

### Acción

Inicializar `t_last_msg` en `on_activate` (`self.t_last_msg = self._now()`), o
tratar `None` como periodo de gracia:

```python
since = 0.0 if self.t_last_msg is None else now - self.t_last_msg
```

Se conserva el vigilante (si tras activar no llega nada en 1 s, salta igual),
pero se elimina la carrera. Merece una prueba en `test_fsm.py`.

---

## 5. Qué comprobar en Gazebo

### 5.1 El «paciente» no tiene cara (bloqueante)

En `worlds/hospital_room.sdf`:

```xml
<model name="patient_face_plane">
  ...<material><ambient>0.95 0.85 0.75 1</ambient>...</material>
```

Es un plano de color piel, **sin textura**. Ningún detector facial va a
encontrar un rostro ahí. Mientras siga así, la demostración en Gazebo se
quedará en `IDLE` para siempre, por muy bien que funcione el resto.

Opciones, de menos a más trabajo:

1. Aplicar una textura de rostro al plano (`<pbr><metal><albedo_map>`) apuntando
   a un PNG en `worlds/materials/textures/`, y declarar la ruta en
   `GZ_SIM_RESOURCE_PATH`. Es lo más rápido y suficiente para la memoria.
2. Insertar un modelo de persona de Fuel (`actor` o `MaleVisitorSit`).
3. Mantener Gazebo solo para la parte cinemática y de arquitectura, y hacer toda
   la validación perceptiva con el inyector (§2). **Es la vía honesta y la más
   defendible**: se declara en el capítulo de metodología que el simulador valida
   integración y movimiento, y el inyector valida percepción.

### 5.2 Lista de verificación en el contenedor

```bash
ros2 launch pedcom_bringup pedcom_sim.launch.py gui:=true rviz:=true
```

| Qué | Comando | Esperado |
|---|---|---|
| Robot generado | ventana de Gazebo | `pedcom` sobre la mesa, a 0,65 m |
| Reloj de simulación | `ros2 topic hz /clock` | ~1000 Hz |
| Controladores | `ros2 control list_controllers` | `joint_state_broadcaster` y `neck_controller` en **active** |
| Cámara puenteada | `ros2 topic info /pedcom/camera/image_raw -v` | 0 suscriptores mientras la percepción está apagada |
| Cámara con datos | activar percepción y `ros2 topic hz /pedcom/camera/image_raw` | ~30 Hz |
| Percepción | `ros2 topic echo /pedcom/perception/face_roi` | `face_present: true` |
| Cuello | `ros2 topic echo /pedcom/neck_controller/joint_trajectory` | la cabeza se mueve en Gazebo |

Notas:

* El puente está en `lazy: true`: **es normal y deseado** que
  `/pedcom/camera/image_raw` no publique nada hasta habilitar la percepción
  (`ros2 service call /pedcom/perception/enable std_srvs/srv/SetBool "{data: true}"`).
  Esa propiedad del grafo es la prueba material del RNF-02: enséñala con
  `ros2 topic info -v` antes y después. Es un resultado, no un problema.
* Si los controladores no llegan a `active`, es el render por software de WSL2.
  El `--switch-timeout 60` ya está puesto; si aun así falla, lanza con
  `gui:=false`.
* El plugin `gz_ros2_control` apunta a `/ros2_ws/install/...` con ruta
  absoluta: solo funciona dentro del contenedor. Si alguna vez lanzas fuera,
  falla ahí.
* Para el escenario E2 (300/500/800 lux) hay que variar `<diffuse>` de
  `ceiling_1`/`ceiling_2`; hoy es fijo. Conviene parametrizarlo por argumento de
  launch para no editar el SDF entre repeticiones.

---

## 6. Qué comprobar en RViz

RViz **no interviene en la campaña**: es solo para la demostración y las figuras
de la memoria. Se lanza con `rviz:=true`.

### 6.1 Aviso importante

`config/pedcom.rviz` contiene Grid, RobotModel, TF y un Marker
(`/pedcom/actuation/led_marker`). **No añadas un display de tipo Image sobre
`/pedcom/camera/image_raw`.** `privacy_guard_node` (con `enforce: true`) detecta
el suscriptor no autorizado, desactiva la percepción y fuerza `SAFE_STOP`. Ya
ocurrió cinco veces el 26 de julio; está en `logs/privacy_audit.jsonl`:

```json
{"event": "UNAUTHORIZED_SUBSCRIBER", "topic": "/pedcom/camera/image_raw", "nodes": ["rviz"]}
```

Dicho lo cual: **eso es exactamente el mecanismo del apartado 4.7 funcionando**.
Vale la pena convertirlo en un experimento deliberado del capítulo de resultados
—«se añade un display de imagen en RViz; el sistema corta la percepción en menos
de 2 s y registra el evento»— con la traza como evidencia. Pasa de accidente a
resultado.

### 6.2 Verificación

| Qué | Esperado |
|---|---|
| Fixed Frame | `base_link` sin errores en rojo |
| RobotModel | el robot aparece (requiere `robot_state_publisher` vivo) |
| TF | `base_link → neck_yaw → neck_pitch → camera_link → camera_optical_frame` |
| Marker LED | cambia de color al cambiar de estado (ámbar IDLE, cian COMFORT, naranja CALM_DOWN) |
| Movimiento | el cuello se mueve al publicar `neck_controller` |

Si el Marker no aparece, comprueba que `led_driver_node` publica en
`/pedcom/actuation/led_marker` con el `frame_id` correcto.

---

## 7. Qué tienes que ejecutar tú

Todo dentro del contenedor. **No uses el `.venv` de Windows para nada que
involucre percepción**: es donde MediaPipe 1.0.0 falla y de donde salieron las
tablas de ceros.

### Paso 0 · Dejar limpio el dominio ROS

Es el primero por una razón: mientras haya gestores duplicados, cualquier
medición que hagas estará contaminada y no lo notarás.

```bash
docker ps                       # ¿queda algún contenedor 'pedcom' vivo?
docker rm -f pedcom             # ciérralo
```

Y ya dentro del contenedor nuevo, antes de cada campaña:

```bash
ros2 node list | sort | uniq -d          # no debe imprimir NADA
ros2 topic hz /pedcom/affect/distress    # debe dar ~20 Hz, no 60
```

### Paso 1 · Reconstruir

```bash
bash docker/run.sh
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` importa: sin él, la textura y los ficheros de configuración
se copian y tendrás que reconstruir cada vez que los cambies.

### Paso 2 · Confirmar que MediaPipe carga

```bash
python3 -c "
import sys; sys.path.insert(0,'/ros2_ws/src/pedcom_perception')
from pedcom_perception.affect_backend import create_backend
b = create_backend('mediapipe',
    landmarker_model='/ros2_ws/src/pedcom_perception/models/face_landmarker.task',
    weights_path='/ros2_ws/src/pedcom_perception/models/mlp_affect.npz')
print('modo:', b.mode, '| motivo:', b.motivo_degradacion)
print('clasificador:', 'MLP' if b.clf.ready else 'reglas')
"
```

Tiene que decir `modo: mediapipe` y `clasificador: MLP`. Si dice `fallback`, el
motivo aparece ahí mismo: resuélvelo antes de seguir. **Este es el paso que, de
haber existido, habría evitado toda la comparativa de ceros.**

### Paso 3 · Rehacer la comparativa del apartado 6.1

**Los dos backends no caben en el mismo intérprete.** `mediapipe 0.10.14` exige
`protobuf<5`; `tensorflow`, que arrastra DeepFace, exige `protobuf>=6.31`.
Instalar DeepFace en el entorno principal deja MediaPipe inservible — pip lo
avisa, pero instala igualmente.

La solución no es elegir uno: es aislar DeepFace en su propio entorno. Lo que la
comparabilidad exige (apartado 5.3.5) es mismo conjunto, misma máquina y misma
instrumentación de latencia, **no** el mismo proceso de Python.

```bash
# 1. Restaurar el entorno principal (MediaPipe)
pip3 install --break-system-packages --force-reinstall "protobuf>=4.25.3,<5"

# 2. Entorno aislado para DeepFace
python3 -m venv /opt/venv_deepface
/opt/venv_deepface/bin/pip install -q deepface tf-keras opencv-python-headless numpy

# 3. Evaluar cada uno en su entorno y componer la tabla
python3 scripts/eval_backends.py --dataset datasets/test \
        --backends mediapipe --out resultados/
/opt/venv_deepface/bin/python scripts/eval_backends.py --dataset datasets/test \
        --backends deepface --out resultados/ --anexar
```

`--anexar` fusiona con el `resultados.json` previo, de modo que
`comparativa_backends.csv` y `report.md` acaban con las dos filas aunque cada
una se haya calculado en una ejecución distinta.

La primera vez, DeepFace descarga sus pesos (unos cientos de MB) a
`~/.deepface`. Tantea con `--limite 300` antes de lanzar las 7 178 imágenes.

Esta incompatibilidad merece un párrafo en la memoria: es un ejemplo concreto de
por qué la reproducibilidad de una comparativa exige fijar el entorno, y
justifica la decisión de aislar cada cadena.

### Paso 4 · Relanzar la campaña

```bash
ros2 launch pedcom_bringup validation.launch.py \
    scenario:=E1 dataset_dir:=/ros2_ws/datasets/E1.txt fps:=15.0 record:=true
```

El lanzamiento **se cierra solo** a los 120 s, al agotarse el manifiesto. En el
arranque comprueba que el inyector dice `manifiesto cargado: 1799 fotogramas`;
si dice «SECUENCIA SINTÉTICA», la ruta está mal y no sigas.

Después:

```bash
python3 scripts/analyze_campaign.py --bag /ros2_ws/bags/E1 --out logs/resultados_E1 --scenario E1
```

El bloque A ya no debe salir vacío. Repite con `E2` y `E3`.

### Paso 5 · Comprobar la simulación

```bash
ros2 launch pedcom_bringup pedcom_sim.launch.py gui:=true rviz:=true autostart:=true
```

Sigue la lista de la §5.2. La cara del paciente ahora se ve texturizada en la
ventana de Gazebo; si aparece blanca, es que `GZ_SIM_RESOURCE_PATH` no llegó
(revisa que reconstruiste tras el cambio).

### Paso 6 · Convertir el incidente de RViz en un resultado

Con el sistema activo, añade a mano en RViz un display de Image sobre
`/pedcom/camera/image_raw` y cronometra. Debe cortarse la percepción y quedar
registrado en `logs/privacy_audit.jsonl`. Es material directo para el capítulo
de resultados (§6.1 de este documento).

---

## 7 bis. Resultados finales (11 ago 2026)

### Comparativa de percepción · apartado 6.1

| | MediaPipe + MLP | DeepFace |
|---|---|---|
| Exactitud | 0,545 | 0,578 |
| F1 macro | 0,480 | 0,546 |
| F1 dolor | 0,304 | 0,497 |
| Sensibilidad dolor | 0,413 | 0,418 |
| Precisión dolor | 0,241 | 0,613 |
| Latencia mediana | **16,8 ms** | 24,5 ms |
| No detectados | **10,7 %** | 28,2 % |
| MAE de D(t) | 0,222 | **0,187** |

Sobre el subconjunto común (4 921 imágenes, sin sesgo de detección): F1 macro
0,491 frente a 0,550. La diferencia es real, no un artefacto del muestreo.

### Campaña de validación · capítulo 6

| | E1 | E2 | E3 |
|---|---|---|---|
| Exactitud | 0,473 | 0,344 | 0,464 |
| F1 macro | 0,455 | 0,360 | 0,411 |
| Precisión dolor | 0,713 | 0,861 | 0,748 |
| Sensibilidad dolor | 0,249 | 0,242 | 0,240 |
| No detectados | 3,9 % | 4,2 % | 2,9 % |
| Transiciones | 6 | 6 | 4 |
| **Espurias/min** | **0,000** | **0,000** | **0,000** |
| Frecuencia | 15,00 Hz | 14,68 Hz | 15,00 Hz |
| Latencia percepción (mediana / p95) | 16,5 / 19,8 ms | 19,6 / 58,7 ms | 16,6 / 20,1 ms |
| Latencia e2e (mediana / p95) | 52,1 / 54,4 ms | 48,3 / 69,5 ms | 57,3 / 65,4 ms |

**RF-01 acreditado**: 15,0 Hz sostenidos con 1 799 de 1 799 fotogramas
procesados, y p95 de 19,8 ms frente a los 66,7 ms que permitiría el requisito.

### La escalada no se alcanza, y la causa no es la máquina de estados

E2 se diseñó para provocar `COMFORT → CALM_DOWN → ALERT`. Con la percepción
funcionando al 100 % **no llega**: se queda en `COMFORT`. En la ejecución
anterior, con solo el 13 % de los fotogramas, sí escalaba.

No es una regresión: es que el muestreo pobre producía excursiones espurias de
D(t) que superaban el umbral por azar. Con el flujo completo, la estimación es
estable y su valor medio durante el bloque de miedo y dolor queda por debajo de
0,60.

La causa está en la percepción, no en la decisión. Con una sensibilidad de 0,24
para la clase de dolor, la mayoría de los fotogramas de esa clase se reparten
entre otras categorías, y el vector de probabilidades promediado proyecta un
D(t) en torno a 0,3–0,4. La máquina de estados hace exactamente lo que debe: no
escalar sin evidencia sostenida.

La cobertura completa de la escalada (`IDLE → ENGAGE → COMFORT → CALM_DOWN →
ALERT`) **sí está verificada**, en `test_fsm.py`, alimentando la FSM con perfiles
de D(t) controlados (escenarios E6 a E9). Esa es la forma correcta de acreditar
la lógica de decisión: aislada de la incertidumbre del clasificador.

Estructura recomendada para el capítulo: las pruebas unitarias acreditan la
corrección de la FSM; la campaña acredita el comportamiento integrado y la
estabilidad (0 transiciones espurias en 6 minutos de operación continua); y la
sensibilidad del clasificador se identifica como el factor limitante, que es
justamente la línea de trabajo futuro del capítulo 7.

---

## 8. Lo que conviene contar en la memoria

Nada de esto es un accidente que haya que esconder: es material del capítulo de
verificación, y contado bien suma.

* **El fallo del 100 % de no detectados** es un buen ejemplo de por qué una
  degradación silenciosa es peor que un fallo ruidoso. Un `except` mudo convirtió
  un problema de instalación en unas tablas de exactitud 0,000 que parecían un
  resultado del algoritmo. La corrección —abortar en vez de degradar— es una
  decisión de diseño defendible ante el tribunal.
* **La carrera del vigilante** ilustra el compromiso entre fallo seguro y
  disponibilidad: un vigilante demasiado ansioso deja el sistema inservible. El
  periodo de gracia y su prueba de regresión son la respuesta.
* **El corte por RViz** demuestra que el mecanismo del RNF-02 no es declarativo.
* **El manifiesto de escenario con semilla fija** es la evidencia material del
  RNF-06: la campaña es regenerable byte a byte.
* **El sesgo del subconjunto**: la exactitud se calcula solo sobre rostros
  detectados, así que dos cadenas con distinta tasa de detección (10,7 % frente
  a 28,2 % de no detectados) no se miden sobre el mismo material, y la que menos
  detecta se queda con las caras fáciles. Por eso `report.md` incluye ahora una
  segunda tabla sobre las imágenes que detectan todas las cadenas. La tabla
  principal sigue siendo la buena para cobertura y latencia, que son propiedades
  de la cadena completa; la del subconjunto común es la que permite afirmar cuál
  clasifica mejor.
* **`resultados/` no estaba montado en el contenedor**, que se lanza con `--rm`.
  La comparativa decía escribir en `/ros2_ws/resultados` y en el anfitrión
  seguían viéndose las tablas de días atrás. Ya está corregido en `docker/run.sh`.
