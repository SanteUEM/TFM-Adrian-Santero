# Modelos de percepción

Este directorio aloja los modelos que **no** se versionan en el repositorio.

| Fichero | Origen | Descarga |
|---|---|---|
| `face_landmarker.task` | MediaPipe Face Landmarker | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task |
| `mlp_affect.npz` | Clasificador afectivo entrenado por el autor | generado por `scripts/train_affect_mlp.py` |

Descarga del modelo (comando de **Ubuntu**, dentro del contenedor):

```bash
cd /ros2_ws/src/pedcom_perception/models
wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Si el fichero no está presente, `perception_node` arranca igualmente en modo de
reserva (OpenCV + reglas) y lo indica en el registro.
