# -*- coding: utf-8 -*-
"""Backend de percepción facial de Pediatric-Companion.

Este fichero es PYTHON PURO: no depende de ROS 2 y por eso puede probarse con
`pytest` sin arrancar ningún nodo. Encapsula toda la parte "sucia" (MediaPipe,
OpenCV) detrás de una interfaz estable que consume `perception_node.py`.

Estrategia de degradación elegante
----------------------------------
1. Si MediaPipe está instalado y existe el modelo `face_landmarker.task`,
   se usa la cadena completa: BlazeFace + Face Landmarker (52 blendshapes).
2. Si MediaPipe no está disponible, se usa un detector Haar de OpenCV y un
   estimador geométrico simplificado. El sistema sigue arrancando y las
   pruebas de integración siguen pasando, lo que evita bloquear el desarrollo
   del resto del sistema por una dependencia de visión.

El clasificador afectivo trabaja SIEMPRE sobre el vector de 52 coeficientes de
expresión, nunca sobre píxeles: ese es el punto exacto en el que el dato deja
de ser biométrico (véase apartado 4.7 de la memoria).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Tamaño mínimo de imagen que se entrega a un detector facial. Los conjuntos de
# referencia del ámbito (FER-2013, CAFE) distribuyen recortes de 48x48 px: a esa
# escala el detector de Haar no puede operar y MediaPipe pierde precisión. El
# preprocesado de `_preparar` reescala y añade margen para que la ventana de
# detección tenga recorrido. Véase el apartado 5.3.4 de la memoria.
MIN_LADO_PX = 96
MARGEN_REL = 0.25          # margen replicado, en fracción del lado


def _preparar(bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
    """Acondiciona un fotograma pequeño para la detección facial.

    Devuelve `(imagen, escala, dx, dy)`, donde la caja detectada sobre la imagen
    preparada se devuelve a coordenadas originales con
    `x_orig = (x_prep - dx) / escala`. Las imágenes que ya son suficientemente
    grandes se devuelven intactas y sin coste alguno (escala 1, sin margen).

    Sin este paso, un recorte de 48x48 nunca produce detección: es la causa del
    100 % de rostros no detectados observado en la primera comparativa.
    """
    if bgr is None or bgr.size == 0:
        return bgr, 1.0, 0, 0
    h, w = bgr.shape[:2]
    lado = min(h, w)
    if lado >= MIN_LADO_PX:
        return bgr, 1.0, 0, 0

    import cv2
    escala = float(MIN_LADO_PX * 2) / float(lado)      # 48 -> 192
    grande = cv2.resize(bgr, (int(round(w * escala)), int(round(h * escala))),
                        interpolation=cv2.INTER_CUBIC)
    margen = int(round(min(grande.shape[:2]) * MARGEN_REL))
    conmargen = cv2.copyMakeBorder(grande, margen, margen, margen, margen,
                                   cv2.BORDER_REPLICATE)
    return conmargen, escala, margen, margen


def _cargar_cascada(cv2):
    """Localiza y carga el clasificador de Haar frontal, o devuelve None.

    `cv2.data.haarcascades` apunta al directorio de datos del paquete de pip.
    En la imagen del proyecto conviven el OpenCV del sistema (python3-opencv) y
    el de pip, y esa ruta puede existir como cadena pero no como fichero. El
    resultado era que `CascadeClassifier` devolvía un objeto vacío —no None—,
    de modo que la comprobación `is None` no lo detectaba y el fallo aparecía
    más tarde, al llamar a `detectMultiScale`.

    Se prueban las ubicaciones habituales y se verifica con `empty()`, que es
    la única forma fiable de saber si el clasificador se ha cargado.
    """
    # En OpenCV 5 el clasificador en cascada dejó de estar expuesto en el
    # espacio de nombres principal. Se comprueba antes de nada, porque el
    # error de atributo es mucho menos legible que este mensaje.
    if not hasattr(cv2, "CascadeClassifier"):
        return None

    nombre = "haarcascade_frontalface_default.xml"

    # Copia propia, versionada junto al modelo de MediaPipe. Va primero a
    # propósito: es la única ubicación que no depende de qué OpenCV haya
    # instalado ni de si el paquete opencv-data está presente. Depender del
    # entorno para un fichero de 900 KB costó dos comparativas inválidas.
    candidatos = [os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "models", nombre)]
    try:                             # ruta de instalación de ROS 2, si la hay
        from ament_index_python.packages import get_package_share_directory
        candidatos.append(os.path.join(
            get_package_share_directory("pedcom_perception"), "models", nombre))
    except Exception:                # noqa: BLE001 - este fichero no exige ROS
        pass

    datos = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if datos:
        candidatos.append(os.path.join(datos, nombre))
    candidatos += [
        os.path.join("/usr/share/opencv4/haarcascades", nombre),
        os.path.join("/usr/share/opencv/haarcascades", nombre),
        os.path.join(os.path.dirname(cv2.__file__), "data", nombre),
    ]
    for ruta in candidatos:
        if not os.path.isfile(ruta):
            continue
        cascada = cv2.CascadeClassifier(ruta)
        if not cascada.empty():
            return cascada
    return None


def _deshacer(box: Tuple[int, int, int, int], escala: float,
              dx: int, dy: int) -> Tuple[int, int, int, int]:
    """Lleva una caja detectada sobre la imagen preparada al sistema original."""
    if escala == 1.0 and dx == 0 and dy == 0:
        return box
    x, y, w, h = box
    return (int(round((x - dx) / escala)), int(round((y - dy) / escala)),
            int(round(w / escala)), int(round(h / escala)))

# Orden canónico de las siete categorías afectivas (coincide con EmotionalState.msg)
EMOTIONS: List[str] = [
    "neutral", "joy", "sadness", "anger", "fear", "surprise", "pain",
]

# Pesos del índice de malestar D(t). Apartado 4.3.3 de la memoria.
DISTRESS_WEIGHTS = np.array([0.00, -0.50, 0.60, 0.55, 0.85, 0.00, 1.00],
                            dtype=np.float32)

# Nombres de los blendshapes de MediaPipe usados por el clasificador de reserva.
# Se corresponden conceptualmente con unidades de acción del sistema FACS.
_BS_KEYS = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "jawOpen", "mouthFrownLeft", "mouthFrownRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthPressLeft", "mouthPressRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthPucker",
    "noseSneerLeft", "noseSneerRight", "cheekSquintLeft", "cheekSquintRight",
]


@dataclass
class FaceObservation:
    """Resultado de una pasada de percepción sobre un único fotograma."""

    face_present: bool = False
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)   # x, y, w, h
    confidence: float = 0.0
    center_uv: Tuple[float, float] = (0.0, 0.0)      # normalizado en [-1, 1]
    probabilities: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    inference_ms: float = 0.0

    @property
    def dominant_index(self) -> int:
        return int(np.argmax(self.probabilities))

    @property
    def dominant_emotion(self) -> str:
        return EMOTIONS[self.dominant_index]


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


class MLPClassifier:
    """Perceptrón multicapa 52 -> 64 -> 32 -> 7 (apartado 4.3.2).

    Los pesos se cargan de un fichero `.npz` generado por el script de
    entrenamiento. Si no existe, se usa un clasificador basado en reglas sobre
    los blendshapes, que permite depurar todo el sistema sin modelo entrenado.
    """

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.ready = False
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        if weights_path and os.path.isfile(weights_path):
            data = np.load(weights_path)
            self.W = [data[f"W{i}"] for i in range(3)]
            self.b = [data[f"b{i}"] for i in range(3)]
            self.ready = True

    @property
    def n_entradas(self) -> int:
        """Dimensión de entrada que espera el modelo entrenado."""
        return int(self.W[0].shape[0]) if self.ready else len(_BS_KEYS)

    def predict(self, features: np.ndarray,
                by_name: Optional[dict] = None) -> np.ndarray:
        """Clasifica un vector de expresión.

        `features` es el vector COMPLETO de blendshapes en el orden canónico de
        MediaPipe (52 coeficientes), que es exactamente lo que consumió el
        entrenamiento (`scripts/train_affect_mlp.py`). `by_name` es el mismo
        dato indexado por nombre de categoría y solo lo usa el clasificador de
        reglas, que sí razona sobre unidades de acción concretas.

        Mantener las dos representaciones separadas evita el error que se
        arrastraba: la inferencia construía un vector de 25 posiciones a partir
        de una selección de nombres, mientras que el MLP esperaba los 52
        coeficientes en bruto. El fallo permaneció latente mientras MediaPipe
        no llegó a cargarse, porque el camino de reserva nunca usa el MLP.
        """
        if not self.ready:
            return self._rule_based(by_name if by_name is not None else features)

        h = np.asarray(features, dtype=np.float32).ravel()
        if h.size != self.n_entradas:
            raise ValueError(
                f"El clasificador afectivo espera {self.n_entradas} "
                f"coeficientes de expresión y ha recibido {h.size}. El vector "
                "debe ser la lista completa de blendshapes de MediaPipe, en su "
                "orden original y sin filtrar por nombre; es la representación "
                "con la que se entrenó el modelo. Si ha cambiado el conjunto de "
                "características, vuelva a entrenar con "
                "scripts/train_affect_mlp.py.")

        for i in range(2):
            h = np.maximum(h @ self.W[i] + self.b[i], 0.0)   # ReLU
        return softmax(h @ self.W[2] + self.b[2])

    # -- clasificador de reserva -------------------------------------------
    @staticmethod
    def _rule_based(features) -> np.ndarray:
        """Puntuaciones heurísticas sobre blendshapes.

        No pretende ser preciso: su función es permitir que el resto del
        sistema (fusión, máquina de estados, actuación) se desarrolle y se
        pruebe antes de disponer del modelo entrenado.

        Admite un diccionario `{nombre: puntuación}` (lo natural cuando hay
        MediaPipe) o una secuencia posicional alineada con `_BS_KEYS` (lo que
        usan las pruebas unitarias).
        """
        if isinstance(features, dict):
            f = {k: float(features.get(k, 0.0)) for k in _BS_KEYS}
        else:
            v = np.asarray(features, dtype=np.float32).ravel()
            f = {k: (float(v[i]) if i < v.size else 0.0)
                 for i, k in enumerate(_BS_KEYS)}
        brow_down = 0.5 * (f["browDownLeft"] + f["browDownRight"])
        brow_up = f["browInnerUp"]
        eye_squint = 0.5 * (f["eyeSquintLeft"] + f["eyeSquintRight"])
        eye_wide = 0.5 * (f["eyeWideLeft"] + f["eyeWideRight"])
        smile = 0.5 * (f["mouthSmileLeft"] + f["mouthSmileRight"])
        frown = 0.5 * (f["mouthFrownLeft"] + f["mouthFrownRight"])
        stretch = 0.5 * (f["mouthStretchLeft"] + f["mouthStretchRight"])
        sneer = 0.5 * (f["noseSneerLeft"] + f["noseSneerRight"])
        jaw = f["jawOpen"]

        scores = np.array([
            0.35,                                             # neutral
            2.2 * smile,                                      # joy
            1.6 * frown + 0.8 * brow_up,                      # sadness
            1.8 * brow_down + 1.0 * f["mouthPressLeft"],      # anger
            1.5 * eye_wide + 1.2 * brow_up + 0.8 * stretch,   # fear
            1.6 * eye_wide + 1.4 * jaw + 1.0 * brow_up,       # surprise
            1.7 * eye_squint + 1.3 * brow_down + 1.2 * sneer  # pain
            + 0.9 * stretch,
        ], dtype=np.float32)
        return softmax(scores * 3.0)


class PerceptionBackend:
    """Cadena completa: fotograma -> observación afectiva."""

    def __init__(self,
                 landmarker_model: str = "",
                 weights_path: str = "",
                 min_detection_confidence: float = 0.5) -> None:
        self.clf = MLPClassifier(weights_path or None)
        self.min_conf = float(min_detection_confidence)
        self.mode = "fallback"
        # Motivo por el que no se está usando MediaPipe. Nunca debe quedar en
        # silencio: una degradación no anunciada se confunde con un resultado.
        self.motivo_degradacion = ""
        self._landmarker = None
        self._cascade = None
        self._init_mediapipe(landmarker_model)
        if self._landmarker is None:
            self._init_opencv()
            warnings.warn(
                "PerceptionBackend degradado a OpenCV/Haar: "
                f"{self.motivo_degradacion or 'causa desconocida'}. "
                "Las métricas obtenidas en este modo NO son comparables con "
                "las de la cadena MediaPipe.", RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------------ init
    def _init_mediapipe(self, model_path: str) -> None:
        if not model_path:
            self.motivo_degradacion = "no se ha indicado ruta del modelo .task"
            return
        if not os.path.isfile(model_path):
            self.motivo_degradacion = f"no existe el modelo {model_path!r}"
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=2,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                min_face_detection_confidence=self.min_conf,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
            self.mode = "mediapipe"
        except Exception as exc:     # noqa: BLE001 - degradación deliberada
            # La degradación es deliberada; ocultar su causa no lo es. Sin esta
            # traza, un fallo de instalación se manifiesta como una exactitud
            # del 0 % y se confunde con un mal resultado del algoritmo.
            self._landmarker = None
            self.motivo_degradacion = f"{type(exc).__name__}: {exc}"

    def _init_opencv(self) -> None:
        try:
            import cv2
            self._cascade = _cargar_cascada(cv2)
            self._cv2 = cv2
            if self._cascade is None:
                self.motivo_degradacion += (
                    " · y el clasificador de Haar tampoco se ha podido cargar: "
                    "no se encuentra haarcascade_frontalface_default.xml. "
                    "Sin MediaPipe ni Haar, este backend no detecta nada.")
        except Exception as exc:     # noqa: BLE001
            self._cascade = None
            self.motivo_degradacion += f" · OpenCV tampoco disponible: {exc}"

    # --------------------------------------------------------------- proceso
    def process(self, bgr: np.ndarray,
                roi_bounds: Optional[Tuple[float, float, float, float]] = None
                ) -> FaceObservation:
        """Procesa un fotograma BGR y devuelve la observación afectiva.

        `roi_bounds` delimita, en coordenadas normalizadas (u0, v0, u1, v1), la
        región donde se espera el rostro del paciente (la cama). Los rostros
        fuera de esa región se descartan: evita estimar el estado emocional de
        un acompañante, lo que sería un tratamiento sin base jurídica (4.3.1).
        """
        import time
        t0 = time.perf_counter()
        obs = FaceObservation()
        if bgr is None or bgr.size == 0:
            return obs
        h, w = bgr.shape[:2]

        if self._landmarker is not None:
            obs = self._process_mediapipe(bgr, w, h)
        elif self._cascade is not None:
            obs = self._process_opencv(bgr, w, h)

        if obs.face_present and roi_bounds is not None:
            u, v = obs.center_uv
            u0, v0, u1, v1 = roi_bounds
            if not (u0 <= u <= u1 and v0 <= v <= v1):
                obs = FaceObservation()          # fuera de la zona del paciente

        obs.inference_ms = (time.perf_counter() - t0) * 1000.0
        return obs

    def _process_mediapipe(self, bgr, w, h) -> FaceObservation:
        rgb = bgr[:, :, ::-1].copy()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return FaceObservation()

        # Selección del rostro de mayor área (el más cercano = el paciente)
        best_i, best_area, best_box = 0, -1.0, (0, 0, 0, 0)
        for i, lms in enumerate(result.face_landmarks):
            xs = [lm.x for lm in lms]
            ys = [lm.y for lm in lms]
            x0, x1 = max(min(xs), 0.0), min(max(xs), 1.0)
            y0, y1 = max(min(ys), 0.0), min(max(ys), 1.0)
            area = (x1 - x0) * (y1 - y0)
            if area > best_area:
                best_area, best_i = area, i
                best_box = (int(x0 * w), int(y0 * h),
                            int((x1 - x0) * w), int((y1 - y0) * h))

        # Dos representaciones del mismo dato, cada una para su consumidor:
        #  · `feats`   : los coeficientes crudos en el orden nativo de MediaPipe,
        #                que es sobre lo que se entrenó el MLP (train_affect_mlp).
        #  · `by_name` : indexados por nombre de categoría, para el clasificador
        #                de reglas, que razona sobre unidades de acción concretas.
        # Confundirlas era la causa del ValueError de matmul (52 frente a 25).
        feats = np.zeros(self.clf.n_entradas, dtype=np.float32)
        by_name: dict = {}
        if result.face_blendshapes:
            categorias = result.face_blendshapes[best_i]
            feats = np.array([c.score for c in categorias], dtype=np.float32)
            by_name = {c.category_name: float(c.score) for c in categorias}

        x, y, bw, bh = best_box
        return FaceObservation(
            face_present=True,
            bbox=best_box,
            confidence=float(min(1.0, 0.5 + best_area * 4.0)),
            center_uv=(2.0 * (x + bw / 2) / w - 1.0, 2.0 * (y + bh / 2) / h - 1.0),
            probabilities=self.clf.predict(feats, by_name),
        )

    def _process_opencv(self, bgr, w, h) -> FaceObservation:
        # Acondicionamiento previo: sin él, un recorte de 48x48 px no puede
        # producir NINGUNA detección, porque la ventana mínima del clasificador
        # es del tamaño de la imagen completa.
        prep, escala, dx, dy = _preparar(bgr)
        gray = self._cv2.cvtColor(prep, self._cv2.COLOR_BGR2GRAY)
        gray = self._cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(gray, 1.15, 5, minSize=(24, 24))
        if len(faces) == 0:
            return FaceObservation()
        x, y, bw, bh = _deshacer(
            tuple(max(faces, key=lambda f: f[2] * f[3])), escala, dx, dy)
        # Sin blendshapes reales: se emite una estimación neutra de baja confianza.
        probs = np.array([0.70, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                         dtype=np.float32)
        return FaceObservation(
            face_present=True,
            bbox=(int(x), int(y), int(bw), int(bh)),
            confidence=0.45,
            center_uv=(2.0 * (x + bw / 2) / w - 1.0, 2.0 * (y + bh / 2) / h - 1.0),
            probabilities=probs,
        )


class DeepFaceBackend:
    """Cadena de percepción alternativa basada en DeepFace (apartado 5.3.3).

    Cumple exactamente la misma interfaz que `PerceptionBackend` —constructor
    compatible y método `process(bgr, roi_bounds) -> FaceObservation`— para que
    la comparativa del objetivo OE3 se realice en igualdad de condiciones: mismo
    preprocesado, mismas imágenes, mismo equipo y misma instrumentación de
    latencia. El resto del sistema es indiferente al backend empleado.

    Advertencia metodológica: DeepFace no dispone de una clase de dolor. La
    asignación de 'disgust' a la clase de dolor se apoya en la proximidad
    morfológica de ambas configuraciones faciales (arrugamiento nasal y
    elevación del labio superior), pero es una aproximación y no una
    equivalencia. Véanse los apartados 5.3.3 y 7.1 de la memoria.
    """

    # Vocabulario de DeepFace -> vocabulario de siete clases del sistema
    _MAP = {
        "neutral": "neutral", "happy": "joy", "sad": "sadness",
        "angry": "anger", "fear": "fear", "surprise": "surprise",
        "disgust": "pain",
    }

    def __init__(self,
                 landmarker_model: str = "",
                 weights_path: str = "",
                 min_detection_confidence: float = 0.5) -> None:
        # Los parámetros de MediaPipe se aceptan y se ignoran: mantener la firma
        # idéntica es lo que permite intercambiar backends sin tocar el nodo.
        self.min_conf = float(min_detection_confidence)
        self.mode = "deepface"
        self.clf = MLPClassifier(None)      # no se usa: DeepFace ya clasifica
        self.available = False
        self._df = None
        self._cascade = None
        self._init_deepface()
        self._init_opencv()

    # ------------------------------------------------------------------ init
    def _init_deepface(self) -> None:
        try:
            from deepface import DeepFace
            self._df = DeepFace
            self.available = True
            self.motivo_degradacion = ""
        except Exception as exc:     # noqa: BLE001 - dependencia opcional
            self._df = None
            self.available = False
            self.motivo_degradacion = f"{type(exc).__name__}: {exc}"

    def _init_opencv(self) -> None:
        """Detector propio: se detecta una sola vez y se recorta la ROI.

        Se evita así que DeepFace ejecute su propia detección, lo que duplicaría
        el coste y haría incomparables las latencias de ambos backends.
        """
        try:
            import cv2
            self._cascade = _cargar_cascada(cv2)
            self._cv2 = cv2
        except Exception as exc:     # noqa: BLE001
            self._cascade = None
            self.motivo_degradacion += f" · OpenCV no disponible: {exc}"

        if self._cascade is None:
            # Sin detector propio, `_detect` entrega el fotograma completo a
            # DeepFace. No es un detalle menor: la etapa de detección
            # desaparece, el porcentaje de rostros no detectados cae a cero de
            # forma artificial y la latencia deja de ser comparable con la de
            # MediaPipe, que sí detecta. La comparativa del OE3 quedaría
            # sesgada a favor de DeepFace sin que nada lo delatase.
            self.detector_propio = False
            self.motivo_degradacion += (
                " · SIN detector de Haar: DeepFace recibiría el fotograma "
                "completo, sin etapa de detección, y las latencias NO serían "
                "comparables con las de MediaPipe. Causas habituales: falta el "
                "paquete opencv-data (apt-get install -y opencv-data), o está "
                "instalado OpenCV 5, que ya no expone CascadeClassifier "
                "(pip install 'opencv-python-headless<5').")
            warnings.warn(
                "DeepFaceBackend sin detector facial propio: la etapa de "
                "detección desaparece y la comparativa de latencias del "
                "apartado 6.1 NO sería válida." + self.motivo_degradacion,
                RuntimeWarning, stacklevel=2)
        else:
            self.detector_propio = True

    # --------------------------------------------------------------- proceso
    def _detect(self, bgr: np.ndarray):
        if self._cascade is None:
            h, w = bgr.shape[:2]
            return (0, 0, w, h)          # sin detector: se usa el fotograma
        prep, escala, dx, dy = _preparar(bgr)
        gray = self._cv2.cvtColor(prep, self._cv2.COLOR_BGR2GRAY)
        gray = self._cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(gray, 1.15, 5, minSize=(24, 24))
        if len(faces) == 0:
            return None
        x, y, w, h = _deshacer(
            tuple(max(faces, key=lambda f: f[2] * f[3])), escala, dx, dy)
        return (int(x), int(y), int(w), int(h))

    def _map_probabilities(self, result) -> np.ndarray:
        """dict de DeepFace (porcentajes) -> vector de 7 clases normalizado."""
        raw = result[0]["emotion"] if isinstance(result, list) else result["emotion"]
        probs = np.zeros(len(EMOTIONS), dtype=np.float32)
        for df_name, score in raw.items():
            target = self._MAP.get(str(df_name).lower())
            if target is None:
                continue
            probs[EMOTIONS.index(target)] += float(score)
        total = float(probs.sum())
        if total <= 0.0:
            probs[EMOTIONS.index("neutral")] = 1.0
            return probs
        return (probs / total).astype(np.float32)

    def process(self, bgr: np.ndarray,
                roi_bounds: Optional[Tuple[float, float, float, float]] = None
                ) -> FaceObservation:
        import time
        t0 = time.perf_counter()
        obs = FaceObservation()
        if bgr is None or bgr.size == 0 or not self.available:
            return obs
        h, w = bgr.shape[:2]

        box = self._detect(bgr)
        if box is None:
            return obs
        x, y, bw, bh = box
        center_uv = (2.0 * (x + bw / 2) / w - 1.0, 2.0 * (y + bh / 2) / h - 1.0)

        # Filtrado por zona del paciente (misma política que en MediaPipe)
        if roi_bounds is not None:
            u0, v0, u1, v1 = roi_bounds
            if not (u0 <= center_uv[0] <= u1 and v0 <= center_uv[1] <= v1):
                return obs

        # La caja puede desbordar el fotograma tras deshacer el margen: se acota
        # antes de recortar para no entregar a DeepFace un recorte vacío.
        x, y = max(0, x), max(0, y)
        bw, bh = max(1, min(bw, w - x)), max(1, min(bh, h - y))
        crop = bgr[y:y + bh, x:x + bw][:, :, ::-1]        # BGR -> RGB
        try:
            res = self._df.analyze(crop, actions=["emotion"],
                                   detector_backend="skip",
                                   enforce_detection=False)
            probs = self._map_probabilities(res)
        except Exception:            # noqa: BLE001 - inferencia fallida
            return obs
        finally:
            del crop                  # PRIVACIDAD: el recorte no sobrevive

        obs = FaceObservation(
            face_present=True,
            bbox=(x, y, bw, bh),
            confidence=float(min(1.0, 0.5 + (bw * bh) / float(w * h) * 4.0)),
            center_uv=center_uv,
            probabilities=probs,
        )
        obs.inference_ms = (time.perf_counter() - t0) * 1000.0
        return obs


def create_backend(kind: str = "auto", **kwargs):
    """Factoría de backends de percepción (apartado 5.3.5 de la memoria).

    kind = "mediapipe" -> cadena del sistema final (MediaPipe + MLP)
    kind = "deepface"  -> línea base comparativa (OE3)
    kind = "auto"      -> MediaPipe, con degradación a OpenCV si no está

    Si se solicita DeepFace y la biblioteca no está instalada, no se aborta: se
    degrada a MediaPipe y se deja constancia en el atributo `mode`, para que un
    equipo sin la dependencia opcional pueda ejecutar igualmente el sistema.
    """
    kind = (kind or "auto").lower()
    if kind == "deepface":
        backend = DeepFaceBackend(**kwargs)
        if backend.available:
            return backend
        motivo = backend.motivo_degradacion or "causa no registrada"
        warnings.warn(
            "Se ha solicitado el backend DeepFace pero la biblioteca no está "
            f"disponible ({motivo}). Se devuelve la cadena MediaPipe: la fila "
            "'deepface' de la comparativa NO mediría DeepFace. Instálela "
            "(pip install deepface tf-keras) o retire ese backend del "
            "experimento.", RuntimeWarning, stacklevel=2)
        alternativa = PerceptionBackend(**kwargs)
        # El motivo viaja con el objeto devuelto: quien reciba este backend debe
        # poder explicar por qué no es el que pidió, sin tener que capturar el
        # aviso. Antes se perdía y el mensaje de error decía «causa no
        # registrada», que no ayuda a nadie.
        alternativa.motivo_degradacion = (
            f"DeepFace no está instalado ({motivo})"
            + (f" · además, {alternativa.motivo_degradacion}"
               if alternativa.motivo_degradacion else ""))
        return alternativa
    return PerceptionBackend(**kwargs)


def distress_from_probabilities(p: np.ndarray) -> float:
    """Proyección del vector afectivo sobre el índice de malestar D(t)."""
    return float(np.clip(float(DISTRESS_WEIGHTS @ np.asarray(p, dtype=np.float32)),
                         0.0, 1.0))
