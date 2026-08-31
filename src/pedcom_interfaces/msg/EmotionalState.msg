# Estimacion afectiva instantanea producida por el subsistema de percepcion.
# header.stamp = instante de CAPTURA del fotograma (no de publicacion),
# necesario para medir la latencia extremo a extremo (RNF-01).
std_msgs/Header header

# Indices del vector de probabilidades
uint8 NEUTRAL=0
uint8 JOY=1
uint8 SADNESS=2
uint8 ANGER=3
uint8 FEAR=4
uint8 SURPRISE=5
uint8 PAIN=6

bool    face_present
string  dominant_emotion        # neutral|joy|sadness|anger|fear|surprise|pain
uint8   dominant_index
float32[7] probabilities        # distribucion softmax, suma 1.0
float32 confidence              # confianza del detector facial [0,1]
float32 inference_ms            # tiempo de inferencia de este fotograma
