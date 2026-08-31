# Indice continuo de malestar tras la fusion temporal.
std_msgs/Header header

float32 distress_index      # D(t) en [0,1]
float32 smoothed_confidence # confianza media ponderada de la ventana
bool    valid               # false si no hay observacion reciente (>2 s)
float32 seconds_since_face  # tiempo desde la ultima deteccion valida
