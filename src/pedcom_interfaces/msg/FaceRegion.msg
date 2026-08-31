# Region de interes del rostro detectado.
# IMPORTANTE: este mensaje NO contiene pixeles, solo geometria y confianza.
std_msgs/Header header

bool  face_present
int32 x          # esquina superior izquierda en pixeles
int32 y
int32 width
int32 height
float32 confidence      # [0,1] confianza del detector facial
float32 center_u        # centro normalizado horizontal  [-1,1]
float32 center_v        # centro normalizado vertical    [-1,1]
