# Orden unificada para la capa de actuacion multisensorial.
std_msgs/Header header

uint8 IDLE=0
uint8 ENGAGE=1
uint8 COMFORT=2
uint8 CALM_DOWN=3
uint8 ALERT=4
uint8 SAFE_STOP=5

uint8   behaviour_state
float32 hue                  # grados [0,360)
float32 saturation           # [0,1]
float32 intensity            # [0,1]
float32 blink_hz             # < 3.0 Hz por seguridad (RNF-04)
string  audio_track          # "" = silencio
float32 audio_gain           # [0,1], limitado a 60 dB(A) (RNF-05)
float32 motion_amplitude_deg # amplitud de los micromovimientos
float32 motion_period_s
