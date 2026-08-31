# Estado publicado por el gestor de comportamiento (auditable).
std_msgs/Header header

uint8   behaviour_state
string  behaviour_name
string  transition_reason    # guarda que provoco la ultima transicion
float32 time_in_state_s
float32 distress_index
bool    alert_active
string  session_id           # identificador aleatorio, sin vinculo clinico
