# Pistas de audio

Coloca aquí los ficheros WAV mono a 44,1 kHz:

- `greeting.wav` — melodía breve de saludo (3 s)
- `ambient_loop.wav` — música ambiental continua para el estado COMFORT
- `breathing_6rpm.wav` — guía respiratoria a 6 ciclos por minuto (CALM_DOWN)

Si faltan, `audio_player_node` funciona en modo simulado y publica la pista que
reproduciría, sin emitir sonido. Las pruebas de integración no requieren audio.
