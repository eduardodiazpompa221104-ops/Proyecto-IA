class ConfirmadorLetra:

    def __init__(self, segundos_estables=0.8, segundos_sin_mano=0.4):
        self.segundos_estables = segundos_estables
        self.segundos_sin_mano = segundos_sin_mano
        # Bloqueo que se activa después de guardar una letra.
        self.esperando_retiro = False
        # Fracción entre 0 y 1 utilizada por la barra de progreso de la interfaz.
        self.progreso = 0.0
        self._candidata = None
        self._desde = None
        self._sin_mano_desde = None

    def actualizar(self, letra, mano_presente, ahora):
        if not mano_presente:
            # La ausencia de mano cancela la candidata y empieza a medir la pausa.
            self._reiniciar_candidata()
            if self._sin_mano_desde is None:
                self._sin_mano_desde = ahora
            if ahora - self._sin_mano_desde >= self.segundos_sin_mano:
                # La pausa permite otra letra, incluso repetir la anterior.
                self.esperando_retiro = False
            return None

        self._sin_mano_desde = None
        if self.esperando_retiro:
            # Ni una letra distinta se guarda hasta que se retire la mano.
            return None
        if letra is None:
            # Una predicción insegura interrumpe el tiempo de estabilidad.
            self._reiniciar_candidata()
            return None
        if letra != self._candidata:
            # Al cambiar de letra, el tiempo de confirmación empieza desde cero.
            self._candidata = letra
            self._desde = ahora
        self.progreso = min((ahora - self._desde) / self.segundos_estables, 1.0)
        if self.progreso >= 1.0:
            # Se emite una sola vez y se bloquean nuevas letras hasta la pausa.
            self.esperando_retiro = True
            return letra
        return None

    def _reiniciar_candidata(self):
        #Borra el progreso de confirmación sin quitar el bloqueo de retiro.
        self._candidata = None
        self._desde = None
        self.progreso = 0.0
