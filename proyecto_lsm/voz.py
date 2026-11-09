import threading
import re


def seleccionar_voz_espanola(voces):
    candidatas = []
    for voz in voces:
        idiomas = voz.languages or []
        if isinstance(idiomas, (str, bytes)):
            # Algunos sistemas entregan un solo idioma en lugar de una lista.
            idiomas = [idiomas]
        idiomas = " ".join(
            idioma.decode("utf-8", errors="ignore")
            if isinstance(idioma, bytes) else str(idioma)
            for idioma in idiomas
        ).lower()
        nombre = f"{voz.id} {voz.name}".lower()
        # Los idiomas de eSpeak pueden empezar con un byte de control.
        etiquetas = re.sub(r"[\x00-\x1f]", "", idiomas).replace("_", "-")
        descripcion = nombre.replace("_", "-")
        if (re.search(r"\bes(?:-[a-z]+)?\b", etiquetas)
                or re.search(r"\bes(?:-[a-z]+)?\b", descripcion)
                or any(p in nombre for p in ("spanish", "español", "espanol"))):
            preferida = any(p in f"{etiquetas} {descripcion}" for p in (
                "es-mx", "es-419", "latin", "mexic",
            ))
            candidatas.append((preferida, voz))
    return max(candidatas, key=lambda candidata: candidata[0])[1] if candidatas else None


class ReproductorVoz:
    def __init__(self):
        # Impide que varias pulsaciones de V generen voces simultáneas.
        self._ocupado = threading.Lock()

    def reproducir(self, frase):
        """Valida el texto e inicia su lectura sin bloquear el ciclo de la cámara."""
        texto = frase.strip()
        if not texto:
            print("Forma una frase antes de reproducirla.")
            return
        if not self._ocupado.acquire(blocking=False):
            print("Espera a que termine la lectura actual.")
            return
        # El hilo recibe el texto actual; editar la frase no cambia la lectura en curso.
        # daemon=True permite salir del programa aunque todavía esté hablando.
        hilo = threading.Thread(target=self._hablar, args=(texto,), daemon=True)
        try:
            hilo.start()
        except Exception:
            self._ocupado.release()
            raise

    def _hablar(self, texto):
        motor = None
        try:
            import pyttsx3

            # Crear y usar el motor en el mismo hilo evita bloquear la cámara.
            motor = pyttsx3.init()
            voz = seleccionar_voz_espanola(motor.getProperty("voices"))
            if voz is None:
                # Evita recurrir a una voz predeterminada que podría estar en inglés.
                print(
                    "No hay una voz en español instalada. Instala una voz en español "
                    "en tu sistema (en Ubuntu/Debian: sudo apt install espeak-ng) "
                    "y reinicia el programa."
                )
                return
            motor.setProperty("voice", voz.id)
            # Velocidad de lectura aproximada en palabras por minuto.
            motor.setProperty("rate", 150)
            print(f"Voz en español: {voz.name} | Reproduciendo: {texto}")
            motor.say(texto)
            # Esta espera ocurre en el hilo de voz, por eso la cámara sigue funcionando.
            motor.runAndWait()
        except ImportError:
            print("Falta pyttsx3. Instala las dependencias: pip install -r requirements.txt")
        except Exception as error:
            print(
                f"No se pudo reproducir la frase: {error}. "
                "Verifica la salida de audio y el sintetizador de voz "
                "(en Linux, instala espeak-ng)."
            )
        finally:
            # Libera el motor y permite una nueva lectura incluso si hubo un error.
            try:
                if motor is not None:
                    motor.stop()
            finally:
                self._ocupado.release()
