import pyttsx3
import threading
import queue
import time
import sys

class TTSEngine:
    def __init__(self):
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not text:
                self.queue.task_done()
                continue

            try:
                driver = None
                if sys.platform == "win32": driver = "sapi5"
                elif sys.platform == "darwin": driver = "nsss"
                else: driver = "espeak"

                engine = pyttsx3.init(driverName=driver)
                engine.setProperty("rate", 150)
                engine.setProperty("volume", 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine 
            except Exception as e:
                print(f"TTS ERROR: {e}")
            finally:
                self.queue.task_done()
                time.sleep(0.05) 

    def speak(self, text):
        if text: self.queue.put(text)

    def stop(self):
        self.stop_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)
