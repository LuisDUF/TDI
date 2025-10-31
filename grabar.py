import sys, os, time, cv2, threading
from PyQt5 import QtCore, QtGui, QtWidgets, uic

# --- --- --- CONFIGURACIÓN (igual a tu script CLI) --- --- --- #
BASE_DIR = "./dataset/train"     # Directorio base para los datos
CLIP_DURATION = 15               # segundos
FPS = 14                         # fotogramas por segundo
RESOLUTION = (640, 480)          # (ancho, alto)
# --- --- ---------------------------------------------- --- --- #

def get_next_clip_number(person_path: str) -> int:
    """Devuelve el siguiente número de clip disponible (1, 2, 3...)."""
    if not os.path.exists(person_path):
        return 1
    existing = [f for f in os.listdir(person_path) if f.endswith(".mp4") and f.startswith("clip_")]
    if not existing:
        return 1
    max_num = 0
    for fname in existing:
        try:
            num = int(fname.replace("clip_", "").replace(".mp4", ""))
            if num > max_num:
                max_num = num
        except ValueError:
            continue
    return max_num + 1

# ---------------- Worker de cámara en QThread ---------------- #

class CameraWorker(QtCore.QObject):
    frameReady = QtCore.pyqtSignal(QtGui.QImage)
    statusMsg = QtCore.pyqtSignal(str, str)          # (texto, color_hex)
    instructionMsg = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)                # 0-100
    countdownTick = QtCore.pyqtSignal(int)           # segundos restantes
    clipSaved = QtCore.pyqtSignal(str)               # ruta del clip guardado
    cameraReady = QtCore.pyqtSignal(int, int, float) # (width, height, fps_real)
    error = QtCore.pyqtSignal(str)

    def __init__(self, person_name: str, parent=None):
        super().__init__(parent)
        self.person_name = person_name
        self._running = True
        self._go_event = threading.Event()  # lo dispara el botón "Grabar clip"
        self._cap = None
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._recording = False

        # Instrucción actual (la define la UI al pulsar grabar)
        self.current_instruction = (
            "Colóquese a ~5 metros.\nCaminar HORIZONTALMENTE (perpendicular a la cámara)."
        )

        # Carpeta ./dataset/train/<persona>
        self.person_path = os.path.join(BASE_DIR, self.person_name)
        os.makedirs(self.person_path, exist_ok=True)

    @QtCore.pyqtSlot()
    def stop(self):
        self._running = False
        self._go_event.set()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.statusMsg.emit("Abriendo cámara…", "#6aa6ff")
            # Si tu cámara en Windows requiere CAP_DSHOW, descomenta esta línea:
            # self._cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                raise IOError("No se puede abrir la cámara web")

            # Ajustar resolución/fps deseados
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
            self._cap.set(cv2.CAP_PROP_FPS, FPS)

            width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or RESOLUTION[0]
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or RESOLUTION[1]
            fps_cam = self._cap.get(cv2.CAP_PROP_FPS) or FPS

            # Hilo para emitir frames al QLabel (sin textos superpuestos)
            grab_thread = threading.Thread(target=self._grab_loop, args=(width, height), daemon=True)
            grab_thread.start()

            # Mensajes iniciales
            self.instructionMsg.emit(
                "Colóquese a ~5 metros.\n"
                "Seleccione el tipo de caminata y pulse «Grabar clip» para iniciar una toma de 15s."
            )
            self.statusMsg.emit("Cámara activa. Esperando a que pulses «Grabar clip».", "#6aa6ff")
            self.cameraReady.emit(width, height, fps_cam)

            # Bucle pasivo: espera a que la UI dispare grabación
            while self._running:
                self._go_event.clear()
                self._go_event.wait()
                if not self._running:
                    break
                self._record_one_clip(width, height, FPS)

            self.statusMsg.emit("Cámara detenida.", "#9aa0a6")

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if self._cap is not None and self._cap.isOpened():
                self._cap.release()

    def _grab_loop(self, width, height):
        while self._running and self._cap and self._cap.isOpened():
            ok, frame = self._cap.read()
            if not ok:
                continue
            # LED rojo cuando está grabando (sin textos)
            if self._recording:
                cv2.circle(frame, (18, 18), 10, (0, 0, 255), -1)
            # Convertir a RGB y QImage
            frame_resized = cv2.resize(frame, (width, height))
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
            self.frameReady.emit(qimg)
            time.sleep(0.01)

    def trigger_record(self, instruction_text: str):
        """La UI llama a esto pasando la instrucción elegida (5 m + tipo de caminata)."""
        self.current_instruction = instruction_text
        # Refrescar el panel de instrucciones antes de la cuenta atrás:
        self.instructionMsg.emit(self.current_instruction)
        self._go_event.set()

    def _record_one_clip(self, width, height, fps_target):
        try:
            # Mostrar instrucción actual en panel (no sobre el video)
            # (ya se emite en trigger_record)
            self.statusMsg.emit("¡Prepárate! Iniciando cuenta regresiva…", "#ffcc00")
            for i in range(3, 0, -1):
                self.countdownTick.emit(i)
                time.sleep(1)
            self.countdownTick.emit(0)

            # Ruta: ./dataset/train/<persona>/clip_XXX.mp4
            clip_num = get_next_clip_number(self.person_path)
            clip_filename = f"clip_{clip_num:03d}.mp4"
            video_path = os.path.join(self.person_path, clip_filename)

            # VideoWriter con parámetros del dataset (RESOLUTION/FPS)
            out = cv2.VideoWriter(video_path, self._fourcc, fps_target, RESOLUTION)

            # Grabación
            self._recording = True
            self.statusMsg.emit(f"GRABANDO Clip #{clip_num}…", "#ff4d4d")
            start = time.time()
            self.progress.emit(0)
            while self._running and (time.time() - start) < CLIP_DURATION:
                ok, frame = self._cap.read()
                if not ok:
                    break
                frame_resized = cv2.resize(frame, RESOLUTION)
                # LED discreto en esquina
                cv2.circle(frame_resized, (18, 18), 10, (0, 0, 255), -1)
                out.write(frame_resized)

                elapsed = time.time() - start
                pct = int(100 * elapsed / CLIP_DURATION)
                self.progress.emit(min(99, pct))
                time.sleep(0.001)

            out.release()
            self._recording = False
            self.progress.emit(100)
            self.statusMsg.emit(f"✅ Clip guardado: {video_path}", "#3ddc97")
            self.clipSaved.emit(video_path)
            time.sleep(0.4)
            # Reafirmar la instrucción (útil si haces varios clips del mismo tipo)
            self.instructionMsg.emit(self.current_instruction)
            self.statusMsg.emit("Listo para otro clip. Pulsa «Grabar clip» cuando quieras.", "#6aa6ff")
        except Exception as e:
            self._recording = False
            raise e

# ---------------- Ventana principal ---------------- #

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("mainwindow.ui", self)

        self.worker = None
        self.thread = None

        # Ajustes de textos para este flujo de dataset
        self.titleLabel.setText("Captura de datos (YOLO)")
        self.subtitleLabel.setText("Clips en ./dataset/train/<persona>/clip_XXX.mp4 (15s • 640×480 • 14 FPS)")
        self.startBtn.setText("Iniciar cámara")
        self.goBtn.setText("Grabar clip")
        self.instructionLbl.setText("Instrucciones aparecerán aquí…")

        # === Añadimos un selector de tipo de caminata (sin modificar el .ui) ===
        # Se insertará arriba del botón "Grabar clip"
        self.modeLabel = QtWidgets.QLabel("Tipo de toma")
        self.modeLabel.setStyleSheet("color:#c9d1d9;")
        self.modeCombo = QtWidgets.QComboBox()
        self.modeCombo.addItems([
            "Caminata HORIZONTAL (perpendicular)",
            "Caminata FRONTAL (hacia la cámara)"
        ])
        # Insertar en el contenedor de controles
        # cardControls es un QFrame con layout vertical
        self.cardControls.layout().insertWidget(0, self.modeLabel)
        self.cardControls.layout().insertWidget(1, self.modeCombo)

        # Conexiones UI
        self.startBtn.clicked.connect(self.on_start)
        self.goBtn.clicked.connect(self.on_record)
        self.goBtn.setEnabled(False)
        self.modeCombo.currentIndexChanged.connect(self.on_mode_changed)

        self._set_status("Estado: Esperando", "#6aa6ff")
        self.countdownLbl.setText("")
        self.videoLabel.setScaledContents(True)

        # Pre-cargar instrucción por defecto (horizontal)
        self._update_instruction_text()

    # --- Helpers --- #
    def _set_status(self, text, color="#6aa6ff"):
        self.statusLbl.setText(text)
        self.statusLbl.setStyleSheet(f"color:{color};")

    def _set_instruction(self, text):
        self.instructionLbl.setText(text)

    def _update_instruction_text(self):
        """Actualiza el bloque de instrucciones a partir del modo elegido."""
        mode = self.modeCombo.currentText()
        if "HORIZONTAL" in mode:
            instruction = (
                "Colóquese a ~5 metros.\n"
                "Caminar HORIZONTALMENTE (perpendicular a la cámara)."
            )
        else:
            instruction = (
                "Colóquese a ~5 metros.\n"
                "Caminar DIRECTAMENTE hacia la cámara."
            )
        self._set_instruction(instruction)
        # Si el worker está activo, que también sincronice su instrucción interna
        if self.worker:
            self.worker.current_instruction = instruction

    # --- Slots --- #
    def on_mode_changed(self, _idx):
        self._update_instruction_text()

    def on_start(self):
        person_name = self.nameEdit.text().strip().replace(" ", "_").lower()
        if not person_name:
            QtWidgets.QMessageBox.warning(self, "Falta identificador", "Introduce el identificador para la persona (p. ej. persona_01).")
            return

        # Crear carpeta base/persona
        os.makedirs(os.path.join(BASE_DIR, person_name), exist_ok=True)

        self.startBtn.setEnabled(False)
        self.nameEdit.setEnabled(False)
        self.goBtn.setEnabled(True)
        self.progressBar.setValue(0)
        self._set_status("Iniciando cámara…", "#6aa6ff")

        # Thread + Worker
        self.thread = QtCore.QThread()
        self.worker = CameraWorker(person_name)
        self.worker.moveToThread(self.thread)

        # Señales
        self.worker.frameReady.connect(self.update_frame)
        self.worker.statusMsg.connect(self._set_status)
        self.worker.instructionMsg.connect(self._set_instruction)
        self.worker.progress.connect(self.progressBar.setValue)
        self.worker.countdownTick.connect(self.update_countdown)
        self.worker.clipSaved.connect(self.on_clip_saved)
        self.worker.error.connect(self.on_error)
        self.worker.cameraReady.connect(self.on_camera_ready)

        self.thread.started.connect(self.worker.run)
        self.destroyed.connect(self.cleanup)
        self.thread.start()

        # Sincronizar instrucción actual con el modo elegido
        self._update_instruction_text()

    def on_record(self):
        if self.worker:
            # Ensamblar la instrucción acorde al modo
            instruction = self.instructionLbl.text()
            self.goBtn.setEnabled(False)  # se re-habilita al terminar el clip
            self.worker.trigger_record(instruction)

    @QtCore.pyqtSlot()
    def cleanup(self):
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait(1000)

    def closeEvent(self, event):
        self.cleanup()
        return super().closeEvent(event)

    @QtCore.pyqtSlot(QtGui.QImage)
    def update_frame(self, qimg):
        self.videoLabel.setPixmap(QtGui.QPixmap.fromImage(qimg))

    @QtCore.pyqtSlot(int)
    def update_countdown(self, secs):
        self.countdownLbl.setText("" if secs == 0 else str(secs))
        if secs > 0:
            self.countdownLbl.setStyleSheet("font-size: 48pt; font-weight:700; color:white;")
            QtCore.QTimer.singleShot(200, lambda: self.countdownLbl.setStyleSheet("font-size: 36pt; font-weight:700; color:white;"))

    @QtCore.pyqtSlot(str)
    def on_clip_saved(self, path):
        # Tras guardar, puedes grabar otro clip
        self.goBtn.setEnabled(True)

    @QtCore.pyqtSlot(int, int, float)
    def on_camera_ready(self, w, h, fps):
        self.subtitleLabel.setText(
            f"Guardando en {BASE_DIR}/<persona>/clip_XXX.mp4 • Resolución: {RESOLUTION[0]}x{RESOLUTION[1]} • FPS: {FPS}"
        )

    @QtCore.pyqtSlot(str)
    def on_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Error de cámara", msg)
        self._reset_ui()

    def _reset_ui(self):
        self.startBtn.setEnabled(True)
        self.nameEdit.setEnabled(True)
        self.goBtn.setEnabled(False)
        self.progressBar.setValue(0)
        self.countdownLbl.setText("")
        self._set_status("Estado: Esperando", "#6aa6ff")
        self._update_instruction_text()
        self.cleanup()

# ---------------- main ---------------- #

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Tema oscuro base
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.Window, QtGui.QColor("#0b0d12"))
    palette.setColor(palette.WindowText, QtCore.Qt.white)
    palette.setColor(palette.Base, QtGui.QColor("#0f1115"))
    palette.setColor(palette.AlternateBase, QtGui.QColor("#111418"))
    palette.setColor(palette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(palette.ToolTipText, QtCore.Qt.white)
    palette.setColor(palette.Text, QtCore.Qt.white)
    palette.setColor(palette.Button, QtGui.QColor("#1c1f24"))
    palette.setColor(palette.ButtonText, QtCore.Qt.white)
    palette.setColor(palette.BrightText, QtCore.Qt.red)
    palette.setColor(palette.Highlight, QtGui.QColor("#2a6df4"))
    palette.setColor(palette.HighlightedText, QtCore.Qt.white)
    app.setPalette(palette)

    w = MainWindow()
    w.resize(980, 600)
    w.show()
    sys.exit(app.exec_())
