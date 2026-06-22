# import sys, time, csv, copy, threading
# from typing import Optional
# import numpy as np
# import cv2 as cv
# import requests
# from requests.auth import HTTPBasicAuth

# from PyQt5 import QtCore, QtGui, QtWidgets

# # === your ML imports ===
# import mediapipe as mp
# from model import KeyPointClassifier
# from app_files import calc_landmark_list, draw_info_text, draw_landmarks, pre_process_landmark


# # ------------------------- Helper: convert cv2 -> QImage -------------------------
# def bgr_to_qimage(frame_bgr: np.ndarray) -> QtGui.QImage:
#     h, w, ch = frame_bgr.shape  # ch=3
#     rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
#     qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
#     return qimg


# # ----------------------------- Thread base (safe stop) -----------------------------
# class StoppableThread(threading.Thread):
#     def __init__(self, *a, **kw):
#         super().__init__(*a, **kw)
#         self._stop_event = threading.Event()
#     def stop(self): self._stop_event.set()
#     def stopped(self): return self._stop_event.is_set()


# # ----------------------------- Capture threads -----------------------------
# class WebcamGrabber(StoppableThread):
#     def __init__(self, device_index: int, size: tuple[int, int], flip: bool):
#         super().__init__(daemon=True, name=f"Webcam-{device_index}")
#         self.w, self.h = size
#         self.flip = flip
#         self.device_index = device_index
#         self._lock = threading.Lock()
#         self.frame: Optional[np.ndarray] = None

#     def run(self):
#         cap = cv.VideoCapture(self.device_index)
#         cap.set(cv.CAP_PROP_FRAME_WIDTH, self.w)
#         cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.h)
#         cap.set(cv.CAP_PROP_FPS, 30)
#         while not self.stopped():
#             ok, f = cap.read()
#             if not ok:
#                 time.sleep(0.01); continue
#             if self.flip:
#                 f = cv.flip(f, 1)
#             if (f.shape[1], f.shape[0]) != (self.w, self.h):
#                 f = cv.resize(f, (self.w, self.h), interpolation=cv.INTER_LINEAR)
#             with self._lock:
#                 self.frame = f
#         cap.release()

#     def get(self) -> Optional[np.ndarray]:
#         with self._lock:
#             return None if self.frame is None else self.frame.copy()


# class IPWebcamGrabber(StoppableThread):
#     """Android IP Webcam: try MJPEG /video, fall back to /shot.jpg polling."""
#     def __init__(self, base_url: str, auth: Optional[HTTPBasicAuth], size: tuple[int, int], flip: bool, try_mjpeg: bool):
#         super().__init__(daemon=True, name="IPWebcam")
#         self.base = base_url.rstrip("/")
#         self.auth = auth
#         self.w, self.h = size
#         self.flip = flip
#         self.try_mjpeg = try_mjpeg
#         self._lock = threading.Lock()
#         self.frame: Optional[np.ndarray] = None

#     def _open_mjpeg(self) -> Optional[cv.VideoCapture]:
#         cap = cv.VideoCapture(f"{self.base}/video", cv.CAP_FFMPEG)
#         if cap.isOpened():
#             cap.set(cv.CAP_PROP_BUFFERSIZE, 2)
#             return cap
#         return None

#     def _snapshot(self, s: requests.Session, timeout=2.0) -> Optional[np.ndarray]:
#         r = s.get(f"{self.base}/shot.jpg", auth=self.auth, timeout=timeout)
#         if r.status_code != 200:
#             return None
#         arr = np.frombuffer(r.content, np.uint8)
#         return cv.imdecode(arr, cv.IMREAD_COLOR)

#     def run(self):
#         cap = self._open_mjpeg() if self.try_mjpeg else None
#         if cap is not None and cap.isOpened():
#             while not self.stopped():
#                 ok, f = cap.read()
#                 if not ok:
#                     cap.release(); cap = None; break
#                 if self.flip: f = cv.flip(f, 1)
#                 if (f.shape[1], f.shape[0]) != (self.w, self.h):
#                     f = cv.resize(f, (self.w, self.h), interpolation=cv.INTER_LINEAR)
#                 with self._lock:
#                     self.frame = f

#         s = requests.Session()
#         while not self.stopped():
#             try:
#                 f = self._snapshot(s)
#                 if f is not None:
#                     if self.flip: f = cv.flip(f, 1)
#                     if (f.shape[1], f.shape[0]) != (self.w, self.h):
#                         f = cv.resize(f, (self.w, self.h), interpolation=cv.INTER_LINEAR)
#                     with self._lock:
#                         self.frame = f
#                 else:
#                     time.sleep(0.03)
#             except Exception:
#                 time.sleep(0.05)

#     def get(self) -> Optional[np.ndarray]:
#         with self._lock:
#             return None if self.frame is None else self.frame.copy()


# # ----------------------------- Processing threads (per source) -----------------------------
# class Processor(StoppableThread):
#     """
#     Dedicated processing thread with its OWN MediaPipe graph and classifier.
#     Send frames via set_input(); read last processed via get_output().
#     """
#     def __init__(self, name: str, labels_path: Optional[str] = None):
#         super().__init__(daemon=True, name=name)
#         self._in_lock = threading.Lock()
#         self._out_lock = threading.Lock()
#         self._in_frame: Optional[np.ndarray] = None
#         self._out_frame: Optional[np.ndarray] = None

#         # Per-thread models
#         self.hands = mp.solutions.hands.Hands(
#             static_image_mode=False, max_num_hands=1,
#             min_detection_confidence=0.7, min_tracking_confidence=0.5
#         )
#         self.classifier = KeyPointClassifier()
#         if labels_path:
#             with open(labels_path, encoding='utf-8-sig') as f:
#                 self.labels = [row[0] for row in csv.reader(f)]
#         else:
#             with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
#                 self.labels = [row[0] for row in csv.reader(f)]

#     def set_input(self, img: Optional[np.ndarray]):
#         if img is None: return
#         with self._in_lock:
#             self._in_frame = img.copy()  # replace latest

#     def get_output(self) -> Optional[np.ndarray]:
#         with self._out_lock:
#             return None if self._out_frame is None else self._out_frame.copy()

#     def _process(self, bgr: np.ndarray) -> np.ndarray:
#         debug = copy.deepcopy(bgr)
#         rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
#         rgb.flags.writeable = False
#         results = self.hands.process(rgb)
#         rgb.flags.writeable = True

#         if results.multi_hand_landmarks:
#             for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
#                 landmark_list = calc_landmark_list(debug, lm)
#                 preproc = pre_process_landmark(landmark_list)
#                 sign_id = self.classifier(preproc)
#                 debug = draw_landmarks(debug, landmark_list)
#                 debug = draw_info_text(debug, handed, self.labels[sign_id])
#         return debug

#     def run(self):
#         while not self.stopped():
#             with self._in_lock:
#                 src = None if self._in_frame is None else self._in_frame.copy()
#                 self._in_frame = None
#             if src is None:
#                 time.sleep(0.002); continue
#             out = self._process(src)
#             with self._out_lock:
#                 self._out_frame = out


# # ----------------------------- UI: Zoom-like tiles -----------------------------
# class VideoTile(QtWidgets.QFrame):
#     def __init__(self, title: str):
#         super().__init__()
#         self.setObjectName("videoTile")
#         self.title = title
#         self._qimage: Optional[QtGui.QImage] = None
#         self.setMinimumSize(480, 270)
#         self.setStyleSheet("""
#             QFrame#videoTile {
#                 background-color: #111;
#                 border-radius: 18px;
#             }
#         """)
#         # soft shadow
#         effect = QtWidgets.QGraphicsDropShadowEffect(self)
#         effect.setBlurRadius(28)
#         effect.setOffset(0, 6)
#         effect.setColor(QtGui.QColor(0, 0, 0, 160))
#         self.setGraphicsEffect(effect)

#     def set_frame(self, frame_bgr: np.ndarray):
#         self._qimage = bgr_to_qimage(frame_bgr)
#         self.update()

#     def paintEvent(self, e: QtGui.QPaintEvent):
#         painter = QtGui.QPainter(self)
#         painter.setRenderHint(QtGui.QPainter.Antialiasing)

#         # ✅ use QRectF, not QRect
#         path = QtGui.QPainterPath()
#         path.addRoundedRect(QtCore.QRectF(self.rect()), 18.0, 18.0)
#         painter.setClipPath(path)

#         # draw video
#         if self._qimage:
#             scaled = self._qimage.scaled(
#                 self.size(),
#                 QtCore.Qt.KeepAspectRatioByExpanding,
#                 QtCore.Qt.SmoothTransformation
#             )
#             x = (scaled.width() - self.width()) // 2
#             y = (scaled.height() - self.height()) // 2
#             painter.drawImage(
#                 QtCore.QRect(0, 0, self.width(), self.height()),
#                 scaled,
#                 QtCore.QRect(x, y, self.width(), self.height())
#             )
#         else:
#             painter.fillRect(self.rect(), QtGui.QColor("#222"))

#         # title badge
#         badge_rect = QtCore.QRect(12, 12, 220, 28)
#         painter.setPen(QtCore.Qt.NoPen)
#         painter.setBrush(QtGui.QColor(0, 0, 0, 150))
#         painter.drawRoundedRect(badge_rect, 12, 12)
#         painter.setPen(QtGui.QPen(QtGui.QColor("#fff")))
#         f = painter.font(); f.setPointSize(11); f.setBold(True); painter.setFont(f)
#         painter.drawText(badge_rect.adjusted(10, 0, -10, 0),
#                         QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
#                         self.title)



# class ControlBar(QtWidgets.QFrame):
#     endClicked = QtCore.pyqtSignal()
#     def __init__(self):
#         super().__init__()
#         self.setStyleSheet("""
#             QFrame {
#                 background-color: #0b0b0c;
#                 border-top-left-radius: 16px;
#                 border-top-right-radius: 16px;
#             }
#             QPushButton {
#                 color: #e8e8e8;
#                 background: #1a1a1c;
#                 padding: 10px 16px;
#                 border-radius: 10px;
#             }
#             QPushButton:hover { background: #232326; }
#             QPushButton#danger {
#                 background: #e53935; color: white;
#             }
#             QPushButton#danger:hover { background: #cf2f2b; }
#         """)
#         layout = QtWidgets.QHBoxLayout(self)
#         layout.setContentsMargins(16, 10, 16, 10)
#         layout.setSpacing(12)

#         self.btn_mute = QtWidgets.QPushButton("Mute")
#         self.btn_video = QtWidgets.QPushButton("Stop Video")
#         self.btn_share = QtWidgets.QPushButton("Share (N/A)")
#         self.btn_end = QtWidgets.QPushButton("End")
#         self.btn_end.setObjectName("danger")

#         layout.addStretch(1)
#         layout.addWidget(self.btn_mute)
#         layout.addWidget(self.btn_video)
#         layout.addWidget(self.btn_share)
#         layout.addWidget(self.btn_end)

#         self.btn_end.clicked.connect(self.endClicked.emit)


# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self, args):
#         super().__init__()
#         self.setWindowTitle("WEBCAM (Processed)  |  PHONE (Processed)")
#         self.setStyleSheet("QMainWindow { background-color: #101014; }")
#         self.resize(1280, 720)

#         # --- Layout: two tiles side-by-side + control bar
#         central = QtWidgets.QWidget()
#         self.setCentralWidget(central)
#         v = QtWidgets.QVBoxLayout(central); v.setContentsMargins(18, 18, 18, 12); v.setSpacing(12)
#         tiles = QtWidgets.QHBoxLayout(); tiles.setSpacing(18)

#         self.tile_left = VideoTile("WEBCAM  (Processed)")
#         self.tile_right = VideoTile("PHONE   (Processed)")
#         tiles.addWidget(self.tile_left, 1)
#         tiles.addWidget(self.tile_right, 1)

#         self.controls = ControlBar()
#         v.addLayout(tiles, 1)
#         v.addWidget(self.controls, 0)

#         self.controls.endClicked.connect(self.close)

#         # --- Threads: capture + processing
#         pane_w, pane_h = args.width, args.height
#         base = f"http://{args.ip}:{args.port}"
#         auth = HTTPBasicAuth(args.user, args.password) if args.user and args.password else None

#         self.cap_left = WebcamGrabber(args.webcam_index, (pane_w, pane_h), flip=args.flip)
#         self.cap_right = IPWebcamGrabber(base, auth, (pane_w, pane_h), flip=args.flip, try_mjpeg=args.mjpeg)
#         self.proc_left = Processor("Proc-Left")
#         self.proc_right = Processor("Proc-Right")

#         for t in (self.cap_left, self.cap_right, self.proc_left, self.proc_right):
#             t.start()

#         # Timer to pull frames and refresh UI
#         self.fps_timer = QtCore.QTimer(self)
#         self.fps_timer.timeout.connect(self._tick)
#         self.fps_timer.start(1000 // max(1, args.ui_fps))

#         self._last_ts = time.time()
#         self.statusBar().setStyleSheet("color: #bdbdbd;")

#     def _tick(self):
#         # feed processors with latest frames
#         f_left = self.cap_left.get()
#         f_right = self.cap_right.get()
#         if f_left is not None: self.proc_left.set_input(f_left)
#         if f_right is not None: self.proc_right.set_input(f_right)

#         # read processed frames to draw
#         o_left = self.proc_left.get_output()
#         o_right = self.proc_right.get_output()
#         if o_left is not None: self.tile_left.set_frame(o_left)
#         if o_right is not None: self.tile_right.set_frame(o_right)

#         # status FPS
#         now = time.time()
#         fps = 1.0 / max(1e-6, (now - self._last_ts))
#         self._last_ts = now
#         self.statusBar().showMessage(f"{fps:4.1f} FPS  |  ESC=Quit")

#     def keyPressEvent(self, e: QtGui.QKeyEvent):
#         if e.key() == QtCore.Qt.Key_Escape:
#             self.close()
#         else:
#             super().keyPressEvent(e)

#     def closeEvent(self, e: QtGui.QCloseEvent):
#         # graceful shutdown
#         for t in (self.cap_left, self.cap_right, self.proc_left, self.proc_right):
#             t.stop()
#         for t in (self.cap_left, self.cap_right, self.proc_left, self.proc_right):
#             t.join(timeout=1.0)
#         return super().closeEvent(e)


# # ----------------------------- Argument parsing & app entry -----------------------------
# def parse_args():
#     import argparse
#     ap = argparse.ArgumentParser(description="Zoom-like dual camera UI (threaded & processed)")
#     ap.add_argument("--ip", required=True, help="Phone IP (e.g. 192.168.1.38)")
#     ap.add_argument("--port", type=int, default=8080)
#     ap.add_argument("--user", default=None)
#     ap.add_argument("--password", default=None)
#     ap.add_argument("--width", type=int, default=640)
#     ap.add_argument("--height", type=int, default=480)
#     ap.add_argument("--flip", action="store_true", help="mirror both tiles (selfie view)")
#     ap.add_argument("--mjpeg", action="store_true", help="try /video MJPEG first")
#     ap.add_argument("--webcam_index", type=int, default=0)
#     ap.add_argument("--ui_fps", type=int, default=30, help="UI refresh FPS")
#     return ap.parse_args()


# def main():
#     args = parse_args()
#     app = QtWidgets.QApplication(sys.argv)
#     # enable High-DPI crisp rendering
#     QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
#     win = MainWindow(args)
#     win.show()
#     sys.exit(app.exec_())


# if __name__ == "__main__":
#     main()


# # python virtualcam.py --ip 192.168.1.38 --flip




import sys, time, csv, copy, threading, json
from typing import Optional
import numpy as np
import cv2 as cv
import requests
from requests.auth import HTTPBasicAuth
from PyQt5 import QtCore, QtGui, QtWidgets

# ML bits
import mediapipe as mp
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, pre_process_landmark

# Offline STT
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ---------------- Helpers ----------------
def bgr_to_qimage(frame_bgr: np.ndarray) -> QtGui.QImage:
    h, w, _ = frame_bgr.shape
    rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)

class StoppableThread(threading.Thread):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._stop_event = threading.Event()
    def stop(self): self._stop_event.set()
    def stopped(self): return self._stop_event.is_set()

# --------------- Capture threads ---------------
class WebcamGrabber(StoppableThread):
    def __init__(self, device_index:int, size:tuple[int,int], flip:bool):
        super().__init__(daemon=True, name=f"Webcam-{device_index}")
        self.w, self.h = size; self.flip = flip; self.device_index = device_index
        self._lock = threading.Lock(); self.frame: Optional[np.ndarray] = None
    def run(self):
        cap = cv.VideoCapture(self.device_index)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.w); cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.h); cap.set(cv.CAP_PROP_FPS, 30)
        while not self.stopped():
            ok, f = cap.read()
            if not ok: time.sleep(0.01); continue
            if self.flip: f = cv.flip(f, 1)
            if (f.shape[1], f.shape[0]) != (self.w, self.h):
                f = cv.resize(f, (self.w, self.h), interpolation=cv.INTER_LINEAR)
            with self._lock: self.frame = f
        cap.release()
    def get(self):
        with self._lock: return None if self.frame is None else self.frame.copy()

class IPWebcamGrabber(StoppableThread):
    """Remote/IP camera: try /video MJPEG, fallback to /shot.jpg polling."""
    def __init__(self, base_url:str, auth:Optional[HTTPBasicAuth], size:tuple[int,int], flip:bool, try_mjpeg:bool):
        super().__init__(daemon=True, name="IPWebcam")
        self.base = base_url.rstrip("/"); self.auth = auth
        self.w, self.h = size; self.flip = flip; self.try_mjpeg = try_mjpeg
        self._lock = threading.Lock(); self.frame: Optional[np.ndarray] = None
    def _open_mjpeg(self) -> Optional[cv.VideoCapture]:
        cap = cv.VideoCapture(f"{self.base}/video", cv.CAP_FFMPEG)
        if cap.isOpened(): cap.set(cv.CAP_PROP_BUFFERSIZE, 2); return cap
        return None
    def _snapshot(self, s:requests.Session, timeout=2.0):
        r = s.get(f"{self.base}/shot.jpg", auth=self.auth, timeout=timeout)
        if r.status_code != 200: return None
        arr = np.frombuffer(r.content, np.uint8)
        return cv.imdecode(arr, cv.IMREAD_COLOR)
    def run(self):
        cap = self._open_mjpeg() if self.try_mjpeg else None
        if cap is not None and cap.isOpened():
            while not self.stopped():
                ok, f = cap.read()
                if not ok: cap.release(); cap=None; break
                if self.flip: f = cv.flip(f, 1)
                if (f.shape[1], f.shape[0]) != (self.w, self.h):
                    f = cv.resize(f, (self.w, self.h), interpolation=cv.INTER_LINEAR)
                with self._lock: self.frame = f
        s = requests.Session()
        while not self.stopped():
            try:
                f = self._snapshot(s)
                if f is not None:
                    if self.flip: f = cv.flip(f, 1)
                    if (f.shape[1], f.shape[0]) != (self.w, self.h):
                        f = cv.resize(f, (self.w, self.h), interpolation=cv.INTER_LINEAR)
                    with self._lock: self.frame = f
                else:
                    time.sleep(0.03)
            except Exception:
                time.sleep(0.05)
    def get(self):
        with self._lock: return None if self.frame is None else self.frame.copy()

# --------------- Processing (toggle hands per source) ---------------
class Processor(StoppableThread):
    """
    Per-source processor. If enable_hands=False, it passes frames through unmodified.
    """
    def __init__(self, name:str, enable_hands:bool=True, labels_path:str=None):
        super().__init__(daemon=True, name=name)
        self.enable_hands = enable_hands
        self._in_lock = threading.Lock(); self._out_lock = threading.Lock()
        self._in_frame: Optional[np.ndarray] = None; self._out_frame: Optional[np.ndarray] = None

        if self.enable_hands:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.7, min_tracking_confidence=0.5
            )
            self.classifier = KeyPointClassifier()
            if labels_path:
                with open(labels_path, encoding='utf-8-sig') as f:
                    self.labels = [row[0] for row in csv.reader(f)]
            else:
                with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
                    self.labels = [row[0] for row in csv.reader(f)]

    def set_input(self, img: Optional[np.ndarray]):
        if img is None: return
        with self._in_lock: self._in_frame = img.copy()

    def get_output(self) -> Optional[np.ndarray]:
        with self._out_lock: return None if self._out_frame is None else self._out_frame.copy()

    def _process(self, bgr: np.ndarray) -> np.ndarray:
        if not self.enable_hands:
            return bgr  # passthrough for remote
        debug = copy.deepcopy(bgr)
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB); rgb.flags.writeable = False
        results = self.hands.process(rgb); rgb.flags.writeable = True
        if results.multi_hand_landmarks:
            for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                lst = calc_landmark_list(debug, lm)
                pre = pre_process_landmark(lst)
                sign_id = self.classifier(pre)
                debug = draw_landmarks(debug, lst)
                debug = draw_info_text(debug, handed, self.labels[sign_id])
        return debug

    def run(self):
        while not self.stopped():
            with self._in_lock:
                src = None if self._in_frame is None else self._in_frame.copy()
                self._in_frame = None
            if src is None: time.sleep(0.002); continue
            out = self._process(src)
            with self._out_lock: self._out_frame = out

# --------------- STT (Vosk) ---------------
class SpeechToTextThread(StoppableThread):
    def __init__(self, model_path:str, mic_index:Optional[int]=None, samplerate:int=16000):
        super().__init__(daemon=True, name="STT")
        self.model_path = model_path; self.mic_index = mic_index; self.samplerate = samplerate
        self.text_lock = threading.Lock()
        self.partial_text = ""; self.final_text = ""
    def run(self):
        model = Model(self.model_path)
        rec = KaldiRecognizer(model, self.samplerate); rec.SetWords(True)
        def cb(indata, frames, time_info, status):
            if self.stopped(): raise sd.CallbackStop()
            data = indata.tobytes()
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result()); txt = (res.get("text") or "").strip()
                if txt:
                    with self.text_lock: self.final_text, self.partial_text = txt, ""
            else:
                res = json.loads(rec.PartialResult()); part = (res.get("partial") or "").strip()
                with self.text_lock: self.partial_text = part
        try:
            with sd.InputStream(device=self.mic_index, channels=1, samplerate=self.samplerate,
                                dtype='int16', blocksize=int(self.samplerate*0.25), callback=cb):
                while not self.stopped(): time.sleep(0.05)
        except Exception as e:
            with self.text_lock: self.final_text = f"[STT error: {e}]"; self.partial_text = ""
    def get_caption(self) -> str:
        with self.text_lock:
            return self.partial_text or self.final_text

# --------------- UI tiles ---------------
class VideoTile(QtWidgets.QFrame):
    def __init__(self, title:str, sub_pos:str="bottom", show_sub:bool=True):
        super().__init__()
        self.setObjectName("videoTile")
        self.title = title; self._qimage: Optional[QtGui.QImage] = None
        self.subtitle: str = ""; self.sub_pos = sub_pos; self.show_sub = show_sub
        self.setMinimumSize(480, 270)
        self.setStyleSheet("""QFrame#videoTile { background-color:#111; border-radius:18px; }""")
        eff = QtWidgets.QGraphicsDropShadowEffect(self); eff.setBlurRadius(28); eff.setOffset(0,6); eff.setColor(QtGui.QColor(0,0,0,160))
        self.setGraphicsEffect(eff)
    def set_frame(self, bgr: np.ndarray):
        self._qimage = bgr_to_qimage(bgr); self.update()
    def set_subtitle(self, text:str):
        self.subtitle = text; self.update()
    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        path = QtGui.QPainterPath(); path.addRoundedRect(QtCore.QRectF(self.rect()), 18.0, 18.0); p.setClipPath(path)
        if self._qimage:
            scaled = self._qimage.scaled(self.size(), QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
            x = (scaled.width()-self.width())//2; y = (scaled.height()-self.height())//2
            p.drawImage(QtCore.QRect(0,0,self.width(),self.height()), scaled, QtCore.QRect(x,y,self.width(),self.height()))
        else:
            p.fillRect(self.rect(), QtGui.QColor("#222"))
        # title
        r = QtCore.QRect(12,12,260,28)
        p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(0,0,0,150)); p.drawRoundedRect(r,12,12)
        p.setPen(QtGui.QPen(QtGui.QColor("#fff"))); f = p.font(); f.setPointSize(11); f.setBold(True); p.setFont(f)
        p.drawText(r.adjusted(10,0,-10,0), QtCore.Qt.AlignVCenter|QtCore.Qt.AlignLeft, self.title)
        # subtitles
        if self.show_sub and self.subtitle:
            pad = 12; bar_h = 44
            sub_rect = (QtCore.QRect(pad, pad+36, self.width()-2*pad, bar_h)
                        if self.sub_pos=="top" else
                        QtCore.QRect(pad, self.height()-bar_h-pad, self.width()-2*pad, bar_h))
            p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(0,0,0,160)); p.drawRoundedRect(sub_rect,12,12)
            p.setPen(QtGui.QPen(QtGui.QColor("#fff"))); f2 = p.font(); f2.setPointSize(12); f2.setBold(True); p.setFont(f2)
            metrics = QtGui.QFontMetrics(f2)
            text = metrics.elidedText(self.subtitle, QtCore.Qt.ElideRight, sub_rect.width()-20)
            p.drawText(sub_rect.adjusted(10,0,-10,0), QtCore.Qt.AlignVCenter|QtCore.Qt.AlignLeft, text)

class ControlBar(QtWidgets.QFrame):
    endClicked = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QFrame{background:#0b0b0c;border-top-left-radius:16px;border-top-right-radius:16px;}
            QPushButton{color:#e8e8e8;background:#1a1a1c;padding:10px 16px;border-radius:10px;}
            QPushButton:hover{background:#232326;}
            QPushButton#danger{background:#e53935;color:white;}
            QPushButton#danger:hover{background:#cf2f2b;}
        """)
        h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(16,10,16,10); h.setSpacing(12)
        self.btn_end = QtWidgets.QPushButton("End"); self.btn_end.setObjectName("danger")
        h.addStretch(1); h.addWidget(self.btn_end); self.btn_end.clicked.connect(self.endClicked.emit)

# --------------- Main Window ---------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Webcam (Landmarks) | Remote (Subtitles)")
        self.setStyleSheet("QMainWindow{background:#101014;}"); self.resize(1280,720)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central); v.setContentsMargins(18,18,18,12); v.setSpacing(12)
        h = QtWidgets.QHBoxLayout(); h.setSpacing(18)

        show_left = args.sub_target in ("both","left")
        show_right = args.sub_target in ("both","right")

        self.tile_left  = VideoTile("WEBCAM  (Landmarks)", sub_pos=args.sub_pos, show_sub=show_left)
        self.tile_right = VideoTile("REMOTE  (Subtitles)", sub_pos=args.sub_pos, show_sub=show_right)
        h.addWidget(self.tile_left,1); h.addWidget(self.tile_right,1)

        self.controls = ControlBar(); v.addLayout(h,1); v.addWidget(self.controls,0)
        self.controls.endClicked.connect(self.close)

        # capture
        size = (args.width, args.height)
        base = f"http://{args.ip}:{args.port}"
        auth = HTTPBasicAuth(args.user, args.password) if args.user and args.password else None
        self.cap_left  = WebcamGrabber(args.webcam_index, size, flip=args.flip)
        self.cap_right = IPWebcamGrabber(base, auth, size, flip=args.flip, try_mjpeg=args.mjpeg)

        # processors: left = hands ON, right = hands OFF (passthrough)
        self.proc_left  = Processor("Proc-Left",  enable_hands=True)
        self.proc_right = Processor("Proc-Right", enable_hands=False)

        for t in (self.cap_left, self.cap_right, self.proc_left, self.proc_right): t.start()

        # STT thread (captions will be shown on selected tile(s); default right)
        self.stt = SpeechToTextThread(model_path=args.vosk_model, mic_index=args.mic_index, samplerate=16000)
        self.stt.start()

        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._tick)
        self.timer.start(1000 // max(1, args.ui_fps))
        self._last_ts = time.time(); self.statusBar().setStyleSheet("color:#bdbdbd;")
        self.sub_target = args.sub_target

    def _tick(self):
        # feed processors
        fL = self.cap_left.get(); fR = self.cap_right.get()
        if fL is not None: self.proc_left.set_input(fL)
        if fR is not None: self.proc_right.set_input(fR)

        # draw
        oL = self.proc_left.get_output(); oR = self.proc_right.get_output()
        if oL is not None: self.tile_left.set_frame(oL)
        if oR is not None: self.tile_right.set_frame(oR)

        # captions (local mic → overlay)
        caption = self.stt.get_caption()
        if self.sub_target in ("both","left"):  self.tile_left.set_subtitle(caption)
        if self.sub_target in ("both","right"): self.tile_right.set_subtitle(caption)

        # FPS
        now = time.time(); fps = 1.0 / max(1e-6, (now - self._last_ts)); self._last_ts = now
        self.statusBar().showMessage(f"{fps:4.1f} FPS  |  ESC=Quit")

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == QtCore.Qt.Key_Escape: self.close()
        else: super().keyPressEvent(e)

    def closeEvent(self, e: QtGui.QCloseEvent):
        for t in (self.cap_left, self.cap_right, self.proc_left, self.proc_right, self.stt): t.stop()
        for t in (self.cap_left, self.cap_right, self.proc_left, self.proc_right, self.stt): t.join(timeout=1.0)
        return super().closeEvent(e)

# --------------- Args / Entry ---------------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Left: landmarks | Right: subtitles")
    ap.add_argument("--ip", required=True, help="Remote/IP camera host (e.g., 10.220.112.240)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--user", default=None); ap.add_argument("--password", default=None)
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--flip", action="store_true"); ap.add_argument("--mjpeg", action="store_true")
    ap.add_argument("--webcam_index", type=int, default=0)
    ap.add_argument("--ui_fps", type=int, default=30)

    # STT/subtitles
    ap.add_argument("--vosk_model", required=True, help="Path to Vosk model folder")
    ap.add_argument("--mic_index", type=int, default=None, help="Microphone index (optional)")
    ap.add_argument("--sub_pos", choices=["top","bottom"], default="bottom")
    ap.add_argument("--sub_target", choices=["both","left","right"], default="right")
    return ap.parse_args()

def main():
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    win = MainWindow(args); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
