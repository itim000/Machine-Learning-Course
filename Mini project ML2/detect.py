import cv2
import face_recognition
import threading

URL = "http://192.168.1.38:8080/video"


class LatestFrame:
    """อ่านภาพจากกล้องใน thread แยก และเก็บเฉพาะภาพล่าสุด"""

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


cam = LatestFrame(URL)

SCALE = 0.25   # ย่อภาพก่อนตรวจจับ
SKIP = 3       # ตรวจจับทุก 3 เฟรม
count = 0
locations = []

while True:
    frame = cam.read()
    if frame is None:
        continue

    count += 1
    if count % SKIP == 0:
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)

    k = int(1 / SCALE)
    for (top, right, bottom, left) in locations:
        cv2.rectangle(frame, (left * k, top * k), (right * k, bottom * k), (0, 255, 0), 2)

    cv2.imshow("detect", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.stop()
cv2.destroyAllWindows()