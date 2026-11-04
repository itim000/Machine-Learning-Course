import cv2
import face_recognition
import threading
import time
import os
import re
import csv
import numpy as np
from datetime import datetime

URL = "http://192.168.1.38:8080/video"
KNOWN_DIR = "known_faces"
SCALE = 0.25
TOLERANCE = 0.48    # ยิ่งน้อยยิ่งเข้มงวด
MARGIN = 0.05       # คนอันดับ 1 ต้องห่างจากคนอันดับ 2 อย่างน้อยเท่านี้ ไม่งั้นถือว่า Unknown
INTERVAL = 0.1      # ตรวจจับ/จำหน้าอย่างมากทุก 0.1 วินาที (~10 ครั้ง/วินาที)

ATTENDANCE_FILE = "attendance.csv"
CONFIRM = 10        # ต้องเจอชื่อเดิม 10 รอบ (~1 วินาที) ถึงจะบันทึก
MISS_LIMIT = 5      # หายจากภาพเกิน 5 รอบติด ถึงจะเริ่มนับใหม่ (กันกะพริบตาแล้วหลุด)

# --- Liveness (ตรวจว่าเป็นคนจริงด้วยการกะพริบตา) ---
EAR_THRESH = 0.21       # ค่า EAR ต่ำกว่านี้ = ตาปิด
BLINKS_REQUIRED = 2     # ต้องกะพริบกี่ครั้งถึงจะผ่าน
BLINK_WINDOW = 10       # นับเฉพาะการกะพริบภายในกี่วินาทีล่าสุด


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


def load_today():
    """อ่านไฟล์เดิม เพื่อรู้ว่าวันนี้ใครเช็กไปแล้ว"""
    today = datetime.now().strftime("%Y-%m-%d")
    marked = set()
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["date"] == today:
                    marked.add(row["name"])
    return marked


def mark(name):
    """เพิ่มหนึ่งแถว: ชื่อ, วันที่, เวลา"""
    now = datetime.now()
    new_file = not os.path.exists(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["name", "date", "time"])
        w.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])


# ---------------------------------------------------------------
# Liveness: Eye Aspect Ratio (EAR) + ตัวนับการกะพริบตา
# ---------------------------------------------------------------
def eye_aspect_ratio(eye):
    """eye = 6 จุดของดวงตาจาก face_landmarks"""
    p = np.array(eye, dtype=float)
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return (a + b) / (2.0 * c)


def get_ear(frame, loc, k):
    """คำนวณ EAR จากภาพความละเอียดเต็ม
    (ภาพที่ย่อเหลือ 0.25 เล็กเกินไป ดวงตาจะไม่ชัดพอ)
    loc อยู่ในพิกัดภาพย่อ จึงคูณ k เพื่อแปลงกลับเป็นภาพเต็ม"""
    top, right, bottom, left = [v * k for v in loc]
    h, w = frame.shape[:2]
    pad = int((bottom - top) * 0.3)
    t, b = max(0, top - pad), min(h, bottom + pad)
    l, r = max(0, left - pad), min(w, right + pad)
    if b - t < 20 or r - l < 20:
        return None
    crop = cv2.cvtColor(frame[t:b, l:r], cv2.COLOR_BGR2RGB)
    box = [(top - t, right - l, bottom - t, left - l)]
    lms = face_recognition.face_landmarks(crop, box)
    if not lms:
        return None
    e = lms[0]
    return (eye_aspect_ratio(e["left_eye"]) + eye_aspect_ratio(e["right_eye"])) / 2


class BlinkTracker:
    """นับการกะพริบตาของคนหนึ่งคน: ตาปิด แล้วเปิดกลับ = 1 ครั้ง"""

    def __init__(self):
        self.closed = False
        self.blinks = []

    def update(self, ear):
        now = time.time()
        if ear < EAR_THRESH:
            self.closed = True
        elif self.closed:
            self.closed = False
            self.blinks.append(now)
        self.blinks = [t for t in self.blinks if now - t <= BLINK_WINDOW]

    def count(self):
        return len(self.blinks)

    def live(self):
        return len(self.blinks) >= BLINKS_REQUIRED


class Recognizer:
    """หาหน้า + จำหน้า + ตรวจกะพริบตา + บันทึกเข้าเรียน ใน thread แยก"""

    def __init__(self, cam, known_encodings, known_names):
        self.cam = cam
        self.known_encodings = known_encodings
        self.known_names = known_names
        # [(พิกัดหน้า, ชื่อ, ระยะห่าง, เช็กแล้วหรือยัง, จำนวนกะพริบ, ผ่าน liveness หรือยัง)]
        self.results = []
        self.lock = threading.Lock()
        self.marked_today = load_today()
        self.streak = {}
        self.missing = {}
        self.trackers = {}
        self.running = True
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _match(self, enc):
        """เทียบกับทุกรูปที่รู้จัก เลือกระยะที่ใกล้สุดของแต่ละคน
        แล้วเช็กว่าผ่าน TOLERANCE และห่างจากคนอันดับ 2 พอ (MARGIN)"""
        dists = face_recognition.face_distance(self.known_encodings, enc)
        best_per_name = {}
        for d, n in zip(dists, self.known_names):
            if n not in best_per_name or d < best_per_name[n]:
                best_per_name[n] = float(d)

        ranked = sorted(best_per_name.items(), key=lambda x: x[1])
        best_name, best_d = ranked[0]
        if best_d >= TOLERANCE:
            return "Unknown", best_d
        if len(ranked) > 1 and ranked[1][1] - best_d < MARGIN:
            return "Unknown", best_d
        return best_name, best_d

    def _work(self):
        k = int(1 / SCALE)
        while self.running:
            start = time.time()
            frame = self.cam.read()
            if frame is None:
                time.sleep(0.05)
                continue

            small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            found = []
            for loc, enc in zip(locations, encodings):
                name, dist = self._match(enc)
                found.append((loc, name, dist))

                if name != "Unknown":
                    ear = get_ear(frame, loc, k)
                    if ear is not None:
                        self.trackers.setdefault(name, BlinkTracker()).update(ear)

            self._update_attendance({n for _, n, _ in found if n != "Unknown"})

            results = []
            for loc, n, d in found:
                tr = self.trackers.get(n)
                results.append((
                    loc, n, d,
                    n in self.marked_today,
                    tr.count() if tr else 0,
                    tr.live() if tr else False,
                ))
            with self.lock:
                self.results = results

            wait = INTERVAL - (time.time() - start)
            if wait > 0:
                time.sleep(wait)   # พักให้ thread แสดงภาพได้ทำงานลื่น

    def _update_attendance(self, names_now):
        for n in names_now:
            self.missing[n] = 0
            self.streak[n] = self.streak.get(n, 0) + 1
            tr = self.trackers.get(n)
            is_live = tr is not None and tr.live()
            # ต้องทั้ง "เจอชื่อเดิมติดกัน" และ "กะพริบตาครบ" ถึงจะบันทึก
            if self.streak[n] >= CONFIRM and is_live and n not in self.marked_today:
                mark(n)
                self.marked_today.add(n)
                print("เช็กชื่อแล้ว:", n)

        for n in list(self.streak):
            if n not in names_now:
                self.missing[n] = self.missing.get(n, 0) + 1
                if self.missing[n] > MISS_LIMIT:
                    self.streak[n] = 0               # หายไปนานพอ เริ่มนับใหม่
                    self.trackers.pop(n, None)       # ต้องกะพริบตาใหม่อีกครั้ง

    def get(self):
        with self.lock:
            return list(self.results)

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


def load_known(folder):
    """โหลดรูปคนที่รู้จัก รองรับ 3 รูปแบบ ใช้ปนกันได้
       1) known_faces/Beam.jpg           -> ชื่อ Beam
       2) known_faces/Beam_1.jpg, Beam_2.jpg -> ชื่อ Beam (หลายรูปต่อคน)
       3) known_faces/Beam/xxx.jpg       -> ชื่อ Beam (โฟลเดอร์ต่อคน)"""
    encodings, names = [], []

    def add(path, name):
        img = face_recognition.load_image_file(path)
        # num_jitters + model="large" ช้าลงตอนโหลด แต่ encoding เสถียรขึ้น
        encs = face_recognition.face_encodings(img, num_jitters=5, model="large")
        if encs:
            encodings.append(encs[0])
            names.append(name)
        else:
            print("ไม่พบใบหน้าในรูป:", path)

    exts = (".jpg", ".jpeg", ".png")
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        if os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                if f.lower().endswith(exts):
                    add(os.path.join(path, f), entry)
        elif entry.lower().endswith(exts):
            stem = os.path.splitext(entry)[0]
            name = re.sub(r"_\d+$", "", stem)   # ตัด _1, _2 ท้ายชื่อออก
            add(path, name)

    return encodings, names


# 1) โหลดรูปคนที่รู้จัก แล้วแปลงเป็น encoding
known_encodings, known_names = load_known(KNOWN_DIR)

if not known_encodings:
    raise SystemExit("ไม่มีรูปในโฟลเดอร์ known_faces")
print("โหลดแล้ว:", sorted(set(known_names)), f"({len(known_names)} รูป)")

# 2) เริ่มกล้อง + thread จำหน้า
cam = LatestFrame(URL)
rec = Recognizer(cam, known_encodings, known_names)

k = int(1 / SCALE)
last_frame = None

# 3) ลูปหลักทำแค่ "แสดงภาพ + วาดกรอบ" จึงลื่นเท่าที่กล้องส่งมา
while True:
    frame = cam.read()
    if frame is not None and frame is not last_frame:
        last_frame = frame
        display = frame.copy()   # วาดบนสำเนา ไม่ไปแตะภาพที่ thread อื่นกำลังใช้

        for (top, right, bottom, left), name, dist, done, blinks, live in rec.get():
            if name == "Unknown":
                color = (0, 0, 255)                  # แดง
                text = f"Unknown {dist:.2f}"
            elif done:
                color = (0, 255, 0)                  # เขียว
                text = f"{name} {dist:.2f} [OK]"
            elif live:
                color = (0, 255, 0)
                text = f"{name} {dist:.2f} LIVE"
            else:
                color = (0, 200, 255)                # เหลือง = รอกะพริบตา
                text = f"{name} - BLINK {blinks}/{BLINKS_REQUIRED}"

            cv2.rectangle(display, (left * k, top * k), (right * k, bottom * k), color, 2)
            cv2.putText(display, text, (left * k, top * k - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("recognize", display)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

rec.stop()
cam.stop()
cv2.destroyAllWindows()