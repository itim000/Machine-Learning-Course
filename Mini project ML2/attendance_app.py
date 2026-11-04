"""
ระบบเช็กชื่อเข้าเรียนด้วยใบหน้า (Face Attendance)

ติดตั้งไลบรารีที่ต้องใช้เพิ่ม:  pip install customtkinter
รัน:                            python attendance_app.py

กติกาเวลา (ช่วงทดลอง ยังไม่อิงตารางเรียนจริง) ปรับได้ที่ CLASS_START / LATE_MINUTES / CLASS_MINUTES
  - มาก่อนหรือไม่เกิน 15 นาทีหลังเริ่ม (09:00-09:15)  -> ตรงเวลา
  - มาหลัง 15 นาที แต่ยังไม่ถึง 3 ชม. (09:15-12:00)   -> สาย
  - ถึง 12:00 แล้วยังไม่มา                            -> ขาด (บันทึกลงไฟล์ให้อัตโนมัติ)

กันการเอารูปบนมือถือมาหลอก (Liveness)
  จำหน้าได้แล้ว ระบบจะ "คาดเดา" ก่อนว่าเป็นคนจริงหรือภาพบนจอ/กระดาษ
    - ดูเหมือนคนจริง  -> บันทึกได้เลย
    - ดูน่าสงสัย      -> ต้องกระพริบตา 1 ครั้งก่อน (รูปนิ่งกระพริบตาไม่ได้ จึงไม่ผ่าน)
  ปรับได้ที่ AUTO_TRUST / SPOOF_THRESHOLD / LIVENESS

AI ตรวจว่าเป็นคนจริงไหม (Anti-Spoofing)
  ใช้โมเดล MiniFASNet (Silent-Face-Anti-Spoofing) ที่ฝึกมาจากภาพคนจริงเทียบกับภาพที่ถ่ายซ้ำจากจอ/กระดาษ
  ให้คะแนน "เป็นคนจริง" 0-1 ต่อใบหน้า แล้วสรุปจาก CONFIRM รอบล่าสุด
    - คะแนนสูง              -> ผ่านตามปกติ
    - คะแนนต่ำ              -> น่าสงสัย ต้องกระพริบตาก่อน
    - คะแนนต่ำมาก           -> ถือว่าเป็นภาพบนจอ/กระดาษ ไม่ให้เช็กชื่อ (ไม่เปิดให้กระพริบตา)
  ต้องติดตั้งเพิ่ม:  pip install onnxruntime   (ไฟล์โมเดล ~3.5 MB จะโหลดเองครั้งแรกลงโฟลเดอร์ models/)
  ปรับได้ที่ AI_LIVENESS / AI_REAL_MIN / AI_BLOCK_BELOW  (เปิด DEBUG_LIVENESS เพื่อดูคะแนนบนหน้าจอ)
"""
import os
import csv
import time
import queue
import statistics
import shutil
import hashlib
import importlib.util
import urllib.request
import threading
import multiprocessing as mp
import warnings
import traceback
from collections import deque
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse

warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import cv2
import numpy as np
import face_recognition
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageOps, ImageTk

try:
    import winsound
except ImportError:          # ไม่ใช่ Windows
    winsound = None

# ───────────────────────── ค่าตั้งต้น ─────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_DIR = os.path.join(BASE_DIR, "known_faces")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
BACKUP_DIR = os.path.join(BASE_DIR, "backup")
IMAGE_EXT = (".jpg", ".jpeg", ".png")

DEFAULT_SOURCE = "0"   # 0 = กล้องโน้ตบุ๊ก (ถ้ามีหลายกล้องลอง 1, 2 / หรือใส่ URL ของ IP Webcam เช่น http://192.168.1.38:8080/video)
SEND_WIDTH = 640        # ส่งภาพให้ process จำหน้ากว้างเท่านี้ (พิกเซล) ใหญ่ = แม่นขึ้นแต่ช้าลง
DETECT_SCALE = 0.5      # หาตำแหน่งหน้าบนภาพที่ย่อลงอีก (เร็วขึ้น) ส่วนการจำหน้า/จับจุดบนใบหน้าใช้ภาพเต็ม SEND_WIDTH
INTERVAL = 0.2          # จำหน้าอย่างมากทุก 0.2 วินาที
DEFAULT_TOLERANCE = 0.50
CONFIRM = 5             # ต้องเจอชื่อเดิมติดกันกี่รอบ ถึงจะเริ่มตรวจว่าเป็นคนจริง

# ── กันรูปบนมือถือมาหลอก (Liveness): ต้องกระพริบตา ──
# จำหน้าได้ติดกัน CONFIRM รอบ -> ระบบคาดเดาว่าเป็นคนจริงไหม
#   ดูเหมือนคนจริง -> บันทึกเลย
#   ดูน่าสงสัย     -> รอดูการกระพริบตา 1 ครั้ง -> จำหน้าได้อีกครั้ง -> ถึงจะบันทึก
LIVENESS = True             # False = ปิดระบบตรวจคนจริงทั้งหมด (จำหน้าได้ก็บันทึกเลย)
AUTO_TRUST = True           # True = ดูเหมือนคนจริงไม่ต้องกระพริบตา / False = ทุกคนต้องกระพริบตา
SPOOF_THRESHOLD = 0.40      # คะแนนความน่าสงสัย (0-1) ตั้งแต่เท่านี้ขึ้นไปต้องกระพริบตา (ต่ำ = เข้มงวดขึ้น)
BRIGHT_RATIO = 1.6          # หน้าสว่างกว่าฉากรอบ ๆ เกินกี่เท่า ถือว่าน่าสงสัย (จอเรืองแสง)
GLARE_FRAC = 0.04           # สัดส่วนจุดแสงจ้าบนใบหน้าที่เกินนี้ ถือว่าน่าสงสัย (แสงสะท้อนจอ/กระดาษมัน)
BLINK_RATIO = 0.70          # ตา "หลับ" = EAR ต่ำกว่า 70% ของตอนลืมตา (เล็ก = ต้องหลับตาแน่นขึ้น)
BLINK_MIN_CLOSED = 1        # ต้องเห็นตาหลับกี่ครั้งติดกัน (ถ้าผ่านเองทั้งที่ไม่ได้กระพริบ ให้เพิ่มเป็น 2)
BLINK_MAX_SEC = 1.0         # หลับตานานกว่านี้ไม่นับว่ากระพริบ
BLINK_MIN_BASE = 0.16       # ตอนลืมตา EAR ต้องไม่ต่ำกว่านี้ (กันภาพที่หลับตาค้าง/ตาเล็กเกินจับไม่ได้)
BLINK_MAX_YAW = 0.20        # หันหน้าเกินนี้ไม่ใช้ค่า (ตามองไม่ชัด)
BASELINE_N = 6              # ต้องเห็นตาลืมกี่ครั้ง ก่อนเริ่มจับการกระพริบ
MIN_EYE_PX = 8              # ตากว้างน้อยกว่านี้ (พิกเซล) ถือว่าไกลเกินไป อ่านค่าไม่ได้
FULL_EVERY = 0.8            # ระหว่างรอกระพริบ จำหน้าเต็มรูปแบบทุกกี่วินาที (รอบอื่นติดตามเฉพาะตา เร็วกว่า)
MAX_BAD = 3                 # จำหน้าเป็นคนอื่น/ไม่รู้จักติดกันเกินนี้ ให้ยกเลิก
LIVENESS_TIMEOUT = 15.0     # รอกระพริบไม่สำเร็จภายในกี่วินาที ให้เริ่มใหม่
LIVENESS_GRACE = 2.0        # ใบหน้าหายไปนานกว่านี้ให้ยกเลิก
FINAL_FRAMES = 1            # กระพริบแล้วต้องจำหน้าได้อีกกี่รอบ (ยืนยันว่ายังเป็นคนเดิม)
DEBUG_LIVENESS = False      # True = แสดงคะแนนน่าสงสัย (sus) / AI real / yaw / ear ใต้กรอบหน้า ไว้ปรับค่าต่าง ๆ

# ── AI ตรวจว่าเป็นคนจริงไหม (โมเดล MiniFASNet) ──
AI_LIVENESS = True          # True = ใช้โมเดล AI ช่วยตัดสิน / False = ปิด (ใช้แค่การตรวจจากภาพ + กระพริบตา)
AI_REAL_MIN = 0.80          # ค่ากลางความน่าจะเป็น "เป็นคนจริง" ของ CONFIRM รอบล่าสุด ต่ำกว่านี้ = น่าสงสัย ต้องกระพริบตา
AI_BLOCK_BELOW = 0.15       # ต่ำกว่านี้ = มั่นใจว่าเป็นภาพบนจอ/กระดาษ ไม่ให้ผ่านเลย (ตั้ง 0 = ไม่บล็อก ให้กระพริบตาแทน)
FAKE_PROMPT = "ตรวจพบภาพบนจอ/กระดาษ ไม่สามารถเช็กชื่อได้"

# ── ขนาดใบหน้าขั้นต่ำ: กันภาพบนจอที่ถือไกลจนตรวจกรอบจอ/ลายจอไม่ได้ ──
# ใบหน้าต้องกว้างอย่างน้อยเท่านี้ (พิกเซลบนภาพที่ส่งไปจำ ซึ่งกว้างไม่เกิน SEND_WIDTH) ถึงจะเริ่มเช็กชื่อได้
# เล็กกว่านี้ = ขึ้นข้อความให้เข้าใกล้กล้อง (คนยืนห่างเกินไปก็โดนด้วย ถ้าเช็กยากให้ลดค่า / 0 = ปิดเงื่อนไขนี้)
MIN_FACE_PX = 110
FAR_PROMPT = "เข้ามาใกล้กล้องอีกนิด"
MODEL_DIR = os.path.join(BASE_DIR, "models")
ANTISPOOF_URL = "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/"
ANTISPOOF_MODELS = (("MiniFASNetV2.onnx", 2.7), ("MiniFASNetV1SE.onnx", 4.0))   # (ไฟล์, สัดส่วนขยายกรอบหน้า)
ANTISPOOF_SHA256 = {
    "MiniFASNetV2.onnx": "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
    "MiniFASNetV1SE.onnx": "ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676",
}

# ── เวลาเรียน (ช่วงทดลอง) ──
# ตอนทดลองอยากให้เห็นผลเร็ว ให้ตั้ง CLASS_START เป็นเวลาปัจจุบัน และลด LATE_MINUTES / CLASS_MINUTES
# เช่น CLASS_START = "14:30", LATE_MINUTES = 1, CLASS_MINUTES = 3
CLASS_START = "22:40"   # เวลาเริ่มเรียน (HH:MM)
LATE_MINUTES = 15       # มาช้ากว่านี้ถือว่า "สาย"กฟกฟก
CLASS_MINUTES = 180     # ช่วงเช็กชื่อทั้งหมด 3 ชม. (09:00 -> 12:00) หลังจากนั้นคนที่ไม่มาถือว่า "ขาด"

ST_ONTIME = "ตรงเวลา"
ST_LATE = "สาย"
ST_ABSENT = "ขาด"
CSV_HEADER = ["name", "date", "time", "status"]

# ───────────────────────── สี / ฟอนต์ ─────────────────────────
BG = "#0b1220"
PANEL = "#111a2e"
CARD = "#17233d"
ROW = "#1d2b4a"
VIDEO_BG = "#050912"
ACCENT = "#3b82f6"
ACCENT_HOVER = "#2563eb"
GREEN = "#22c55e"
AMBER = "#f59e0b"
RED = "#ef4444"
PURPLE = "#8b5cf6"      # ขาดเรียน
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
FONT = "Leelawadee UI"

STATUS = {
    "idle": ("กล้องปิดอยู่", MUTED),
    "connecting": ("กำลังเชื่อมต่อกล้อง...", AMBER),
    "live": ("กล้องทำงาน", GREEN),
    "lost": ("สัญญาณกล้องขาดหาย", RED),
}

ctk.set_appearance_mode("dark")

# ───────────────────────── ฟอนต์สำหรับวาดบนภาพ (รองรับภาษาไทย) ─────────────────────────
_font_cache = {}


def pil_font(size, bold=False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    names = ["LeelaUIb.ttf", "tahomabd.ttf", "segoeuib.ttf"] if bold \
        else ["LeelawUI.ttf", "tahoma.ttf", "segoeui.ttf"]
    font = None
    for n in names:
        try:
            font = ImageFont.truetype(os.path.join(fonts_dir, n), size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    _font_cache[key] = font
    return font


# ───────────────────────── ช่วงเวลาเรียน ─────────────────────────
def session_times(now=None):
    """คืนค่า (เวลาเริ่ม, เวลาที่เริ่มนับสาย, เวลาปิดรับเช็กชื่อ) ของวันที่ now"""
    now = now or datetime.now()
    h, m = (int(x) for x in CLASS_START.split(":"))
    start = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return start, start + timedelta(minutes=LATE_MINUTES), start + timedelta(minutes=CLASS_MINUTES)


def session_phase(now=None):
    """'ontime' = ยังไม่เกินเวลาสาย, 'late' = เลยเวลาสายแต่ยังเช็กชื่อได้, 'closed' = หมดเวลา (ขาด)"""
    now = now or datetime.now()
    _, late_at, end = session_times(now)
    if now <= late_at:
        return "ontime"
    if now < end:
        return "late"
    return "closed"


# ───────────────────────── ไฟล์เช็กชื่อ ─────────────────────────
def load_today():
    """อ่าน attendance.csv คืนค่า {ชื่อ: (เวลา, สถานะ)} ของวันนี้"""
    today = datetime.now().strftime("%Y-%m-%d")
    marked = {}
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("date") == today and row.get("name"):
                    marked[row["name"]] = (row.get("time", ""), row.get("status") or ST_ONTIME)
    return marked


def mark(name, status):
    """เพิ่มหนึ่งแถว (ชื่อ, วันที่, เวลา, สถานะ) คืนค่าเวลา ถ้าไฟล์ถูกเปิดค้างจะเกิด PermissionError"""
    now = datetime.now()
    new_file = (not os.path.exists(ATTENDANCE_FILE)) or os.path.getsize(ATTENDANCE_FILE) == 0
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), status])
    return now.strftime("%H:%M:%S")


def mark_absent(names):
    """บันทึกชื่อทั้งหมดใน names เป็น 'ขาด' (เวลาที่ลงคือเวลาปิดรับเช็กชื่อ) คืนค่าเวลานั้น"""
    end = session_times()[2]
    t = end.strftime("%H:%M:%S")
    if not names:
        return t
    date = datetime.now().strftime("%Y-%m-%d")
    new_file = (not os.path.exists(ATTENDANCE_FILE)) or os.path.getsize(ATTENDANCE_FILE) == 0
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        for n in names:
            w.writerow([n, date, t, ST_ABSENT])
    return t


def count_records(scope):
    """นับจำนวนรายการในไฟล์ scope = 'today' หรือ 'all'"""
    today = datetime.now().strftime("%Y-%m-%d")
    n = 0
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if scope == "all" or row.get("date") == today:
                    n += 1
    return n


def backup_attendance():
    """สำรองไฟล์เช็กชื่อไว้ในโฟลเดอร์ backup คืนค่า path ของไฟล์สำรอง (หรือ None ถ้าไม่มีไฟล์)"""
    if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dst = os.path.join(BACKUP_DIR, "attendance_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
    shutil.copy2(ATTENDANCE_FILE, dst)
    return dst


def migrate_csv():
    """ไฟล์เวอร์ชันเก่ามีแค่ name,date,time -> เติมคอลัมน์ status ให้ (แถวเก่าถือว่า 'ตรงเวลา') และสำรองไฟล์เดิมก่อน"""
    if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
        return
    with open(ATTENDANCE_FILE, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "status" in (reader.fieldnames or []):
            return
        rows = list(reader)
    backup_attendance()
    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow([r.get("name", ""), r.get("date", ""), r.get("time", ""), ST_ONTIME])


def reset_records(scope):
    """
    รีเซ็ตการเข้าเรียน (สำรองไฟล์เดิมก่อนเสมอ)
      scope = 'today' : ลบเฉพาะแถวของวันนี้ ประวัติวันอื่นยังอยู่
      scope = 'all'   : ล้างประวัติทั้งหมด เหลือแต่บรรทัดหัวตาราง
    ถ้าไฟล์ถูกเปิดค้างใน Excel จะเกิด PermissionError
    """
    backup = backup_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    keep = []
    if scope == "today" and os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("date") != today:
                    keep.append([row.get("name", ""), row.get("date", ""), row.get("time", ""),
                                 row.get("status") or ST_ONTIME])
    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(keep)
    return backup


# ───────────────────────── กล้อง ─────────────────────────
def parse_source(text):
    """รับข้อความจากช่องกรอก แล้วแปลงเป็นแหล่งภาพที่ OpenCV ใช้ได้"""
    text = text.strip()
    if text.isdigit():
        return int(text)
    if "://" not in text:
        text = "http://" + text
    u = urlparse(text)
    if u.scheme in ("http", "https") and u.path in ("", "/"):
        u = u._replace(path="/video")      # หน้าแรกของ IP Webcam ไม่ใช่สตรีม ต้องต่อ /video
    return urlunparse(u)


class Camera:
    """อ่านภาพใน thread แยก เก็บเฉพาะภาพล่าสุด (ไม่ให้ภาพสะสมจนดีเลย์)"""

    def __init__(self, src):
        if isinstance(src, int) and os.name == "nt":
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.cap.isOpened():              # DirectShow เปิดไม่ได้ -> ลองวิธีมาตรฐานของ OpenCV
                self.cap.release()
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src)
        self.ok = self.cap.isOpened()
        self.frame = None
        self.stamp = 0.0
        self.running = self.ok
        if self.ok:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            threading.Thread(target=self._reader, daemon=True).start()
        else:
            self.cap.release()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
                self.stamp = time.time()
            else:
                time.sleep(0.02)
        self.cap.release()     # thread นี้เป็นเจ้าของกล้อง จึงปล่อยเองตอนจบ

    def stop(self):
        self.running = False


# ───────────────────────── จำหน้า (ทำงานใน process แยก ไม่แย่งกับหน้าจอ) ─────────────────────────
def load_known_faces(folder):
    """โหลดรูปในโฟลเดอร์ แล้วแปลงเป็น encoding"""
    encs, names, skipped = [], [], []
    for file in sorted(os.listdir(folder)):
        if not file.lower().endswith(IMAGE_EXT):
            continue
        try:
            img = ImageOps.exif_transpose(Image.open(os.path.join(folder, file))).convert("RGB")
            img.thumbnail((800, 800))
            found = face_recognition.face_encodings(np.array(img))
        except Exception:
            found = []
        if found:
            encs.append(found[0])
            names.append(os.path.splitext(file)[0])
        else:
            skipped.append(file)
    return encs, names, skipped


def _eye_ratio(eye):
    """EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)  ยิ่งมากยิ่งลืมตา (ตาเปิด ~0.25-0.35, หลับตา ~0.1)"""
    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    w = dist(eye[0], eye[3])
    if w < MIN_EYE_PX:
        return None
    return (dist(eye[1], eye[5]) + dist(eye[2], eye[4])) / (2.0 * w)


def face_metrics(lm):
    """จากจุดบนใบหน้า คืนค่า (yaw, ear) หรือ None ถ้าคำนวณไม่ได้
       yaw ~0.5 = หน้าตรง   ear = ความเปิดของตา (เฉลี่ยสองข้าง)"""
    try:
        chin = lm["chin"]
        nose = lm["nose_bridge"][-1]
        left, right = chin[0][0], chin[16][0]
        width = right - left
        if width < 20:
            return None
        yaw = (nose[0] - left) / width
        a, b = _eye_ratio(lm["left_eye"]), _eye_ratio(lm["right_eye"])
        if a is None or b is None:
            return None
        return float(yaw), float((a + b) / 2.0)
    except Exception:
        return None


def track_faces(rgb, items):
    """ติดตามเฉพาะใบหน้าที่กำลังรอกระพริบตา (เร็วกว่าจำหน้าเต็มรูปแบบ เพราะไม่ต้องหาทั้งภาพและไม่คำนวณ encoding)
       items = [(ชื่อ, พิกัดหน้าเดิม)]  คืนค่า [(ชื่อ, พิกัดใหม่ หรือ None, (yaw, ear) หรือ None)]"""
    h, w = rgb.shape[:2]
    out = []
    for name, (t, r, b, l) in items:
        pad = int(max(r - l, b - t) * 0.6)
        y0, y1 = max(t - pad, 0), min(b + pad, h)
        x0, x1 = max(l - pad, 0), min(r + pad, w)
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        locs = face_recognition.face_locations(crop, number_of_times_to_upsample=0)
        if not locs:
            locs = face_recognition.face_locations(crop, number_of_times_to_upsample=1)
        if not locs:
            out.append((name, None, None))
            continue
        cx, cy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        best = min(locs, key=lambda L: ((L[1] + L[3]) / 2.0 - cx) ** 2 + ((L[0] + L[2]) / 2.0 - cy) ** 2)
        lm = face_recognition.face_landmarks(crop, [best])
        loc = (best[0] + y0, best[1] + x0, best[2] + y0, best[3] + x0)
        out.append((name, loc, face_metrics(lm[0]) if lm else None))
    return out


def spoof_score(gray, loc):
    """คะแนนความน่าสงสัยว่าเป็นภาพบนจอ/กระดาษ (0 = ดูเหมือนคนจริง, 1 = น่าสงสัยมาก)
       เป็นการประเมินจากลักษณะภาพ (ไม่ใช่โมเดลที่ฝึกมา) ประกอบด้วย
       1) มีเส้นตรงยาว ๆ ล้อมใบหน้า ซ้าย+ขวา และบนหรือล่าง (เหมือนขอบมือถือ/จอ/กระดาษ)  +0.5 (ครบ 4 ด้าน +0.7)
       2) หน้าสว่างกว่าฉากรอบ ๆ มาก (จอเรืองแสง)                                       +0.2
       3) มีจุดแสงจ้าบนใบหน้า (แสงสะท้อนจอ/กระดาษมัน)                                   +0.2"""
    try:
        h, w = gray.shape
        t, r, b, l = loc
        t, l, b, r = max(t, 0), max(l, 0), min(b, h), min(r, w)
        fw, fh = r - l, b - t
        if fw < 30 or fh < 30:
            return 0.0
        ex, ey = int(fw * 1.5), int(fh * 1.5)
        x0, x1 = max(l - ex, 0), min(r + ex, w)
        y0, y1 = max(t - ey, 0), min(b + ey, h)
        roi = gray[y0:y1, x0:x1]
        score = 0.0

        # 1) กรอบสี่เหลี่ยมล้อมหน้า
        edges = cv2.Canny(cv2.GaussianBlur(roi, (5, 5), 0), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=int(min(fw, fh)), maxLineGap=8)
        sides = set()
        if lines is not None:
            for xa, ya, xb, yb in lines[:, 0]:
                xa, xb, ya, yb = xa + x0, xb + x0, ya + y0, yb + y0
                dx, dy = abs(xb - xa), abs(yb - ya)
                if dx <= dy * 0.15 and dy >= fh * 1.1:               # เส้นตั้งยาวกว่าหน้า
                    x = (xa + xb) / 2.0
                    if min(max(ya, yb), b) - max(min(ya, yb), t) >= fh * 0.5:
                        if x < l - fw * 0.05:
                            sides.add("left")
                        elif x > r + fw * 0.05:
                            sides.add("right")
                elif dy <= dx * 0.15 and dx >= fw * 1.1:             # เส้นนอนยาวกว่าหน้า
                    y = (ya + yb) / 2.0
                    if min(max(xa, xb), r) - max(min(xa, xb), l) >= fw * 0.5:
                        if y < t - fh * 0.05:
                            sides.add("top")
                        elif y > b + fh * 0.05:
                            sides.add("bottom")
        if "left" in sides and "right" in sides and len(sides) >= 3:
            score += 0.7 if len(sides) == 4 else 0.5

        # 2) ความสว่างของหน้าเทียบกับฉากรอบ ๆ
        face = gray[t:b, l:r]
        ring_area = roi.size - face.size
        if ring_area > roi.size * 0.2:
            ring_mean = (float(roi.sum()) - float(face.sum())) / ring_area
            if (float(face.mean()) + 1.0) / (ring_mean + 1.0) > BRIGHT_RATIO:
                score += 0.2

        # 3) จุดแสงจ้าบนใบหน้า
        if float((face >= 245).mean()) > GLARE_FRAC:
            score += 0.2
        return min(score, 1.0)
    except Exception:
        return 0.0


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_antispoof_models():
    """คืนค่า [(path ไฟล์โมเดล, สัดส่วนขยายกรอบ)] ถ้ายังไม่มีไฟล์จะโหลดจาก GitHub ให้ครั้งแรก (ตรวจ sha256 ว่าไฟล์ไม่เสีย)"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    out = []
    for fname, scale in ANTISPOOF_MODELS:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path) or _sha256(path) != ANTISPOOF_SHA256[fname]:
            tmp = path + ".part"
            with urllib.request.urlopen(ANTISPOOF_URL + fname, timeout=30) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            if _sha256(tmp) != ANTISPOOF_SHA256[fname]:
                os.remove(tmp)
                raise RuntimeError(f"ไฟล์ {fname} ที่โหลดมาไม่ตรงกับต้นฉบับ")
            os.replace(tmp, path)
        out.append((path, scale))
    return out


class AntiSpoof:
    """
    โมเดล MiniFASNet 2 ตัว (V2 ขยายกรอบ 2.7 เท่า, V1SE ขยาย 4.0 เท่า) เฉลี่ยผลกัน
    ตอบความน่าจะเป็นว่า "เป็นคนจริง" (0 = ภาพบนจอ/กระดาษ, 1 = คนจริง) ใช้ CPU ตัวละไม่กี่มิลลิวินาที
    ต้นฉบับ: Silent-Face-Anti-Spoofing (Minivision AI, Apache-2.0)
    """

    def __init__(self, models):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self.nets = []
        for path, scale in models:
            sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
            self.nets.append((sess, scale, sess.get_inputs()[0].name, sess.get_outputs()[0].name))

    @staticmethod
    def _crop(bgr, box, scale):
        """ตัดภาพรอบใบหน้า (ขยายกรอบตาม scale) แล้วย่อเป็น 80x80"""
        H, W = bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        scale = min((H - 1) / bh, (W - 1) / bw, scale)
        nw, nh = bw * scale, bh * scale
        cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
        a, b = max(0, int(cx - nw / 2.0)), max(0, int(cy - nh / 2.0))
        c, d = min(W - 1, int(cx + nw / 2.0)), min(H - 1, int(cy + nh / 2.0))
        crop = bgr[b:d + 1, a:c + 1]
        if crop.size == 0:
            crop = bgr
        return cv2.resize(crop, (80, 80))

    def real_prob(self, bgr, loc):
        """bgr = ภาพสี BGR  loc = (top, right, bottom, left)  คืนค่าความน่าจะเป็นว่าเป็นคนจริง"""
        t, r, b, l = loc
        total = np.zeros(3, dtype=np.float32)
        for sess, scale, inp, out in self.nets:
            x = self._crop(bgr, (l, t, r, b), scale).astype(np.float32).transpose(2, 0, 1)[None]
            logits = sess.run([out], {inp: x})[0][0]
            e = np.exp(logits - logits.max())
            total += e / e.sum()
        return float((total / len(self.nets))[1])      # ช่อง 1 = "คนจริง"


def recognizer_process(in_q, out_q):
    """process ลูก: รับคำสั่งจากโปรแกรมหลัก ทำงานหนัก (หาหน้า/จำหน้า) แล้วส่งผลกลับ"""
    known_enc, known_names = [], []

    spoofer = None
    if AI_LIVENESS:
        try:
            if importlib.util.find_spec("onnxruntime") is None:
                raise ImportError("onnxruntime")
            spoofer = AntiSpoof(ensure_antispoof_models())
        except ImportError:
            reason = "ยังไม่ได้ติดตั้ง onnxruntime  (พิมพ์ใน Terminal: pip install onnxruntime)"
        except Exception as e:
            reason = (f"โหลดโมเดลไม่ได้: {e}\n\nดาวน์โหลด MiniFASNetV2.onnx และ MiniFASNetV1SE.onnx เองได้จาก\n"
                      f"{ANTISPOOF_URL}\nแล้วนำไปใส่ในโฟลเดอร์ {MODEL_DIR}")
        if spoofer is None:
            out_q.put(("warn", "เปิดระบบ AI ตรวจคนจริงไม่ได้ จะใช้การตรวจแบบเดิม (ตรวจจากภาพ + กระพริบตา) แทน\n\n" + reason))
        out_q.put(("ai", spoofer is not None))

    while True:
        msg = in_q.get()
        kind = msg[0]
        try:
            if kind == "stop":
                break

            elif kind == "load_dir":
                known_enc, known_names, skipped = load_known_faces(msg[1])
                out_q.put(("loaded", list(known_names)))
                if skipped:
                    out_q.put(("warn", "ข้ามรูปที่ไม่พบใบหน้า: " + ", ".join(skipped)))

            elif kind == "frame":
                rgb, tolerance = msg[1], msg[2]
                h, w = rgb.shape[:2]
                small = cv2.resize(rgb, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
                inv = 1.0 / DETECT_SCALE
                locations = [(max(int(t * inv), 0), min(int(r * inv), w), min(int(bt * inv), h), max(int(l * inv), 0))
                             for t, r, bt, l in face_recognition.face_locations(small)]
                encodings = face_recognition.face_encodings(rgb, locations)
                landmarks = face_recognition.face_landmarks(rgb, locations)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if locations else None
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if (locations and spoofer is not None) else None
                found = []
                for loc, enc, lm in zip(locations, encodings, landmarks):
                    name, dist = "Unknown", 1.0
                    if known_enc:
                        dists = face_recognition.face_distance(known_enc, enc)
                        best = int(dists.argmin())
                        dist = float(dists[best])
                        if dist < tolerance:
                            name = known_names[best]
                    ai = None                      # ความน่าจะเป็นว่าเป็นคนจริง (เช็กเฉพาะคนที่จำชื่อได้)
                    if bgr is not None and name != "Unknown":
                        try:
                            ai = spoofer.real_prob(bgr, loc)
                        except Exception:
                            ai = None
                    found.append((tuple(int(v) for v in loc), name, dist, face_metrics(lm),
                                  (spoof_score(gray, loc), ai)))
                out_q.put(("result", found))

            elif kind == "track":
                out_q.put(("track_result", track_faces(msg[1], msg[2])))

            elif kind == "register":
                name, rgb = msg[1], msg[2]
                locations = face_recognition.face_locations(rgb)
                if len(locations) != 1:
                    out_q.put(("register_fail",
                               f"พบใบหน้า {len(locations)} ใบ ต้องมีใบหน้าเดียวในภาพ ลองใหม่อีกครั้ง"))
                    continue
                enc = face_recognition.face_encodings(rgb, locations)[0]
                Image.fromarray(rgb).save(os.path.join(KNOWN_DIR, name + ".jpg"), quality=95)
                known_enc.append(enc)
                known_names.append(name)
                out_q.put(("registered", name))
        except Exception as e:
            traceback.print_exc()
            if kind == "frame":
                out_q.put(("result", []))
            elif kind == "track":
                out_q.put(("track_result", []))
            elif kind == "register":
                out_q.put(("register_fail", f"ลงทะเบียนไม่สำเร็จ: {e}"))


class Recognizer:
    """ส่งภาพไปให้ process ลูกจำหน้า แล้วรับผลกลับมา + จัดการบันทึกเข้าเรียน"""

    def __init__(self, events, marked):
        self.events = events
        self.cam = None
        self.tolerance = DEFAULT_TOLERANCE
        self.marked = dict(marked)    # {ชื่อ: (เวลา, สถานะ)} รวมคนที่ถูกบันทึกเป็น "ขาด" แล้ว
        self.streak = {}
        self.results = []         # [(พิกัดหน้า, ชื่อ, ระยะห่าง, (เวลา, สถานะ) หรือ None, ข้อความให้ทำท่า, (yaw, mouth))]
        self.live = {}            # {ชื่อ: สถานะการรอกระพริบตา} ของคนที่กำลังพิสูจน์ว่าเป็นคนจริง
        self.susp = {}            # {ชื่อ: คะแนนความน่าสงสัยของ CONFIRM รอบล่าสุด}
        self.force_full = False   # True = รอบถัดไปให้จำหน้าเต็มรูปแบบทันที (ใช้ยืนยันตัวตนหลังกระพริบ)
        self.last_full = 0.0
        self.last_sent_frame = None
        self.pending_scale = 1.0  # สัดส่วนย่อของภาพที่ส่งไปจำล่าสุด
        self.send_scale = 1.0     # สัดส่วนย่อของภาพที่ใช้ได้ผลล่าสุด (ไว้แปลงพิกัดตอนวาด)
        self.error_until = 0.0
        self.busy = False         # กำลังรอผลจาก process ลูกอยู่หรือไม่
        self.last_sent = 0.0
        self.dead_reported = False

        ctx = mp.get_context("spawn")
        self.in_q = ctx.Queue()
        self.out_q = ctx.Queue()
        self.proc = ctx.Process(target=recognizer_process, args=(self.in_q, self.out_q), daemon=True)
        self.proc.start()

    def load_dir(self):
        self.in_q.put(("load_dir", KNOWN_DIR))

    def register(self, name, frame):
        self.in_q.put(("register", name, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def get(self):
        return list(self.results)

    def update(self):
        """เรียกจากลูปหลักทุกรอบ: รับผลที่เสร็จแล้ว และส่งภาพใหม่ไปให้จำ"""
        while True:
            try:
                msg = self.out_q.get_nowait()
            except queue.Empty:
                break
            if msg[0] in ("result", "track_result"):
                self.busy = False
                if self.cam is None:
                    continue
                if msg[0] == "result":
                    self._update_attendance(msg[1])
                else:
                    self._update_tracks(msg[1])
            else:
                self.events.put(msg)       # loaded / registered / register_fail / warn

        if not self.proc.is_alive():
            if not self.dead_reported:
                self.dead_reported = True
                self.events.put(("error", "ระบบจำหน้าหยุดทำงาน กรุณาปิดแล้วเปิดโปรแกรมใหม่"))
            return

        cam = self.cam
        if cam is None:
            self.results = []
            self.streak.clear()
            self.live.clear()
            self.susp.clear()
            return

        frame = cam.frame
        if frame is not None and not self.busy and frame is not self.last_sent_frame:
            now = time.time()
            tracks = [(n, st["loc"]) for n, st in self.live.items()]
            # ไม่มีใครรอกระพริบ หรือถึงเวลาจำหน้าเต็มรูปแบบ -> ส่งแบบเต็ม (จำกัดความถี่ด้วย INTERVAL)
            # ระหว่างรอกระพริบ -> ส่งแบบติดตามตา ไม่จำกัดความถี่ เพื่อไม่พลาดการกระพริบที่เร็วมาก
            full = (not tracks) or self.force_full or now - self.last_full >= FULL_EVERY
            if not (full and now - self.last_sent < INTERVAL):
                s = min(1.0, SEND_WIDTH / frame.shape[1])
                small = cv2.resize(frame, (0, 0), fx=s, fy=s) if s < 1.0 else frame
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                self.pending_scale = s
                if full:
                    self.in_q.put(("frame", rgb, self.tolerance))
                    self.last_full = now
                    self.force_full = False
                else:
                    self.in_q.put(("track", rgb, tracks))
                self.busy = True
                self.last_sent = now
                self.last_sent_frame = frame

    # ───── ตรวจว่าเป็นคนจริง: ต้องกระพริบตา (Liveness) ─────
    @staticmethod
    def _near(prev_loc, loc):
        """ใบหน้า loc อยู่ใกล้ตำแหน่งเดิม prev_loc พอที่จะเป็นคนเดิมหรือไม่"""
        pt, pr, pb, pl = prev_loc
        t, r, b, l = loc
        size = max(pr - pl, pb - pt)
        dx = ((l + r) - (pl + pr)) / 2.0
        dy = ((t + b) - (pt + pb)) / 2.0
        return (dx * dx + dy * dy) ** 0.5 < size * 0.8

    @staticmethod
    def _new_challenge(loc):
        now = time.time()
        return {"open": deque(maxlen=15),      # ค่า EAR ตอนลืมตา (ใช้หาค่ากลางของคนนี้)
                "closed": 0, "closed_at": 0.0, "blinked": False, "lost": False,
                "final": 0, "bad": 0, "t0": now, "seen": now, "loc": loc}

    @staticmethod
    def _prompt(st):
        if st["lost"]:
            return "ขอตรวจสอบเพิ่ม: มองตรงและเข้าใกล้กล้องอีกนิด"
        if st["blinked"] or len(st["open"]) < BASELINE_N:
            return "ขอตรวจสอบเพิ่ม: มองตรงเข้ากล้อง"
        return "ขอตรวจสอบเพิ่ม: กระพริบตา 1 ครั้ง"

    def _feed_blink(self, st, metrics, now):
        """ป้อนค่า EAR ทีละเฟรม ตรวจลำดับ ลืมตา -> หลับตา -> ลืมตาอีกครั้ง"""
        if st["blinked"]:
            return
        if metrics is None or abs(metrics[0] - 0.5) > BLINK_MAX_YAW:
            st["lost"] = True                   # อ่านค่าตาไม่ได้ / หันหน้าเกินไป
            return
        st["lost"] = False
        ear = metrics[1]
        op = st["open"]
        if len(op) < BASELINE_N:                # ยังเก็บค่าตอนลืมตาไม่ครบ
            op.append(ear)
            return
        base = statistics.median(op)
        if base < BLINK_MIN_BASE:               # ตาปิดค้าง/เล็กเกินไป ยังตัดสินไม่ได้
            op.append(ear)
            return
        if ear < base * BLINK_RATIO:            # ตากำลังหลับ
            if st["closed"] == 0:
                st["closed_at"] = now
            st["closed"] += 1
            return
        if st["closed"] >= BLINK_MIN_CLOSED and now - st["closed_at"] <= BLINK_MAX_SEC:
            st["blinked"] = True                # เห็นกระพริบแล้ว รอบถัดไปให้จำหน้าเต็มรูปแบบเพื่อยืนยันตัวตน
            self.force_full = True
        st["closed"] = 0
        op.append(ear)

    def _mark_now(self, n, phase):
        status = ST_ONTIME if phase == "ontime" else ST_LATE
        try:
            t = mark(n, status)
        except PermissionError:
            if time.time() > self.error_until:
                self.error_until = time.time() + 6
                self.events.put(("error", "บันทึกไม่ได้ กรุณาปิดไฟล์ attendance.csv ที่เปิดค้างอยู่"))
            return False
        self.marked[n] = (t, status)
        self.events.put(("checkin", n, t, status))
        return True

    def _update_tracks(self, items):
        """ผลติดตามตา (เร็ว) ระหว่างรอกระพริบ: อัปเดตตำแหน่งหน้า + ป้อนค่า EAR"""
        now = time.time()
        moved = {}
        for n, loc, metrics in items:
            st = self.live.get(n)
            if st is None:
                continue
            if loc is None:                     # หาใบหน้าไม่เจอ ปล่อยให้ผ่อนผันตาม LIVENESS_GRACE
                st["lost"] = True
                if now - st["seen"] > LIVENESS_GRACE:
                    del self.live[n]
                continue
            st["loc"], st["seen"] = loc, now
            self._feed_blink(st, metrics, now)
            moved[n] = (loc, metrics, self._prompt(st))
        if moved:                               # ให้กรอบบนภาพตามหน้าทัน
            self.results = [(moved[d][0], d, dist, info, moved[d][2], moved[d][1], sp) if d in moved
                            else (loc, d, dist, info, prompt, m, sp)
                            for loc, d, dist, info, prompt, m, sp in self.results]

    @staticmethod
    def _too_small(loc):
        """ใบหน้าเล็ก/ไกลเกินไปหรือไม่ (loc = top, right, bottom, left บนภาพที่ส่งไปจำ)"""
        return MIN_FACE_PX > 0 and (loc[1] - loc[3]) < MIN_FACE_PX

    def _verdict(self, n):
        """สรุปจากผลของ CONFIRM รอบล่าสุด:
           'fake' = AI มั่นใจว่าเป็นภาพบนจอ/กระดาษ   'suspicious' = น่าสงสัย ต้องกระพริบตา   'real' = ดูเหมือนคนจริง"""
        hist = list(self.susp.get(n) or [])
        if not hist:
            return "real"
        heur = max(h for h, _ in hist)
        ai = [a for _, a in hist if a is not None]
        if ai:
            mid = statistics.median(ai)
            if AI_BLOCK_BELOW > 0 and mid < AI_BLOCK_BELOW:
                return "fake"
            if mid < AI_REAL_MIN:
                return "suspicious"
        return "suspicious" if heur >= SPOOF_THRESHOLD else "real"

    def _update_attendance(self, faces):
        """
        faces = [(พิกัดหน้า, ชื่อ, ระยะห่าง, (yaw, ear) หรือ None, คะแนนความน่าสงสัย)]  (ผลจำหน้าเต็มรูปแบบ)
        ลำดับ: จำหน้าได้ติดกัน CONFIRM รอบ -> รอกระพริบตา -> กระพริบแล้วจำหน้าได้ซ้ำ -> บันทึก
        """
        now = time.time()
        phase = session_phase()
        names_now = {}                           # ชื่อ -> ลำดับใบหน้า (เฉพาะที่จำชื่อได้)
        far = {}                                 # ใบหน้าที่เล็ก/ไกลเกินไป: ลำดับ -> ชื่อ
        for i, (loc, n, _, _, _) in enumerate(faces):
            if n == "Unknown" or n in names_now:
                continue
            if self._too_small(loc):
                far[i] = n
                continue
            names_now[n] = i

        shown = {}                               # ลำดับใบหน้า -> (ชื่อที่แสดง, ข้อความบอกให้ทำ)
        taken = set()
        for i, n in far.items():
            if n not in self.marked:
                shown[i] = (n, FAR_PROMPT)

        # 1) คนที่กำลังรอกระพริบอยู่
        for n, st in list(self.live.items()):
            if n in self.marked or phase == "closed":
                del self.live[n]
                continue
            i, swapped = None, False
            for j, (loc, name, _, _, _) in enumerate(faces):
                if j in taken or not self._near(st["loc"], loc):
                    continue
                if name not in ("Unknown", n):
                    swapped = True               # มีคนอื่นที่รู้จักมายืนแทนตำแหน่งเดิม
                    continue
                if name == n:
                    i = j
                    break
                if i is None:
                    i = j
            if i is None:
                if swapped or now - st["seen"] > LIVENESS_GRACE:
                    del self.live[n]
                continue
            taken.add(i)
            loc, name, _, metrics, _ = faces[i]
            st["loc"], st["seen"] = loc, now

            if self._too_small(loc):                     # ถอยห่างกล้องระหว่างพิสูจน์ตัว -> ยกเลิก
                del self.live[n]
                shown[i] = (n, FAR_PROMPT)
                continue

            if LIVENESS and self._verdict(n) == "fake":      # ระหว่างพิสูจน์ตัว AI เริ่มมั่นใจว่าเป็นภาพบนจอ -> ยกเลิก
                del self.live[n]
                shown[i] = (n, FAKE_PROMPT)
                continue

            if now - st["t0"] > LIVENESS_TIMEOUT:        # รอนานเกินไป เริ่มใหม่
                self.live[n] = st = self._new_challenge(loc)
                self.events.put(("error", "หมดเวลา ลองใหม่อีกครั้ง"))
                shown[i] = (n, self._prompt(st))
                continue

            if name == n:
                st["bad"] = 0
            else:
                st["bad"] += 1
                if st["bad"] > MAX_BAD:                  # จำหน้าเป็นคนเดิมไม่ได้ต่อเนื่อง ยกเลิก
                    del self.live[n]
                    continue
            self._feed_blink(st, metrics, now)
            if st["blinked"]:                            # กระพริบแล้ว ต้องจำหน้าได้อีกครั้งก่อนบันทึก
                st["final"] = st["final"] + 1 if name == n else 0
                if st["final"] >= FINAL_FRAMES:
                    if self._mark_now(n, phase):
                        del self.live[n]
                        shown[i] = (n, None)
                        continue
                    st["final"] = 0                      # บันทึกไม่สำเร็จ ลองใหม่รอบหน้า
            shown[i] = (n, self._prompt(st))

        # 2) นับรอบที่จำหน้าได้ติดกัน และเริ่มตรวจคนที่จำได้ครบ CONFIRM รอบ
        for n in names_now:
            self.streak[n] = self.streak.get(n, 0) + 1
        for n in list(self.streak):
            if n not in names_now:
                self.streak[n] = 0

        for n, i in names_now.items():
            self.susp.setdefault(n, deque(maxlen=CONFIRM)).append(faces[i][4])
        for n in list(self.susp):
            if n not in names_now:
                del self.susp[n]

        for n, i in names_now.items():
            if n in self.marked or n in self.live or self.streak[n] < CONFIRM:
                continue
            if phase == "closed":                        # หมดเวลาเช็กชื่อแล้ว ไม่บันทึกให้
                self.streak[n] = 0
                if time.time() > self.error_until:
                    self.error_until = time.time() + 6
                    end = session_times()[2]
                    self.events.put(("error", f"หมดเวลาเช็กชื่อแล้ว ({end:%H:%M})"))
                continue
            verdict = self._verdict(n) if LIVENESS else "real"
            if verdict == "fake":                        # AI มั่นใจว่าเป็นภาพบนจอ/กระดาษ -> ไม่ให้ผ่าน (ไม่เปิดให้กระพริบตา)
                shown[i] = (n, FAKE_PROMPT)
                if time.time() > self.error_until:
                    self.error_until = time.time() + 6
                    self.events.put(("error", f"ตรวจพบว่าไม่ใช่คนจริง (ภาพบนจอ/กระดาษ): {n}"))
                continue
            suspicious = verdict == "suspicious"
            if not LIVENESS or (AUTO_TRUST and not suspicious):   # ดูเหมือนคนจริง (หรือปิดระบบ) -> บันทึกเลย
                if not self._mark_now(n, phase):
                    self.streak[n] = 0
                continue
            self.live[n] = st = self._new_challenge(faces[i][0])
            self._feed_blink(st, faces[i][3], now)
            shown[i] = (n, self._prompt(st))

        # 3) ผลที่ใช้วาดบนภาพ
        self.send_scale = self.pending_scale
        out = []
        for i, (loc, n, d, metrics, sp) in enumerate(faces):
            disp, prompt = shown.get(i, (n, None))
            out.append((loc, disp, d, self.marked.get(disp), prompt, metrics, sp))
        self.results = out

    def stop(self):
        try:
            self.in_q.put(("stop",))
            self.proc.join(timeout=1.5)
            if self.proc.is_alive():
                self.proc.terminate()
        finally:
            self.in_q.cancel_join_thread()
            self.out_q.cancel_join_thread()


# ───────────────────────── หน้าตาโปรแกรม ─────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ระบบเช็กชื่อเข้าเรียนด้วยใบหน้า")
        self.geometry("1280x780")
        self.minsize(1120, 700)
        self.configure(fg_color=BG)

        self.f_title = ctk.CTkFont(family=FONT, size=22, weight="bold")
        self.f_head = ctk.CTkFont(family=FONT, size=15, weight="bold")
        self.f_body = ctk.CTkFont(family=FONT, size=14)
        self.f_small = ctk.CTkFont(family=FONT, size=12)
        self.f_big = ctk.CTkFont(family=FONT, size=28, weight="bold")

        self.events = queue.Queue()
        try:
            migrate_csv()                        # ไฟล์เก่าไม่มีคอลัมน์ status ให้เติมให้
        except PermissionError:
            pass
        self.attendance = load_today()           # {ชื่อ: (เวลา, สถานะ)} ของวันนี้ (รวมคนที่ขาด)
        self.registered = []                     # ชื่อคนที่ลงทะเบียนใบหน้าไว้
        self.roster_ready = False                # โหลดรายชื่อจาก known_faces เสร็จหรือยัง
        self.absent_day = None                   # วันที่ที่บันทึก "ขาด" ให้แล้ว
        self.absent_err_until = 0.0
        self.phase = None
        self.recognizer = Recognizer(self.events, self.attendance)
        self.camera = None
        self.connecting = False
        self.registering = False
        self.toast = None                        # (ข้อความ, สี, เวลาหมดอายุ)
        self.photo = None
        self.last_frame = None
        self.dirty = True
        self.today = datetime.now().strftime("%Y-%m-%d")
        self._status = None

        self._build_ui()
        self._set_status("idle")

        self.recognizer.load_dir()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._clock()
        self._tick()

    # ───── สร้างหน้าจอ ─────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # ส่วนหัว
        header = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=68)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)

        box = ctk.CTkFrame(header, fg_color="transparent")
        box.grid(row=0, column=0, sticky="w", padx=24, pady=10)
        ctk.CTkLabel(box, text="ระบบเช็กชื่อเข้าเรียนด้วยใบหน้า", font=self.f_title,
                     text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(box, text="Face Attendance  •  ข้อมูลทั้งหมดเก็บในเครื่องนี้เท่านั้น",
                     font=self.f_small, text_color=MUTED).pack(anchor="w")

        clock_box = ctk.CTkFrame(header, fg_color="transparent")
        clock_box.grid(row=0, column=1, padx=24)
        self.clock_label = ctk.CTkLabel(clock_box, text="", font=self.f_head, text_color=TEXT)
        self.clock_label.pack(anchor="e")
        self.session_label = ctk.CTkLabel(clock_box, text="", font=self.f_small, text_color=MUTED)
        self.session_label.pack(anchor="e")

        # การ์ดวิดีโอ (ซ้าย)
        left = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=18)
        left.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=20)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.status = ctk.CTkLabel(left, text="", font=self.f_head, anchor="w")
        self.status.grid(row=0, column=0, sticky="w", padx=18, pady=(14, 6))

        self.video_box = tk.Frame(left, bg=VIDEO_BG, width=640, height=480, highlightthickness=0)
        self.video_box.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.video_box.grid_propagate(False)
        self.video_label = tk.Label(self.video_box, bg=VIDEO_BG, fg=MUTED, bd=0,
                                    font=(FONT, 16), justify="center")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self._show_placeholder()

        # แถบด้านขวา
        side = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=18, width=390)
        side.grid(row=1, column=1, sticky="ns", padx=(10, 20), pady=20)
        side.grid_propagate(False)

        # การ์ดสถิติ
        stats = ctk.CTkFrame(side, fg_color="transparent")
        stats.pack(fill="x", padx=14, pady=(14, 8))
        for i in range(4):
            stats.grid_columnconfigure(i, weight=1, uniform="stat")
        self.stat_total = self._stat_card(stats, 0, "ลงทะเบียน", ACCENT)
        self.stat_present = self._stat_card(stats, 1, "มาแล้ว", GREEN)
        self.stat_wait = self._stat_card(stats, 2, "ยังไม่มา", AMBER)
        self.stat_absent = self._stat_card(stats, 3, "ขาด", PURPLE)

        # รายชื่อ
        self.tabs = ctk.CTkTabview(
            side, fg_color=CARD, corner_radius=14,
            segmented_button_fg_color=BG,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT_HOVER,
            segmented_button_unselected_color=BG,
            segmented_button_unselected_hover_color=ROW,
            text_color=TEXT,
        )
        self.tabs.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        tab_in = self.tabs.add("มาแล้ว")
        tab_out = self.tabs.add("ยังไม่มา")
        tab_abs = self.tabs.add("ขาด")
        self.list_in = ctk.CTkScrollableFrame(tab_in, fg_color="transparent")
        self.list_in.pack(fill="both", expand=True)
        self.list_out = ctk.CTkScrollableFrame(tab_out, fg_color="transparent")
        self.list_out.pack(fill="both", expand=True)
        self.list_abs = ctk.CTkScrollableFrame(tab_abs, fg_color="transparent")
        self.list_abs.pack(fill="both", expand=True)

        # ตั้งค่า
        cfg = ctk.CTkFrame(side, fg_color=CARD, corner_radius=14)
        cfg.pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkLabel(cfg, text="แหล่งภาพ (URL กล้อง หรือเลข 0 = เว็บแคม)", font=self.f_small,
                     text_color=MUTED).pack(anchor="w", padx=14, pady=(10, 2))
        self.src_entry = ctk.CTkEntry(cfg, font=self.f_body, fg_color=BG, border_color=ROW,
                                      text_color=TEXT, height=34)
        self.src_entry.insert(0, DEFAULT_SOURCE)
        self.src_entry.pack(fill="x", padx=14)

        self.tol_label = ctk.CTkLabel(cfg, text="", font=self.f_small, text_color=MUTED)
        self.tol_label.pack(anchor="w", padx=14, pady=(10, 0))
        self.tol_slider = ctk.CTkSlider(cfg, from_=0.35, to=0.65, number_of_steps=30,
                                        progress_color=ACCENT, button_color=ACCENT,
                                        button_hover_color=ACCENT_HOVER,
                                        command=self._on_tolerance)
        self.tol_slider.set(DEFAULT_TOLERANCE)
        self.tol_slider.pack(fill="x", padx=14, pady=(2, 12))
        self._on_tolerance(DEFAULT_TOLERANCE)

        # ปุ่ม
        btns = ctk.CTkFrame(side, fg_color="transparent")
        btns.pack(fill="x", padx=14, pady=(0, 14))
        self.cam_btn = ctk.CTkButton(btns, text="เริ่มกล้อง", font=self.f_head, height=42,
                                     fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                     corner_radius=12, command=self.toggle_camera)
        self.cam_btn.pack(fill="x", pady=(0, 8))
        self.reg_btn = ctk.CTkButton(btns, text="เพิ่มบุคคลใหม่", font=self.f_body, height=38,
                                     fg_color=ROW, hover_color="#26385f", corner_radius=12,
                                     command=self.add_person)
        self.reg_btn.pack(fill="x", pady=(0, 8))
        row = ctk.CTkFrame(btns, fg_color="transparent")
        row.pack(fill="x")
        row.grid_columnconfigure((0, 1), weight=1, uniform="b")
        ctk.CTkButton(row, text="เปิดไฟล์เช็กชื่อ", font=self.f_body, height=38,
                      fg_color=ROW, hover_color="#26385f", corner_radius=12,
                      command=self.open_csv).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(row, text="รีเซ็ตการเข้าเรียน", font=self.f_body, height=38,
                      fg_color="#3b1f2b", hover_color="#5b2a3d", text_color="#fca5a5",
                      corner_radius=12, command=self.open_reset_dialog
                      ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _stat_card(self, parent, col, caption, color):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14)
        card.grid(row=0, column=col, sticky="ew", padx=3)
        num = ctk.CTkLabel(card, text="0", font=self.f_big, text_color=color)
        num.pack(pady=(8, 0))
        ctk.CTkLabel(card, text=caption, font=self.f_small, text_color=MUTED).pack(pady=(0, 8))
        return num

    def _show_placeholder(self):
        self.video_label.configure(image="", text="กล้องปิดอยู่\nกด \"เริ่มกล้อง\" เพื่อเริ่มเช็กชื่อ")
        self.photo = None

    # ───── ตัวช่วยเล็ก ๆ ─────
    def _set_status(self, key):
        if key == self._status:
            return
        self._status = key
        text, color = STATUS[key]
        self.status.configure(text=f"●  {text}", text_color=color)

    def _on_tolerance(self, value):
        value = round(float(value), 2)
        self.recognizer.tolerance = value
        self.tol_label.configure(text=f"ความเข้มงวดในการจำหน้า: {value:.2f}   (น้อย = เข้มงวด)")

    def show_toast(self, text, color=GREEN, secs=3.0):
        self.toast = (text, color, time.time() + secs)

    def _clock(self):
        now = datetime.now()
        self.clock_label.configure(text=now.strftime("%d/%m/%Y   %H:%M:%S"))
        today = now.strftime("%Y-%m-%d")
        if today != self.today:            # ข้ามวัน เริ่มเช็กชื่อใหม่
            self.today = today
            self.attendance.clear()
            self.recognizer.marked.clear()
            self.recognizer.streak.clear()
            self.recognizer.live.clear()
            self.absent_day = None
            self.dirty = True
        self._check_session(now)
        self.after(1000, self._clock)

    # ───── สาย / ขาด ตามเวลาเรียน ─────
    def _check_session(self, now):
        """อัปเดตข้อความสถานะช่วงเวลา และเมื่อหมดเวลาให้บันทึกคนที่ยังไม่มาเป็น 'ขาด'"""
        start, late_at, end = session_times(now)
        phase = session_phase(now)
        if phase != self.phase:
            self.phase = phase
            self.dirty = True              # ให้รายชื่อ "ยังไม่มา" เปลี่ยนเป็น "สายแล้ว" / "ขาด"
            if phase == "ontime":
                text = f"คาบทดลอง {start:%H:%M}–{end:%H:%M}  •  สายหลัง {late_at:%H:%M}"
                color = MUTED
            elif phase == "late":
                text = f"ขณะนี้นับเป็นมาสาย  •  ขาดเมื่อ {end:%H:%M}"
                color = AMBER
            else:
                text = f"หมดเวลาเช็กชื่อแล้ว ({end:%H:%M})  •  คนที่ไม่มาถูกบันทึกเป็นขาด"
                color = PURPLE
            self.session_label.configure(text=text, text_color=color)

        if phase == "closed" and self.roster_ready and self.absent_day != self.today:
            self._close_session()

    def _close_session(self):
        missing = [n for n in self.registered if n not in self.attendance]
        try:
            t = mark_absent(missing)
        except PermissionError:
            # ไฟล์ถูกเปิดค้างอยู่ จะลองใหม่ทุกวินาที (แจ้งเตือนไม่ถี่เกินไป)
            if time.time() > self.absent_err_until:
                self.absent_err_until = time.time() + 6
                self.show_toast("บันทึกขาดเรียนไม่ได้ กรุณาปิดไฟล์ attendance.csv", RED, 5.0)
            return
        self.absent_day = self.today
        for n in missing:
            self.attendance[n] = (t, ST_ABSENT)
            self.recognizer.marked[n] = (t, ST_ABSENT)
        self.dirty = True
        if missing:
            self.show_toast(f"หมดเวลา บันทึกขาด {len(missing)} คน", PURPLE, 4.0)

    # ───── เปิด/ปิดกล้อง ─────
    def toggle_camera(self):
        if self.connecting:
            return
        if self.camera is not None:
            self.stop_camera()
            return
        src = parse_source(self.src_entry.get())
        self.connecting = True
        self.cam_btn.configure(text="กำลังเชื่อมต่อ...", state="disabled")
        threading.Thread(target=self._open_camera, args=(src,), daemon=True).start()

    def _open_camera(self, src):
        cam = Camera(src)
        if cam.ok:
            t0 = time.time()
            while cam.frame is None and time.time() - t0 < 8:
                time.sleep(0.05)
        if not cam.ok or cam.frame is None:
            cam.stop()
            self.events.put(("cam", None))
        else:
            self.events.put(("cam", cam))

    def stop_camera(self):
        self.recognizer.cam = None
        if self.camera is not None:
            self.camera.stop()
        self.camera = None
        self.last_frame = None
        self.cam_btn.configure(text="เริ่มกล้อง", fg_color=ACCENT, hover_color=ACCENT_HOVER)
        self._show_placeholder()

    # ───── ลงทะเบียนบุคคลใหม่ ─────
    def add_person(self):
        if self.registering:
            return
        if self.camera is None or self.camera.frame is None:
            messagebox.showinfo("เพิ่มบุคคลใหม่", "กรุณาเปิดกล้องก่อน แล้วให้ผู้ลงทะเบียนมองตรงเข้ากล้อง")
            return
        dialog = ctk.CTkInputDialog(text="ชื่อที่จะแสดงและบันทึกในไฟล์เช็กชื่อ:", title="เพิ่มบุคคลใหม่")
        raw = dialog.get_input()
        if not raw:
            return
        name = "".join(ch for ch in raw if ch not in '<>:"/\\|?*').strip(" .")[:40]
        if not name:
            messagebox.showwarning("เพิ่มบุคคลใหม่", "ชื่อไม่ถูกต้อง")
            return
        if name in self.registered:
            messagebox.showwarning("เพิ่มบุคคลใหม่", f"มีชื่อ \"{name}\" อยู่แล้ว")
            return
        ok = messagebox.askyesno(
            "ขอความยินยอม",
            f"เจ้าของใบหน้า \"{name}\" ยินยอมให้ถ่ายและเก็บภาพใบหน้า\n"
            "เพื่อใช้เช็กชื่อเข้าเรียนในโปรแกรมนี้ใช่หรือไม่?\n\n"
            "(ภาพจะเก็บในโฟลเดอร์ known_faces ของเครื่องนี้เท่านั้น)")
        if not ok:
            return
        self.registering = True
        self.reg_btn.configure(state="disabled")
        self._countdown(3, name)

    def _countdown(self, n, name):
        if self.camera is None or self.camera.frame is None:
            self._reg_finish()
            return
        if n > 0:
            self.show_toast(f"มองตรงเข้ากล้อง...  {n}", ACCENT, 1.2)
            self.after(1000, self._countdown, n - 1, name)
            return
        self.recognizer.register(name, self.camera.frame.copy())
        self.show_toast("กำลังบันทึกใบหน้า...", ACCENT, 2.0)

    def _reg_finish(self):
        self.registering = False
        self.reg_btn.configure(state="normal")

    def open_csv(self):
        if not os.path.exists(ATTENDANCE_FILE):
            messagebox.showinfo("เปิดไฟล์", "ยังไม่มีไฟล์เช็กชื่อ (ยังไม่มีใครเช็กชื่อ)")
            return
        if hasattr(os, "startfile"):
            os.startfile(ATTENDANCE_FILE)
        else:
            messagebox.showinfo("เปิดไฟล์", ATTENDANCE_FILE)

    # ───── รีเซ็ตการเข้าเรียน ─────
    def open_reset_dialog(self):
        win = getattr(self, "reset_win", None)
        if win is not None and win.winfo_exists():
            win.focus()
            return
        win = ctk.CTkToplevel(self)
        self.reset_win = win
        win.title("รีเซ็ตการเข้าเรียน")
        win.geometry("460x370")
        win.resizable(False, False)
        win.configure(fg_color=PANEL)
        win.transient(self)
        win.after(150, win.grab_set)
        win.after(200, win.focus)

        ctk.CTkLabel(win, text="รีเซ็ตการเข้าเรียน", font=self.f_title, text_color=TEXT) \
            .pack(pady=(22, 4))
        ctk.CTkLabel(win, text="ระบบจะสำรองไฟล์เดิมไว้ในโฟลเดอร์ backup ให้ก่อนเสมอ",
                     font=self.f_small, text_color=MUTED).pack(pady=(0, 14))

        def choose(scope):
            win.destroy()
            self.reset_attendance(scope)

        ctk.CTkButton(win, text="รีเซ็ตเฉพาะวันนี้", font=self.f_head, height=46,
                      fg_color=AMBER, hover_color="#d97706", text_color="#1f1300", corner_radius=12,
                      command=lambda: choose("today")).pack(fill="x", padx=30)
        ctk.CTkLabel(win, text="ล้างรายชื่อ \"มาแล้ว\" ของวันนี้ ประวัติของวันอื่นยังอยู่",
                     font=self.f_small, text_color=MUTED).pack(pady=(4, 14))

        ctk.CTkButton(win, text="ล้างประวัติทั้งหมด", font=self.f_head, height=46,
                      fg_color=RED, hover_color="#dc2626", corner_radius=12,
                      command=lambda: choose("all")).pack(fill="x", padx=30)
        ctk.CTkLabel(win, text="ลบทุกวันออกจาก attendance.csv (เหลือแต่หัวตาราง)",
                     font=self.f_small, text_color=MUTED).pack(pady=(4, 14))

        ctk.CTkButton(win, text="ยกเลิก", font=self.f_body, height=38, fg_color=ROW,
                      hover_color="#26385f", corner_radius=12,
                      command=win.destroy).pack(fill="x", padx=30)

    def reset_attendance(self, scope):
        try:
            n = count_records(scope)
        except PermissionError:
            messagebox.showerror("รีเซ็ตการเข้าเรียน",
                                 "อ่านไฟล์ attendance.csv ไม่ได้ กรุณาปิดไฟล์ที่เปิดค้างอยู่ใน Excel ก่อน")
            return
        if n == 0:
            messagebox.showinfo("รีเซ็ตการเข้าเรียน",
                                "ไม่มีรายการให้ลบ" if scope == "all" else "วันนี้ยังไม่มีรายการเช็กชื่อ")
            return

        if scope == "today":
            msg = f"ลบรายการเช็กชื่อของวันนี้ ({n} รายการ) ใช่หรือไม่?\nประวัติของวันอื่นจะไม่ถูกลบ"
        else:
            msg = f"ลบประวัติการเข้าเรียนทั้งหมด ({n} รายการ) ใช่หรือไม่?"
        if not messagebox.askyesno("ยืนยันการรีเซ็ต", msg + "\n\nไฟล์เดิมจะถูกสำรองไว้ในโฟลเดอร์ backup",
                                   icon="warning"):
            return

        try:
            backup = reset_records(scope)
        except PermissionError:
            messagebox.showerror("รีเซ็ตการเข้าเรียน",
                                 "เขียนไฟล์ attendance.csv ไม่ได้ กรุณาปิดไฟล์ที่เปิดค้างอยู่ใน Excel ก่อน แล้วลองใหม่")
            return

        # ล้างสถานะในโปรแกรมให้ตรงกับไฟล์
        self.attendance.clear()
        self.recognizer.marked.clear()
        self.recognizer.streak.clear()
        self.recognizer.live.clear()
        self.absent_day = None          # ถ้าเลยเวลาเช็กชื่อแล้ว ระบบจะบันทึกคนที่ไม่มาเป็นขาดใหม่
        self.dirty = True

        note = f"\nสำรองไว้ที่: {os.path.basename(BACKUP_DIR)}/{os.path.basename(backup)}" if backup else ""
        extra = "\n\n(ถ้ายังมีคนยืนอยู่หน้ากล้อง ระบบจะเช็กชื่อคนนั้นใหม่อีกครั้ง)"
        if session_phase() == "closed":
            extra = ("\n\n(ตอนนี้เลยเวลาเช็กชื่อแล้ว ระบบจะบันทึกทุกคนเป็น \"ขาด\" อีกครั้ง "
                     "ถ้าจะทดลองใหม่ให้แก้ CLASS_START ในโค้ด)")
        messagebox.showinfo("รีเซ็ตการเข้าเรียน", "รีเซ็ตเรียบร้อยแล้ว" + note + extra)

    # ───── รับเหตุการณ์จาก thread ต่าง ๆ ─────
    def _poll_events(self):
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            kind = ev[0]
            if kind == "loaded":
                self.registered = list(ev[1])
                self.roster_ready = True
                self.dirty = True
            elif kind == "cam":
                self.connecting = False
                self.cam_btn.configure(state="normal")
                if ev[1] is None:
                    self.cam_btn.configure(text="เริ่มกล้อง")
                    messagebox.showerror(
                        "เชื่อมต่อกล้องไม่สำเร็จ",
                        "ตรวจสอบช่อง \"แหล่งภาพ\"\n"
                        "- กล้องโน้ตบุ๊ก: ใส่เลข 0 (ลอง 1, 2 ถ้ามีหลายกล้อง) และปิดโปรแกรมอื่นที่ใช้กล้องอยู่\n"
                        "  (Windows: Settings > Privacy > Camera ต้องอนุญาตแอปเดสก์ท็อป)\n"
                        "- IP Webcam: ใส่ URL และให้มือถือกับคอมพิวเตอร์ต่อ Wi-Fi เดียวกัน")
                else:
                    self.camera = ev[1]
                    self.recognizer.cam = self.camera
                    self.cam_btn.configure(text="หยุดกล้อง", fg_color=RED, hover_color="#dc2626")
            elif kind == "checkin":
                _, name, t, status = ev
                self.attendance[name] = (t, status)
                self.dirty = True
                if status == ST_LATE:
                    self.show_toast(f"มาสาย: {name}", AMBER, 3.0)
                else:
                    self.show_toast(f"เช็กชื่อสำเร็จ: {name}", GREEN, 3.0)
                if winsound is not None:
                    try:
                        winsound.MessageBeep(winsound.MB_OK)
                    except Exception:
                        pass
            elif kind == "registered":
                name = ev[1]
                self.registered.append(name)
                self.dirty = True
                self.show_toast(f"ลงทะเบียนสำเร็จ: {name}", ACCENT, 3.0)
                self._reg_finish()
            elif kind == "register_fail":
                self._reg_finish()
                messagebox.showwarning("เพิ่มบุคคลใหม่", ev[1])
            elif kind == "ai":
                self.title("ระบบเช็กชื่อเข้าเรียนด้วยใบหน้า" +
                           ("   •   AI ตรวจคนจริง: เปิด" if ev[1] else "   •   AI ตรวจคนจริง: ปิด"))
            elif kind == "warn":
                messagebox.showwarning("แจ้งเตือน", ev[1])
            elif kind == "error":
                self.show_toast(ev[1], RED, 5.0)

    # ───── อัปเดตรายชื่อ ─────
    def _render_lists(self):
        for frame in (self.list_in, self.list_out, self.list_abs):
            for child in frame.winfo_children():
                child.destroy()

        present = sorted(((n, t, s) for n, (t, s) in self.attendance.items() if s != ST_ABSENT),
                         key=lambda x: x[1], reverse=True)
        absent = sorted(n for n, (t, s) in self.attendance.items() if s == ST_ABSENT)
        waiting = sorted(n for n in self.registered if n not in self.attendance)

        if present:
            for name, t, s in present:
                color = AMBER if s == ST_LATE else GREEN
                self._row(self.list_in, name, f"{t}  •  {s}", color)
        else:
            self._empty(self.list_in, "ยังไม่มีใครเช็กชื่อวันนี้")

        if waiting:
            if self.phase == "late":
                for name in waiting:
                    self._row(self.list_out, name, "สายแล้ว", AMBER)
            else:
                for name in waiting:
                    self._row(self.list_out, name, "ยังไม่มา", MUTED)
        elif self.registered:
            self._empty(self.list_out, "มาครบทุกคนแล้ว" if not absent else "ไม่มีคนค้างแล้ว")
        else:
            self._empty(self.list_out, "ยังไม่มีผู้ลงทะเบียน\nกด \"เพิ่มบุคคลใหม่\"")

        if absent:
            for name in absent:
                self._row(self.list_abs, name, "ขาดเรียน", PURPLE)
        else:
            self._empty(self.list_abs, "ยังไม่มีผู้ขาดเรียน")

        self.stat_total.configure(text=str(len(self.registered)))
        self.stat_present.configure(text=str(len(present)))
        self.stat_wait.configure(text=str(len(waiting)))
        self.stat_absent.configure(text=str(len(absent)))

    def _row(self, parent, name, right_text, right_color):
        row = ctk.CTkFrame(parent, fg_color=ROW, corner_radius=10)
        row.pack(fill="x", pady=3, padx=2)
        row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(row, text=name, font=self.f_body, text_color=TEXT, anchor="w") \
            .grid(row=0, column=0, sticky="w", padx=12, pady=8)
        ctk.CTkLabel(row, text=right_text, font=self.f_small, text_color=right_color) \
            .grid(row=0, column=1, padx=12)

    def _empty(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=self.f_body, text_color=MUTED, justify="center") \
            .pack(pady=30)

    # ───── วาดภาพจากกล้อง ─────
    def _tick(self):
        try:
            self.recognizer.update()
            self._poll_events()
            if self.dirty:
                self.dirty = False
                self._render_lists()

            if self.connecting:
                self._set_status("connecting")
            elif self.camera is not None:
                lost = time.time() - self.camera.stamp > 3
                self._set_status("lost" if lost else "live")
            else:
                self._set_status("idle")

            self._draw_video()
        except Exception:
            traceback.print_exc()
        finally:
            self.after(30, self._tick)

    def _draw_video(self):
        cam = self.camera
        if cam is None:
            return
        frame = cam.frame
        if frame is None or frame is self.last_frame:
            return
        self.last_frame = frame

        bw, bh = self.video_box.winfo_width(), self.video_box.winfo_height()
        if bw < 50 or bh < 50:
            return
        h, w = frame.shape[:2]
        scale = min(bw / w, bh / h)
        dw, dh = max(int(w * scale), 1), max(int(h * scale), 1)

        img = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil, "RGBA")

        k = scale / self.recognizer.send_scale     # แปลงพิกัดจากภาพที่ส่งไปจำ -> ภาพที่แสดง
        banners = []
        font = pil_font(max(16, int(dh * 0.034)), bold=True)
        for (top, right, bottom, left), name, dist, info, prompt, metrics, susp in self.recognizer.get():
            if name == "Unknown":
                color, text = RED, "ไม่รู้จัก"
            elif info is None and prompt == FAR_PROMPT:
                color, text = AMBER, f"{name}  อยู่ไกลเกินไป"
            elif info is None and prompt == FAKE_PROMPT:
                color, text = RED, f"{name}  ไม่ใช่คนจริง"
            elif info is None:
                color, text = ACCENT, (f"{name}  กำลังตรวจสอบ" if prompt else name)
            elif info[1] == ST_ABSENT:
                color, text = PURPLE, f"{name}  ขาด"
            elif info[1] == ST_LATE:
                color, text = AMBER, f"{name}  มาสาย"
            else:
                color, text = GREEN, f"{name}  เช็กชื่อแล้ว"
            if prompt:
                banners.append(f"{name}: {prompt}")
            rgb = ImageColor.getrgb(color)
            l, t, r, b = left * k, top * k, right * k, bottom * k
            draw.rounded_rectangle([l, t, r, b], radius=12, outline=rgb, width=3)

            x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
            tw, th = x1 - x0, y1 - y0
            tag_top = t - th - 16
            if tag_top < 0:
                tag_top = b
            draw.rounded_rectangle([l, tag_top, l + tw + 20, tag_top + th + 16], radius=8, fill=rgb + (225,))
            draw.text((l + 10, tag_top + 6 - y0), text, font=font, fill=(255, 255, 255))

            if DEBUG_LIVENESS:
                dbg = f"sus {susp[0]:.2f}"
                if susp[1] is not None:
                    dbg += f"  AI real {susp[1]:.2f}"
                if metrics is not None:
                    dbg += f"  yaw {metrics[0]:.2f}  ear {metrics[1]:.2f}"
                draw.text((l, b + 6), dbg, font=pil_font(14), fill=(255, 255, 255))

        if banners:                                       # ข้อความบอกให้ทำท่า (ด้านบนของภาพ)
            bfont = pil_font(max(20, int(dh * 0.045)), bold=True)
            by = 16
            for line in banners:
                x0, y0, x1, y1 = draw.textbbox((0, 0), line, font=bfont)
                tw, th = x1 - x0, y1 - y0
                bx0 = (dw - tw) / 2 - 24
                draw.rounded_rectangle([bx0, by, bx0 + tw + 48, by + th + 28], radius=16,
                                       fill=(17, 26, 46, 235),
                                       outline=ImageColor.getrgb(RED if line.endswith(FAKE_PROMPT) else ACCENT), width=3)
                draw.text((bx0 + 24, by + 14 - y0), line, font=bfont, fill=(255, 255, 255))
                by += th + 40

        if self.toast is not None:
            text, color, until = self.toast
            if time.time() < until:
                tfont = pil_font(max(20, int(dh * 0.045)), bold=True)
                x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=tfont)
                tw, th = x1 - x0, y1 - y0
                bx0 = (dw - tw) / 2 - 24
                by0 = dh - th - 60
                draw.rounded_rectangle([bx0, by0, bx0 + tw + 48, by0 + th + 28], radius=16,
                                       fill=ImageColor.getrgb(color) + (235,))
                draw.text((bx0 + 24, by0 + 14 - y0), text, font=tfont, fill=(255, 255, 255))
            else:
                self.toast = None

        if self.photo is not None and self.photo.width() == dw and self.photo.height() == dh:
            self.photo.paste(pil)           # ใช้รูปเดิมซ้ำ เร็วกว่าสร้างใหม่ทุกเฟรม
        else:
            self.photo = ImageTk.PhotoImage(pil)
            self.video_label.configure(image=self.photo, text="")

    # ───── ปิดโปรแกรม ─────
    def on_close(self):
        self.recognizer.stop()
        if self.camera is not None:
            self.camera.stop()
        self.destroy()


if __name__ == "__main__":
    mp.freeze_support()
    os.makedirs(KNOWN_DIR, exist_ok=True)
    App().mainloop()