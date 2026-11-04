"""
patch_min_face.py  -  เพิ่มเงื่อนไข "ขนาดใบหน้าขั้นต่ำ" ให้ attendance_app.py

วิธีใช้
  1) วางไฟล์นี้ไว้โฟลเดอร์เดียวกับ attendance_app.py
  2) รัน:  python patch_min_face.py
  3) สคริปต์จะสำรองไฟล์เดิมเป็น attendance_app.py.bak แล้วแก้ไฟล์ให้
     (ค่าอื่นที่คุณแก้ไว้ เช่น CLASS_START จะไม่ถูกแตะ)

ถ้าสคริปต์แจ้งว่าหาจุดแก้ไม่เจอ แปลว่าคุณแก้โค้ดบริเวณนั้นไว้ ไฟล์จะไม่ถูกแก้เลย
"""
import os
import shutil
import sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance_app.py")

with open(PATH, encoding="utf-8", newline="") as f:
    src = f.read()

if "MIN_FACE_PX" in src:
    sys.exit("ไฟล์นี้ถูกแพตช์แล้ว (พบ MIN_FACE_PX) ไม่ต้องทำซ้ำ")

nl = "\r\n" if "\r\n" in src else "\n"

EDITS = []


def edit(label, old, new):
    EDITS.append((label, old.replace("\n", nl), new.replace("\n", nl)))


# 1) ค่าตั้งต้น
edit(
    "ค่าตั้งต้น MIN_FACE_PX",
    'FAKE_PROMPT = "ตรวจพบภาพบนจอ/กระดาษ ไม่สามารถเช็กชื่อได้"',
    'FAKE_PROMPT = "ตรวจพบภาพบนจอ/กระดาษ ไม่สามารถเช็กชื่อได้"\n'
    '\n'
    '# ── ขนาดใบหน้าขั้นต่ำ: กันภาพบนจอที่ถือไกลจนตรวจกรอบจอ/ลายจอไม่ได้ ──\n'
    '# ใบหน้าต้องกว้างอย่างน้อยเท่านี้ (พิกเซลบนภาพที่ส่งไปจำ ซึ่งกว้างไม่เกิน SEND_WIDTH) ถึงจะเริ่มเช็กชื่อได้\n'
    '# เล็กกว่านี้ = ขึ้นข้อความให้เข้าใกล้กล้อง (คนยืนห่างเกินไปก็โดนด้วย ถ้าเช็กยากให้ลดค่า / 0 = ปิดเงื่อนไขนี้)\n'
    'MIN_FACE_PX = 110\n'
    'FAR_PROMPT = "เข้ามาใกล้กล้องอีกนิด"',
)

# 2) ฟังก์ชันเช็กขนาดหน้า (วางก่อน _verdict)
edit(
    "ฟังก์ชัน _too_small",
    '    def _verdict(self, n):',
    '    @staticmethod\n'
    '    def _too_small(loc):\n'
    '        """ใบหน้าเล็ก/ไกลเกินไปหรือไม่ (loc = top, right, bottom, left บนภาพที่ส่งไปจำ)"""\n'
    '        return MIN_FACE_PX > 0 and (loc[1] - loc[3]) < MIN_FACE_PX\n'
    '\n'
    '    def _verdict(self, n):',
)

# 3) ตัดใบหน้าที่เล็กเกินไปออกจากรายชื่อที่จะเช็กชื่อ
edit(
    "คัดใบหน้าเล็กออกจาก names_now",
    '        for i, (_, n, _, _, _) in enumerate(faces):\n'
    '            if n != "Unknown" and n not in names_now:\n'
    '                names_now[n] = i\n',
    '        far = {}                                 # ใบหน้าที่เล็ก/ไกลเกินไป: ลำดับ -> ชื่อ\n'
    '        for i, (loc, n, _, _, _) in enumerate(faces):\n'
    '            if n == "Unknown" or n in names_now:\n'
    '                continue\n'
    '            if self._too_small(loc):\n'
    '                far[i] = n\n'
    '                continue\n'
    '            names_now[n] = i\n',
)

# 4) ให้ขึ้นข้อความเตือนกับใบหน้าที่ไกลเกินไป (ยกเว้นคนที่เช็กชื่อไปแล้ว)
edit(
    "ข้อความเตือนใบหน้าไกล",
    '        taken = set()',
    '        taken = set()\n'
    '        for i, n in far.items():\n'
    '            if n not in self.marked:\n'
    '                shown[i] = (n, FAR_PROMPT)',
)

# 5) ระหว่างรอกระพริบตา ถ้าถอยห่างกล้อง ให้ยกเลิกและเริ่มใหม่
edit(
    "ยกเลิกการพิสูจน์ตัวเมื่อถอยห่าง",
    '            loc, name, _, metrics, _ = faces[i]\n'
    '            st["loc"], st["seen"] = loc, now\n',
    '            loc, name, _, metrics, _ = faces[i]\n'
    '            st["loc"], st["seen"] = loc, now\n'
    '\n'
    '            if self._too_small(loc):                     # ถอยห่างกล้องระหว่างพิสูจน์ตัว -> ยกเลิก\n'
    '                del self.live[n]\n'
    '                shown[i] = (n, FAR_PROMPT)\n'
    '                continue\n',
)

# 6) ป้ายบนภาพ
edit(
    "ป้ายบนภาพ",
    '            elif info is None and prompt == FAKE_PROMPT:',
    '            elif info is None and prompt == FAR_PROMPT:\n'
    '                color, text = AMBER, f"{name}  อยู่ไกลเกินไป"\n'
    '            elif info is None and prompt == FAKE_PROMPT:',
)

# ตรวจว่าทุกจุดหาเจอ "พอดีหนึ่งที่" ก่อนแก้จริง
problems = []
for label, old, _ in EDITS:
    c = src.count(old)
    if c != 1:
        problems.append(f"  - {label}: พบ {c} ที่ (ต้องพบ 1 ที่)")
if problems:
    sys.exit("หาจุดแก้ไม่ตรง ยังไม่ได้แก้ไฟล์ใด ๆ:\n" + "\n".join(problems))

out = src
for label, old, new in EDITS:
    out = out.replace(old, new, 1)

shutil.copyfile(PATH, PATH + ".bak")
with open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(out)

print("แพตช์สำเร็จ  (สำรองไฟล์เดิมไว้ที่ attendance_app.py.bak)")
print("ค่าที่ปรับได้: MIN_FACE_PX ใกล้ ๆ ค่าตั้งต้นของไฟล์ (ตอนนี้ = 110)")