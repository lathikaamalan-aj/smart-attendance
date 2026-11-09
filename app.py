from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import os
import re
from datetime import datetime
import cv2
from PIL import Image
import base64
import numpy as np
import json
import hmac
import threading
import webbrowser
from datetime import timedelta, timezone

# Load variables from a local .env file (ignored if python-dotenv is missing).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Turso is called over its HTTP API with `requests` (pure Python, no wheels needed).
import requests

TURSO_URL = os.environ.get("TURSO_DATABASE_URL", "").strip()
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "").strip()
USE_TURSO = bool(TURSO_URL and TURSO_TOKEN)
TURSO_HTTP = TURSO_URL.replace("libsql://", "https://", 1).rstrip("/")

IntegrityError = sqlite3.IntegrityError
OperationalError = sqlite3.OperationalError

# Render runs in UTC; attendance periods are in local (India) time by default.
TZ_OFFSET_MINUTES = int(os.environ.get("TZ_OFFSET_MINUTES", "330"))


def now_local():
    return datetime.now(timezone(timedelta(minutes=TZ_OFFSET_MINUTES))).replace(tzinfo=None)


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Key the kiosk page sends so it can scan faces without an admin login.
KIOSK_KEY = os.environ.get("KIOSK_KEY", "change-this-key")

# Re-read template files when they change (needed because debug=False).
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# On Render, attach a Persistent Disk (e.g. /var/data) and set DATA_DIR=/var/data
# so face photos + trained model survive redeploys.
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE = os.path.join(DATA_DIR, "attendance.db")

FACE_DIR = os.path.join(DATA_DIR, "faces")
os.makedirs(FACE_DIR, exist_ok=True)

# =========================================================
# FACE CASCADE
# =========================================================

FACE_CASCADE_PATH = os.path.join(
    BASE_DIR, "haarcascades", "haarcascade_frontalface_default.xml"
)

if not os.path.isfile(FACE_CASCADE_PATH):
    raise FileNotFoundError(f"Face cascade file not found: {FACE_CASCADE_PATH}")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError(f"Failed to load face cascade: {FACE_CASCADE_PATH}")

print("FACE CASCADE LOADED:", FACE_CASCADE_PATH)

# =========================================================
# DATABASE CONNECTION (Turso or local SQLite)
# =========================================================


class Row(dict):
    """Behaves like sqlite3.Row: row["col"], row[0], dict(row)."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


_http = requests.Session()


def _to_arg(v):
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    return {"type": "text", "value": str(v)}


def _from_val(v):
    t = v.get("type")
    if t == "null":
        return None
    if t == "integer":
        return int(v["value"])
    if t == "float":
        return float(v["value"])
    if t == "blob":
        return base64.b64decode(v.get("base64", "") + "==")
    return v.get("value")


class _Cursor:
    def __init__(self, cols, rows, lastrowid=None):
        self.cols = cols
        self.rows = rows
        self.lastrowid = lastrowid
        self.pos = 0

    def fetchone(self):
        if self.pos >= len(self.rows):
            return None
        row = self.rows[self.pos]
        self.pos += 1
        return Row(zip(self.cols, row))

    def fetchall(self):
        out = [Row(zip(self.cols, r)) for r in self.rows[self.pos:]]
        self.pos = len(self.rows)
        return out


class _Conn:
    """Turso over HTTP. Each statement is sent (and saved) immediately."""

    def execute(self, sql, params=()):
        body = {"requests": [
            {"type": "execute",
             "stmt": {"sql": sql, "args": [_to_arg(p) for p in params]}},
            {"type": "close"},
        ]}
        try:
            resp = _http.post(
                TURSO_HTTP + "/v2/pipeline",
                json=body,
                headers={"Authorization": "Bearer " + TURSO_TOKEN},
                timeout=30,
            )
        except requests.RequestException as e:
            raise sqlite3.OperationalError("Cannot reach Turso: " + str(e))

        if resp.status_code != 200:
            raise sqlite3.OperationalError(
                f"Turso HTTP {resp.status_code}: {resp.text[:200]}"
            )

        first = resp.json()["results"][0]

        if first.get("type") == "error":
            msg = first["error"].get("message", "Turso error")
            if "constraint" in msg.lower():
                raise sqlite3.IntegrityError(msg)
            raise sqlite3.OperationalError(msg)

        result = first["response"]["result"]
        cols = [c["name"] for c in result.get("cols", [])]
        rows = [[_from_val(v) for v in r] for r in result.get("rows", [])]
        return _Cursor(cols, rows, result.get("last_insert_rowid"))

    def executescript(self, script):
        for stmt in script.split(";"):
            if stmt.strip():
                self.execute(stmt)

    def commit(self):
        pass  # every statement is already committed

    def rollback(self):
        pass

    def close(self):
        pass


def get_db_connection():
    if USE_TURSO:
        return _Conn()

    # Local fallback (no Turso env vars set)
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_student_contact_columns():
    """Add email/phone columns to older students tables if missing."""
    db = get_db_connection()
    try:
        columns = {row["name"] for row in db.execute("PRAGMA table_info(students)").fetchall()}
        if "email" not in columns:
            db.execute("ALTER TABLE students ADD COLUMN email TEXT")
        if "phone" not in columns:
            db.execute("ALTER TABLE students ADD COLUMN phone TEXT")
        db.commit()
    finally:
        db.close()


def get_face_folder(student_id):
    folder = os.path.join(FACE_DIR, str(student_id))
    os.makedirs(folder, exist_ok=True)
    return folder

# =========================================================
# FACE RECOGNITION MODEL
# =========================================================

TRAINER_PATH = os.path.join(DATA_DIR, "trainer.yml")
FACE_LABELS_PATH = os.path.join(DATA_DIR, "face_labels.json")

# Same size + lighting normalisation for training AND recognition
FACE_SIZE = (200, 200)

# LBPH: lower value = better match. Raise if real students show "Unknown face",
# lower if wrong students get matched.
CONFIDENCE_THRESHOLD = 75

# Smaller minSize so faces are detected a bit farther from the webcam
FACE_MIN_SIZE = (60, 60)


def preprocess_face(gray_face):
    face = cv2.resize(gray_face, FACE_SIZE)
    return cv2.equalizeHist(face)


def train_face_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    next_label = 1

    if not os.path.exists(FACE_DIR):
        return False, "Face folder not found."

    for student_id in sorted(os.listdir(FACE_DIR)):
        student_folder = os.path.join(FACE_DIR, student_id)

        if not os.path.isdir(student_folder):
            continue

        image_files = [
            f for f in os.listdir(student_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            continue

        current_label = next_label
        next_label += 1
        label_map[str(current_label)] = str(student_id)

        for image_name in image_files:
            image_path = os.path.join(student_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            faces.append(preprocess_face(image))
            labels.append(current_label)

    if not faces:
        return False, "No face images found."

    recognizer.train(faces, np.array(labels, dtype=np.int32))
    recognizer.write(TRAINER_PATH)

    with open(FACE_LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=4)

    return True, f"Face model trained with {len(faces)} images."

# =========================================================
# LOGIN
# =========================================================

@app.route("/")
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        try:
            db = get_db_connection()

            user = db.execute("""
                SELECT *
                FROM users
                WHERE username = ?
                AND password = ?
            """, (username, password)).fetchone()

            db.close()

            if user:
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                session["role"] = user["role"]

                if user["role"] == "admin":
                    redirect_url = "/admin"
                elif user["role"] in ("representative", "faculty"):
                    redirect_url = "/staff"
                else:
                    redirect_url = "/student"

                return jsonify({"success": True, "redirect": redirect_url})

            return jsonify({
                "success": False,
                "message": "Invalid username or password."
            })

        except Exception as e:
            print("LOGIN ERROR:", e)
            return jsonify({
                "success": False,
                "message": "Database error: " + str(e)
            }), 500

    return render_template("login.html")


# =========================================================
# STUDENT SELF REGISTRATION
# =========================================================

@app.route("/register", methods=["POST"])
def register():
    data = request.form

    name = data.get("name", "").strip()
    student_id = data.get("student_id", "").strip()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    year = data.get("year", "").strip()
    department = data.get("department", "").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()

    if not all([name, student_id, username, password, year, department, email, phone]):
        return jsonify({"success": False, "message": "Please fill all fields."}), 400

    if not re.fullmatch(r"\d{12}", student_id):
        return jsonify({
            "success": False,
            "message": "Register number must contain exactly 12 digits."
        }), 400

    if len(password) < 6:
        return jsonify({
            "success": False,
            "message": "Password must contain at least 6 characters."
        }), 400

    if not re.fullmatch(r"\S+@\S+\.\S+", email):
        return jsonify({
            "success": False,
            "message": "Please enter a valid email address."
        }), 400

    if not re.fullmatch(r"\+?\d{10,13}", phone):
        return jsonify({
            "success": False,
            "message": "Please enter a valid phone number."
        }), 400

    db = None
    try:
        ensure_student_contact_columns()
        db = get_db_connection()

        existing_student = db.execute(
            """
            SELECT student_id, username
            FROM students
            WHERE student_id = ? OR username = ?
            LIMIT 1
            """,
            (student_id, username)
        ).fetchone()

        if existing_student:
            message = (
                "Register number already exists."
                if existing_student["student_id"] == student_id
                else "Username already exists."
            )
            db.close()
            db = None
            return jsonify({"success": False, "message": message}), 400

        existing_user = db.execute(
            "SELECT id FROM users WHERE username = ? LIMIT 1",
            (username,)
        ).fetchone()

        if existing_user:
            db.close()
            db = None
            return jsonify({"success": False, "message": "Username already exists."}), 400

        db.execute(
            """
            INSERT INTO students
            (student_id, name, username, password, year, department, email, phone)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (student_id, name, username, password, year, department, email, phone)
        )

        db.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, password, "student")
        )

        db.commit()
        db.close()
        db = None

        session["face_enrollment_student_id"] = student_id
        session["face_enrollment_name"] = name

        return jsonify({
            "success": True,
            "message": "Account created successfully. Please complete face enrollment.",
            "redirect": "/face_enrollment"
        })

    except IntegrityError as e:
        if db is not None:
            try:
                db.rollback()
                db.close()
            except Exception:
                pass
        print("REGISTER INTEGRITY ERROR:", e)
        return jsonify({
            "success": False,
            "message": "Student ID or username already exists."
        }), 400

    except Exception as e:
        if db is not None:
            try:
                db.rollback()
                db.close()
            except Exception:
                pass
        print("REGISTER ERROR:", e)
        return jsonify({
            "success": False,
            "message": "Database error: " + str(e)
        }), 500


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/admin")
def admin():
    if session.get("role") != "admin":
        return "Access Denied"
    return render_template("admin.html")


# =========================================================
# ADMIN FACE REGISTRATION
# =========================================================

@app.route("/face_register")
def face_register():
    if session.get("role") != "admin":
        return "Access Denied", 403

    db = get_db_connection()
    try:
        students = db.execute(
            """
            SELECT student_id, name, year, department
            FROM students
            ORDER BY name
            """
        ).fetchall()
        return render_template("face_register.html", students=students)
    finally:
        db.close()


@app.route("/capture_face", methods=["POST"])
def capture_face():
    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403

    try:
        data = request.get_json(silent=True) or {}
        student_id = str(data.get("student_id", "")).strip()
        image_data = data.get("image", "")

        if not student_id or not image_data:
            return jsonify({
                "success": False,
                "message": "Student and image are required."
            }), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data, validate=True)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "Invalid image."}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=FACE_MIN_SIZE
        )

        if len(faces) == 0:
            return jsonify({
                "success": False,
                "message": "No face detected. Please look at the camera."
            }), 400

        x, y, w, h = faces[0]
        face = gray[y:y + h, x:x + w]
        student_folder = get_face_folder(student_id)

        existing = [
            f for f in os.listdir(student_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(existing) >= 20:
            return jsonify({
                "success": False,
                "message": "Maximum 20 face images already enrolled."
            }), 400

        sample_number = len(existing) + 1
        file_path = os.path.join(student_folder, f"{sample_number}.jpg")

        if not cv2.imwrite(file_path, face):
            return jsonify({
                "success": False,
                "message": "Failed to save face image."
            }), 500

        if sample_number >= 20:
            try:
                ok, msg = train_face_model()
                print("AUTO TRAIN:", ok, msg)
            except Exception as train_err:
                print("AUTO TRAIN ERROR:", train_err)

        return jsonify({
            "success": True,
            "message": f"Face sample {sample_number} captured.",
            "image_number": sample_number,
            "completed": sample_number >= 20
        })

    except Exception as e:
        print("FACE CAPTURE ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# =========================================================
# STUDENT FACE ENROLLMENT
# =========================================================

@app.route("/face_enrollment")
def face_enrollment():
    if not session.get("face_enrollment_student_id"):
        return redirect(url_for("login"))
    return render_template("face_enrollment.html")


@app.route("/api/face_enrollment/student", methods=["GET"])
def get_face_enrollment_student():
    student_id = session.get("face_enrollment_student_id")

    if not student_id:
        return jsonify({
            "success": False,
            "message": "No student is waiting for face enrollment."
        }), 401

    db = get_db_connection()
    try:
        student = db.execute(
            "SELECT student_id, name FROM students WHERE student_id = ?",
            (student_id,)
        ).fetchone()

        if not student:
            session.pop("face_enrollment_student_id", None)
            session.pop("face_enrollment_name", None)
            return jsonify({
                "success": False,
                "message": "Student not found."
            }), 404

        return jsonify({
            "success": True,
            "student_id": student["student_id"],
            "name": student["name"]
        })
    finally:
        db.close()


@app.route("/api/face_enrollment", methods=["POST"])
def api_face_enrollment():
    student_id = session.get("face_enrollment_student_id")

    if not student_id:
        return jsonify({
            "success": False,
            "message": "Face enrollment session not found. Please register again."
        }), 401

    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get("image", "")

        if not image_data:
            return jsonify({
                "success": False,
                "message": "No webcam image received."
            }), 400

        db = get_db_connection()
        try:
            student = db.execute(
                "SELECT student_id, name FROM students WHERE student_id = ?",
                (student_id,)
            ).fetchone()
        finally:
            db.close()

        if student is None:
            session.pop("face_enrollment_student_id", None)
            session.pop("face_enrollment_name", None)
            return jsonify({
                "success": False,
                "message": "Student not found."
            }), 404

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except Exception:
            return jsonify({
                "success": False,
                "message": "Invalid image data."
            }), 400

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "success": False,
                "message": "Invalid webcam image."
            }), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=FACE_MIN_SIZE
        )

        if len(faces) == 0:
            return jsonify({
                "success": False,
                "message": "No face detected. Keep your face inside the camera."
            }), 400

        if len(faces) > 1:
            return jsonify({
                "success": False,
                "message": "Multiple faces detected. Only one student should be in front of the camera."
            }), 400

        student_folder = get_face_folder(student_id)
        existing_files = [
            f for f in os.listdir(student_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(existing_files) >= 20:
            return jsonify({
                "success": False,
                "message": "Maximum 20 face images already enrolled."
            }), 400

        x, y, w, h = faces[0]
        face_image = gray[y:y + h, x:x + w]
        image_number = len(existing_files) + 1
        image_path = os.path.join(student_folder, f"{image_number}.jpg")

        if not cv2.imwrite(image_path, face_image):
            return jsonify({
                "success": False,
                "message": "Failed to save face image."
            }), 500

        completed = image_number >= 20

        if completed:
            try:
                ok, msg = train_face_model()
                print("AUTO TRAIN:", ok, msg)
            except Exception as train_err:
                print("AUTO TRAIN ERROR:", train_err)

            session.pop("face_enrollment_student_id", None)
            session.pop("face_enrollment_name", None)

        return jsonify({
            "success": True,
            "message": (
                "Enrollment completed successfully. 20 face images saved."
                if completed
                else f"Face image {image_number} captured successfully."
            ),
            "student_id": student["student_id"],
            "student_name": student["name"],
            "image_number": image_number,
            "total_images": image_number,
            "completed": completed
        })

    except Exception as e:
        print("FACE ENROLLMENT ERROR:", e)
        return jsonify({
            "success": False,
            "message": "Face enrollment error: " + str(e)
        }), 500


# =========================================================
# UPLOAD STUDENTS PAGE
# =========================================================

@app.route("/upload_students")
def upload_students():
    if session.get("role") != "admin":
        return "Access Denied"
    return render_template("upload_students.html")


# =========================================================
# UPLOAD CSV STUDENTS
# =========================================================

@app.route("/upload_students_csv", methods=["POST"])
def upload_students_csv():

    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403

    try:
        data = request.get_json()

        if not data or "rows" not in data:
            return jsonify({
                "success": False,
                "message": "No student data received."
            }), 400

        rows = data["rows"]

        if len(rows) < 2:
            return jsonify({
                "success": False,
                "message": "CSV file must contain student data."
            }), 400

        headers = [
            str(h).strip().lower().replace(" ", "_")
            for h in rows[0]
        ]

        required_headers = [
            "student_id", "name", "username", "password", "year", "department"
        ]

        if headers != required_headers:
            return jsonify({
                "success": False,
                "message":
                "Invalid CSV columns. Use: "
                "student_id, name, username, password, year, department"
            }), 400

        db = get_db_connection()

        added = 0
        skipped = 0
        errors = []

        for row_number, row in enumerate(rows[1:], start=2):

            if not row or not any(str(cell).strip() for cell in row):
                continue

            if len(row) < 6:
                skipped += 1
                errors.append(f"Row {row_number}: Not enough columns")
                continue

            student_id = str(row[0]).strip()
            name = str(row[1]).strip()
            username = str(row[2]).strip()
            password = str(row[3]).strip()
            year = str(row[4]).strip()
            department = str(row[5]).strip()

            if not all([student_id, name, username, password, year, department]):
                skipped += 1
                errors.append(f"Row {row_number}: Missing required value")
                continue

            try:
                db.execute("""
                    INSERT INTO students
                    (student_id, name, username, password, year, department)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (student_id, name, username, password, year, department))

                db.execute("""
                    INSERT INTO users (username, password, role)
                    VALUES (?, ?, ?)
                """, (username, password, "student"))

                added += 1

            except IntegrityError:
                skipped += 1
                errors.append(
                    f"Row {row_number}: Student ID or username already exists"
                )

        db.commit()
        db.close()

        return jsonify({
            "success": True,
            "added": added,
            "skipped": skipped,
            "errors": errors,
            "message": f"{added} student(s) added successfully."
        })

    except Exception as e:
        print("CSV UPLOAD ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# =========================================================
# ADD SINGLE STUDENT
# =========================================================

@app.route("/add_student", methods=["POST"])
def add_student():

    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403

    try:
        data = request.get_json()

        student_id = str(data.get("student_id", "")).strip()
        name = str(data.get("name", "")).strip()
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", "")).strip()
        year = str(data.get("year", "")).strip()
        department = str(data.get("department", "")).strip()

        if not all([student_id, name, username, password, year, department]):
            return jsonify({
                "success": False,
                "message": "Please fill all fields."
            }), 400

        db = get_db_connection()

        print("DATABASE USED:", TURSO_URL if USE_TURSO else DATABASE)

        existing_student = db.execute("""
            SELECT id FROM students WHERE student_id = ?
        """, (student_id,)).fetchone()

        if existing_student:
            db.close()
            return jsonify({
                "success": False,
                "message": "Student ID already exists."
            }), 400

        existing_user = db.execute("""
            SELECT id FROM users WHERE username = ?
        """, (username,)).fetchone()

        if existing_user:
            db.close()
            return jsonify({
                "success": False,
                "message": "Username already exists."
            }), 400

        db.execute("""
            INSERT INTO students
            (student_id, name, username, password, year, department)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (student_id, name, username, password, year, department))

        db.execute("""
            INSERT INTO users (username, password, role)
            VALUES (?, ?, ?)
        """, (username, password, "student"))

        db.commit()
        db.close()

        return jsonify({
            "success": True,
            "message": f"{name} added successfully."
        })

    except Exception as e:
        print("ADD STUDENT ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# =========================================================
# VIEW STUDENTS PAGE
# =========================================================

@app.route("/view_students")
def view_students():

    if session.get("role") != "admin":
        return "Access Denied"

    try:
        db = get_db_connection()

        students = db.execute("""
            SELECT id, student_id, name, username, year, department
            FROM students
            ORDER BY id ASC
        """).fetchall()

        db.close()

        students = [dict(student) for student in students]

        return render_template("view_students.html", students=students)

    except Exception as e:
        print("VIEW STUDENTS ERROR:", e)
        return "Database error: " + str(e)


# =========================================================
# API - GET ALL STUDENTS
# =========================================================

@app.route("/api/students")
def api_students():

    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403

    try:
        db = get_db_connection()

        students = db.execute("""
            SELECT id, student_id, name, username, year, department
            FROM students
            ORDER BY id ASC
        """).fetchall()

        db.close()

        return jsonify([dict(student) for student in students])

    except Exception as e:
        print("GET STUDENTS ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# =========================================================
# UPDATE STUDENT
# =========================================================

@app.route("/update_student", methods=["POST"])
def update_student():

    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403

    try:
        data = request.get_json()

        student_db_id = data.get("id")
        name = data.get("name", "").strip()
        username = data.get("username", "").strip()
        year = data.get("year", "").strip()
        department = data.get("department", "").strip()

        if not student_db_id:
            return jsonify({
                "success": False,
                "message": "Student ID is missing."
            }), 400

        if not name or not username or not year or not department:
            return jsonify({
                "success": False,
                "message": "Please fill all fields."
            }), 400

        db = get_db_connection()

        student = db.execute("""
            SELECT * FROM students WHERE id = ?
        """, (student_db_id,)).fetchone()

        if not student:
            db.close()
            return jsonify({
                "success": False,
                "message": "Student not found."
            }), 404

        old_username = student["username"]

        existing_user = db.execute("""
            SELECT * FROM users WHERE username = ?
        """, (username,)).fetchone()

        if existing_user and username != old_username:
            db.close()
            return jsonify({
                "success": False,
                "message": "Username already exists."
            }), 400

        db.execute("""
            UPDATE students
            SET name = ?, username = ?, year = ?, department = ?
            WHERE id = ?
        """, (name, username, year, department, student_db_id))

        db.execute("""
            UPDATE users
            SET username = ?
            WHERE username = ?
            AND role = 'student'
        """, (username, old_username))

        db.commit()
        db.close()

        return jsonify({
            "success": True,
            "message": f"{name} updated successfully."
        })

    except Exception as e:
        print("UPDATE STUDENT ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# =========================================================
# DELETE STUDENT
# =========================================================

@app.route("/delete_student", methods=["POST"])
def delete_student():

    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403

    try:
        data = request.get_json()
        student_db_id = data.get("id")

        if not student_db_id:
            return jsonify({
                "success": False,
                "message": "Student ID is missing."
            }), 400

        db = get_db_connection()

        student = db.execute("""
            SELECT * FROM students WHERE id = ?
        """, (student_db_id,)).fetchone()

        if not student:
            db.close()
            return jsonify({
                "success": False,
                "message": "Student not found."
            }), 404

        username = student["username"]
        name = student["name"]

        db.execute("DELETE FROM students WHERE id = ?", (student_db_id,))

        db.execute("""
            DELETE FROM users
            WHERE username = ?
            AND role = 'student'
        """, (username,))

        db.commit()
        db.close()

        return jsonify({
            "success": True,
            "message": f"{name} deleted successfully."
        })

    except Exception as e:
        print("DELETE STUDENT ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# =========================================================
# MANUAL ATTENDANCE APIs
# =========================================================

def normalize_year(value):
    year_map = {
        "1st year": "I",
        "2nd year": "II",
        "3rd year": "III",
        "4th year": "IV",
        "1st": "I",
        "2nd": "II",
        "3rd": "III",
        "4th": "IV",
        "i": "I",
        "ii": "II",
        "iii": "III",
        "iv": "IV"
    }
    value = (value or "").strip()
    return year_map.get(value.lower(), value)


_TIME_TOKEN = re.compile(r"(\d{1,2})[:.](\d{2})\s*([ap])?\.?m?\.?", re.IGNORECASE)


def period_start_minutes(period):
    """Minutes since midnight for the START of a period like '01:30 PM-02:20 PM'.

    Handles '9:00 AM - 9:50 AM', '09:00-09:50', '1:30-2:20' (college hours,
    so 1-7 without AM/PM means afternoon) and 24-hour values like '13:30'.
    Unparseable values sort last.
    """
    tokens = _TIME_TOKEN.findall(str(period or ""))
    if not tokens:
        return 10 ** 6

    h, m, mer = tokens[0]
    h, m, mer = int(h), int(m), mer.lower()
    end_mer = tokens[1][2].lower() if len(tokens) > 1 else ""
    end_h = int(tokens[1][0]) if len(tokens) > 1 else None

    if not mer and end_mer:
        # "11:30-12:20 PM" -> start is AM; "01:30-02:20 PM" -> start is PM
        if end_mer == "p":
            mer = "p" if (h % 12) <= (end_h % 12) else "a"
        else:
            mer = "a"

    if mer == "p":
        return (h % 12) * 60 + m + 720
    if mer == "a":
        return (h % 12) * 60 + m
    if h >= 13 or h == 0:          # 24-hour clock
        return h * 60 + m
    if h <= 7:                     # no AM/PM: 1-7 means afternoon
        return (h + 12) * 60 + m
    return h * 60 + m              # 8-12


def sort_periods(periods):
    return sorted(periods, key=lambda p: (period_start_minutes(p), str(p)))


def require_attendance_role():
    return session.get("role") in ["admin", "representative", "faculty"]


def kiosk_or_staff():
    """Allow admin/representative session OR a valid kiosk key header."""
    key = request.headers.get("X-Kiosk-Key", "")
    return require_attendance_role() or (
        bool(key) and hmac.compare_digest(key, KIOSK_KEY)
    )


@app.route("/get_students")
def get_students():
    """Return students for the selected year and department."""

    if not require_attendance_role():
        return jsonify({"success": False, "message": "Access Denied"}), 403

    year = request.args.get("year", "").strip()
    department = request.args.get("department", "").strip()

    if not year or not department:
        return jsonify({
            "success": False,
            "message": "Year and department are required."
        }), 400

    # Accept different year formats
    year_map = {
        "i": ["I", "1st", "1st Year", "1"],
        "ii": ["II", "2nd", "2nd Year", "2"],
        "iii": ["III", "3rd", "3rd Year", "3"],
        "iv": ["IV", "4th", "4th Year", "4"]
    }

    possible_years = year_map.get(year.lower(), [year])

    try:
        db = get_db_connection()

        placeholders = ",".join(["?"] * len(possible_years))

        rows = db.execute(
            f"""
            SELECT student_id, name
            FROM students
            WHERE LOWER(TRIM(year)) IN ({placeholders})
            AND LOWER(TRIM(department)) = LOWER(TRIM(?))
            ORDER BY name COLLATE NOCASE
            """,
            [y.lower() for y in possible_years] + [department]
        ).fetchall()

        db.close()

        return jsonify({
            "success": True,
            "students": [
                {"roll_no": row["student_id"], "name": row["name"]}
                for row in rows
            ]
        })

    except Exception as e:
        print("GET STUDENTS ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/manual_periods")
def manual_periods():
    """Return periods from the uploaded timetable for date + year + department."""
    if not require_attendance_role():
        return jsonify({"success": False, "message": "Access Denied"}), 403

    date_value = request.args.get("date", "").strip()
    year = normalize_year(request.args.get("year", ""))
    department = request.args.get("department", "").strip()

    if not date_value or not year or not department:
        return jsonify({
            "success": False,
            "message": "Date, year and department are required."
        }), 400

    try:
        day_name = datetime.strptime(date_value, "%Y-%m-%d").strftime("%A")
    except ValueError:
        return jsonify({"success": False, "message": "Invalid date."}), 400

    try:
        db = get_db_connection()
        subject_value = request.args.get("subject", "").strip()
        sql = """
            SELECT DISTINCT time
            FROM timetable
            WHERE UPPER(TRIM(year)) = UPPER(TRIM(?))
              AND LOWER(TRIM(department)) = LOWER(TRIM(?))
              AND LOWER(TRIM(day)) = LOWER(TRIM(?))
        """
        params = [year, department, day_name]
        if subject_value:
            sql += " AND LOWER(TRIM(subject)) = LOWER(TRIM(?))"
            params.append(subject_value)
        rows = db.execute(sql, params).fetchall()
        db.close()

        return jsonify({
            "success": True,
            "day": day_name,
            "periods": sort_periods([row["time"] for row in rows])
        })

    except Exception as e:
        print("MANUAL PERIODS ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/manual_subjects")
def manual_subjects():
    """Return subject(s) matching the exact selected date/year/department/time."""
    if not require_attendance_role():
        return jsonify({"success": False, "message": "Access Denied"}), 403

    date_value = request.args.get("date", "").strip()
    year = normalize_year(request.args.get("year", ""))
    department = request.args.get("department", "").strip()
    time_value = request.args.get("time", "").strip()

    if not date_value or not year or not department:
        return jsonify({
            "success": False,
            "message": "Date, year and department are required."
        }), 400

    try:
        day_name = datetime.strptime(date_value, "%Y-%m-%d").strftime("%A")
    except ValueError:
        return jsonify({"success": False, "message": "Invalid date."}), 400

    try:
        db = get_db_connection()
        sql = """
            SELECT DISTINCT subject
            FROM timetable
            WHERE UPPER(TRIM(year)) = UPPER(TRIM(?))
              AND LOWER(TRIM(department)) = LOWER(TRIM(?))
              AND LOWER(TRIM(day)) = LOWER(TRIM(?))
        """
        params = [year, department, day_name]
        if time_value:
            sql += " AND LOWER(TRIM(time)) = LOWER(TRIM(?))"
            params.append(time_value)
        sql += " ORDER BY subject COLLATE NOCASE"
        rows = db.execute(sql, params).fetchall()
        db.close()

        return jsonify({
            "success": True,
            "day": day_name,
            "subjects": [row["subject"] for row in rows]
        })

    except Exception as e:
        print("MANUAL SUBJECTS ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin_timetable")
def admin_timetable():
    """Full timetable for a year + department (admin / representative)."""
    if not require_attendance_role():
        return jsonify({"success": False, "message": "Access Denied"}), 403

    year = normalize_year(request.args.get("year", ""))
    department = request.args.get("department", "").strip()

    if not year or not department:
        return jsonify({
            "success": False,
            "message": "Year and department are required."
        }), 400

    db = None
    try:
        db = get_db_connection()
        rows = db.execute("""
            SELECT day, time, subject, faculty
            FROM timetable
            WHERE UPPER(TRIM(year)) = UPPER(TRIM(?))
              AND LOWER(TRIM(department)) = LOWER(TRIM(?))
        """, (year, department)).fetchall()

        day_order = ["monday", "tuesday", "wednesday", "thursday",
                     "friday", "saturday", "sunday"]

        def sort_key(r):
            d = str(r["day"] or "").strip().lower()
            return (day_order.index(d) if d in day_order else 9,
                    period_start_minutes(r["time"]))

        data = [dict(r) for r in sorted(rows, key=sort_key)]
        return jsonify({"success": True, "timetable": data})

    except Exception as e:
        print("ADMIN TIMETABLE ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500

    finally:
        if db is not None:
            db.close()


@app.route("/save_attendance", methods=["POST"])
def save_attendance():
    """Save the complete manual attendance roster."""
    if not require_attendance_role():
        return jsonify({"success": False, "message": "Access Denied"}), 403

    data = request.get_json(silent=True) or {}
    date_value = str(data.get("date", "")).strip()
    year = normalize_year(data.get("year", ""))
    department = str(data.get("department", "")).strip()
    period = str(data.get("period", "")).strip()
    subject = str(data.get("subject", "")).strip()
    records = data.get("records", [])

    if not all([date_value, year, department, period, subject]):
        return jsonify({
            "success": False,
            "message": "Date, year, department, period and subject are required."
        }), 400

    if not isinstance(records, list) or not records:
        return jsonify({
            "success": False,
            "message": "Attendance records are required."
        }), 400

    try:
        datetime.strptime(date_value, "%Y-%m-%d")
    except ValueError:
        return jsonify({"success": False, "message": "Invalid date."}), 400

    try:
        db = get_db_connection()

        # Make sure the attendance table exists for an existing project DB.
        db.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                subject TEXT NOT NULL,
                status TEXT NOT NULL,
                marked_at TEXT
            )
        """)

        # Verify that the selected period really belongs to the selected class/date.
        day_name = datetime.strptime(date_value, "%Y-%m-%d").strftime("%A")
        timetable_match = db.execute("""
            SELECT 1
            FROM timetable
            WHERE UPPER(TRIM(year)) = UPPER(TRIM(?))
              AND LOWER(TRIM(department)) = LOWER(TRIM(?))
              AND LOWER(TRIM(day)) = LOWER(TRIM(?))
              AND LOWER(TRIM(time)) = LOWER(TRIM(?))
              AND LOWER(TRIM(subject)) = LOWER(TRIM(?))
            LIMIT 1
        """, (year, department, day_name, period, subject)).fetchone()

        if not timetable_match:
            db.close()
            return jsonify({
                "success": False,
                "message": "Selected period and subject do not match the uploaded timetable."
            }), 400

        now = now_local().strftime("%Y-%m-%d %H:%M:%S")
        saved = 0
        skipped = 0

        for record in records:
            student_id = str(record.get("roll_no", "")).strip()
            status = str(record.get("status", "")).strip().lower()

            if not student_id or status not in ["present", "absent"]:
                skipped += 1
                continue

            student = db.execute("""
                SELECT student_id, year
                FROM students
                WHERE student_id = ?
                  AND LOWER(TRIM(department)) = LOWER(TRIM(?))
                LIMIT 1
            """, (student_id, department)).fetchone()

            # Students may store the year as "1st Year" while the form sends "I",
            # so compare the normalized values instead of the raw text.
            if not student or normalize_year(student["year"]).upper() != year.upper():
                skipped += 1
                continue

            existing = db.execute("""
                SELECT id
                FROM attendance
                WHERE student_id = ?
                  AND date = ?
                  AND time = ?
                  AND subject = ?
                LIMIT 1
            """, (student_id, date_value, period, subject)).fetchone()

            if existing:
                db.execute("""
                    UPDATE attendance
                    SET status = ?, marked_at = ?
                    WHERE id = ?
                """, (status.capitalize(), now, existing["id"]))
            else:
                db.execute("""
                    INSERT INTO attendance
                    (student_id, date, time, subject, status, marked_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (student_id, date_value, period, subject, status.capitalize(), now))

            saved += 1

        db.commit()
        db.close()

        return jsonify({
            "success": True,
            "saved": saved,
            "skipped": skipped,
            "message": f"Attendance saved for {saved} student(s)."
        })

    except Exception as e:
        print("SAVE ATTENDANCE ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# =========================================================
# UPLOAD TIMETABLE PAGE
# =========================================================

@app.route("/upload_timetable")
def upload_timetable():
    if session.get("role") != "admin":
        return "Access Denied", 403
    return render_template("upload_timetable.html")


# =========================================================
# UPLOAD TIMETABLE CSV
# =========================================================

@app.route("/upload_timetable_csv", methods=["POST"])
def upload_timetable_csv():

    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403

    try:
        data = request.get_json(silent=True)

        if not data or "rows" not in data:
            return jsonify({
                "success": False,
                "message": "No timetable data received."
            }), 400

        rows = data["rows"]

        if len(rows) < 2:
            return jsonify({
                "success": False,
                "message": "Timetable file must contain timetable data."
            }), 400

        headers = [
            str(h).strip().lower().replace(" ", "_")
            for h in rows[0]
        ]

        valid_headers_with_id = [
            "id", "year", "department", "day", "time", "subject", "faculty"
        ]
        valid_headers_without_id = [
            "year", "department", "day", "time", "subject", "faculty"
        ]

        if headers == valid_headers_with_id:
            has_id = True
        elif headers == valid_headers_without_id:
            has_id = False
        else:
            return jsonify({
                "success": False,
                "message": (
                    "Invalid timetable columns. Use: "
                    "id, year, department, day, time, subject, faculty"
                )
            }), 400

        db = get_db_connection()
        added = 0
        skipped = 0
        errors = []

        valid_days = {
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"
        }

        for row_number, row in enumerate(rows[1:], start=2):

            if not row or not any(str(cell).strip() for cell in row):
                continue

            start = 1 if has_id else 0

            if len(row) < (7 if has_id else 6):
                skipped += 1
                errors.append(f"Row {row_number}: Not enough columns")
                continue

            year = str(row[start]).strip()
            department = str(row[start + 1]).strip()
            day = str(row[start + 2]).strip().capitalize()
            time_value = str(row[start + 3]).strip()
            subject = str(row[start + 4]).strip()
            faculty = str(row[start + 5]).strip()

            if day.lower() not in valid_days:
                skipped += 1
                errors.append(f"Row {row_number}: Invalid day '{day}'")
                continue

            if not all([year, department, day, time_value, subject]):
                skipped += 1
                errors.append(
                    f"Row {row_number}: Year, department, day, time and subject are required"
                )
                continue

            try:
                db.execute("""
                    INSERT INTO timetable
                    (year, department, day, time, subject, faculty)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (year, department, day, time_value, subject, faculty))
                added += 1

            except IntegrityError as e:
                skipped += 1
                errors.append(f"Row {row_number}: {str(e)}")

        db.commit()
        db.close()

        return jsonify({
            "success": True,
            "added": added,
            "skipped": skipped,
            "errors": errors,
            "message": f"{added} timetable row(s) uploaded successfully."
        })

    except Exception as e:
        print("TIMETABLE UPLOAD ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# =========================================================
# REPRESENTATIVE
# =========================================================

@app.route("/representative")
def representative():
    return redirect(url_for("staff_home"))


# ------------------------------------------------------------
# /student - renders the dashboard for the logged-in student
# ------------------------------------------------------------
@app.route("/student")
def student():
    if "username" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "student":
        return redirect(url_for("login"))

    conn = get_db_connection()

    # Support databases created before the gender column was added.
    columns = [row["name"] for row in conn.execute("PRAGMA table_info(students)").fetchall()]

    if "gender" in columns:
        student_row = conn.execute(
            """
            SELECT id, student_id, name, username, year, department, gender
            FROM students
            WHERE username = ?
            """,
            (session["username"],)
        ).fetchone()
    else:
        student_row = conn.execute(
            """
            SELECT id, student_id, name, username, year, department,
                   NULL AS gender
            FROM students
            WHERE username = ?
            """,
            (session["username"],)
        ).fetchone()

    conn.close()

    if student_row is None:
        return redirect(url_for("login"))

    return render_template("user.html", student=student_row)


# ------------------------------------------------------------
# /api/timetable?date=YYYY-MM-DD
# ------------------------------------------------------------

@app.route("/api/timetable")
def api_timetable():

    if session.get("role") != "student":
        return jsonify([]), 403

    date_value = request.args.get("date", "").strip()

    if not date_value:
        return jsonify([]), 400

    try:
        day_name = datetime.strptime(date_value, "%Y-%m-%d").strftime("%A")
    except ValueError:
        return jsonify([]), 400

    db = get_db_connection()

    try:
        student = db.execute("""
            SELECT student_id, year, department
            FROM students
            WHERE username = ?
            LIMIT 1
        """, (session.get("username"),)).fetchone()

        if not student:
            return jsonify([]), 404

        year = normalize_year(student["year"])
        department = str(student["department"] or "").strip()

        rows = db.execute("""
            SELECT time, subject, faculty
            FROM timetable
            WHERE UPPER(TRIM(year)) = UPPER(TRIM(?))
              AND LOWER(TRIM(department)) = LOWER(TRIM(?))
              AND LOWER(TRIM(day)) = LOWER(TRIM(?))
        """, (year, department, day_name)).fetchall()

        # Attendance already marked for this student on this date.
        status_map = {}
        try:
            marked = db.execute("""
                SELECT time, subject, status
                FROM attendance
                WHERE student_id = ? AND date = ?
            """, (student["student_id"], date_value)).fetchall()

            for a in marked:
                key = (str(a["time"]).strip().lower(),
                       str(a["subject"]).strip().lower())
                status_map[key] = str(a["status"]).strip().capitalize()
        except OperationalError:
            pass  # attendance table not created yet

        periods = []
        for row in rows:
            key = (str(row["time"]).strip().lower(),
                   str(row["subject"]).strip().lower())
            periods.append({
                "time": row["time"],
                "subject": row["subject"],
                "faculty": row["faculty"],
                "status": status_map.get(key)
            })

        periods.sort(key=lambda r: (period_start_minutes(r["time"]), r["subject"]))

        return jsonify(periods)

    except Exception as e:
        print("TIMETABLE API ERROR:", e)
        return jsonify([]), 500

    finally:
        db.close()


# ------------------------------------------------------------
# /api/attendance_summary
# ------------------------------------------------------------

@app.route("/api/attendance_summary")
def api_attendance_summary():

    if session.get("role") != "student":
        return jsonify({"present": 0, "absent": 0, "percentage": None}), 403

    db = get_db_connection()

    try:
        student = db.execute("""
            SELECT student_id FROM students WHERE username = ? LIMIT 1
        """, (session.get("username"),)).fetchone()

        present = absent = 0

        if student:
            try:
                rows = db.execute("""
                    SELECT LOWER(TRIM(status)) AS status, COUNT(*) AS total
                    FROM attendance
                    WHERE student_id = ?
                    GROUP BY LOWER(TRIM(status))
                """, (student["student_id"],)).fetchall()

                for r in rows:
                    if r["status"] == "present":
                        present = r["total"]
                    elif r["status"] == "absent":
                        absent = r["total"]
            except OperationalError:
                pass  # no attendance table yet

        total = present + absent
        percentage = round(present * 100 / total, 1) if total else None

        return jsonify({
            "present": present,
            "absent": absent,
            "percentage": percentage
        })

    except Exception as e:
        print("ATTENDANCE SUMMARY ERROR:", e)
        return jsonify({"present": 0, "absent": 0, "percentage": None}), 500

    finally:
        db.close()


# ------------------------------------------------------------
# /api/attendance_report
# ------------------------------------------------------------

@app.route("/api/attendance_report")
def api_attendance_report():

    empty = {"summary": {"present": 0, "absent": 0, "total": 0, "percentage": None},
             "subjects": [], "records": []}

    if session.get("role") != "student":
        return jsonify(empty), 403

    db = get_db_connection()

    try:
        student = db.execute(
            "SELECT student_id FROM students WHERE username = ? LIMIT 1",
            (session.get("username"),)
        ).fetchone()

        if not student:
            return jsonify(empty)

        try:
            rows = db.execute("""
                SELECT date, time, subject, status
                FROM attendance
                WHERE student_id = ?
                ORDER BY date DESC, marked_at DESC
            """, (student["student_id"],)).fetchall()
        except OperationalError:
            return jsonify(empty)   # attendance table not created yet

        records, subjects = [], {}
        present = absent = 0

        for r in rows:
            status = str(r["status"]).strip().capitalize()
            subject = str(r["subject"]).strip()
            records.append({"date": r["date"], "time": r["time"],
                            "subject": subject, "status": status})

            s = subjects.setdefault(subject, {"subject": subject, "present": 0,
                                              "absent": 0, "total": 0})
            if status == "Present":
                s["present"] += 1
                present += 1
            elif status == "Absent":
                s["absent"] += 1
                absent += 1
            s["total"] = s["present"] + s["absent"]

        total = present + absent

        return jsonify({
            "summary": {
                "present": present, "absent": absent, "total": total,
                "percentage": round(present * 100 / total, 1) if total else None
            },
            "subjects": sorted(subjects.values(), key=lambda s: s["subject"].lower()),
            "records": records
        })

    except Exception as e:
        print("ATTENDANCE REPORT ERROR:", e)
        return jsonify(empty), 500

    finally:
        db.close()


@app.route("/train_faces", methods=["POST"])
def train_faces():

    if session.get("role") not in ["admin", "representative"]:
        return jsonify({"success": False, "message": "Access Denied"}), 403

    try:
        success, message = train_face_model()
        return jsonify({"success": success, "message": message})

    except Exception as e:
        print("FACE TRAINING ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# ------------------------------------------------------------
# /logout
# ------------------------------------------------------------

@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# =========================================================
# RECOGNIZE FACE
# =========================================================

def recognize_face(frame):
    """
    Returns (student_dict, confidence, error).
    Failure example: (None, 72.3, "Unknown face.")
    """

    if not os.path.exists(TRAINER_PATH):
        return None, None, "Face model not trained."

    if not os.path.exists(FACE_LABELS_PATH):
        return None, None, "Face label mapping not found."

    try:
        with open(FACE_LABELS_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_PATH)

        if frame is None:
            return None, None, "Invalid camera frame."

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=FACE_MIN_SIZE
        )

        if len(faces) == 0:
            return None, None, "No face detected."

        # Use the largest face
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        face = gray[y:y + h, x:x + w]

        if face.size == 0:
            return None, None, "Invalid face region."

        face = preprocess_face(face)
        label, confidence = recognizer.predict(face)
        confidence = float(confidence)

        print(f"FACE PREDICTION -> label={label}, confidence={confidence:.2f}")

        student_id = label_map.get(str(label))

        if not student_id:
            return None, confidence, "Unknown face."

        # LBPH: lower confidence = better match
        if confidence > CONFIDENCE_THRESHOLD:
            return None, confidence, "Unknown face."

        db = get_db_connection()

        try:
            student = db.execute(
                """
                SELECT student_id, name, year, department
                FROM students
                WHERE student_id = ?
                LIMIT 1
                """,
                (str(student_id),)
            ).fetchone()
        finally:
            db.close()

        if not student:
            return None, confidence, "Student not found."

        student_data = {
            "student_id": student["student_id"],
            "name": student["name"],
            "year": student["year"],
            "department": student["department"]
        }

        print("FACE RECOGNIZED:", student_data)

        return student_data, confidence, None

    except Exception as e:
        print("FACE RECOGNITION ERROR:", e)
        return None, None, str(e)


# =========================================================
# FACE RECOGNITION PAGE
# =========================================================

@app.route("/face_recognition")
def face_recognition():
    if not require_attendance_role():
        return "Access Denied", 403
    return render_template("face_recognition.html")


# =========================================================
# AUTOMATIC ATTENDANCE HELPERS
# =========================================================

def _time_to_minutes(hour, minute, ampm=None):
    """
    Convert time to minutes since midnight.
    Without AM/PM: 1-7 = afternoon, 8-12 = morning, 13+ = 24-hour.
    """
    hour = int(hour)
    minute = int(minute)
    ampm = str(ampm or "").strip().lower()

    if ampm:
        if ampm == "pm":
            if hour != 12:
                hour += 12
        elif ampm == "am":
            if hour == 12:
                hour = 0
    else:
        if hour >= 13:
            pass
        elif hour == 0:
            hour = 0
        elif 1 <= hour <= 7:
            hour += 12
        else:
            pass

    return hour * 60 + minute


def _parse_timetable_range(value):
    """
    Parse timetable period such as:
    09:00 AM - 10:00 AM, 9:00-9:50, 1:30-2:20, 13:30-14:20, 01:30-02:20 PM
    """
    text = str(value or "").strip()

    if not text:
        return None

    match = re.search(
        r"""
        (\d{1,2})[:.](\d{2})
        \s*(AM|PM)?
        \s*[-–—]\s*
        (\d{1,2})[:.](\d{2})
        \s*(AM|PM)?
        """,
        text,
        re.IGNORECASE | re.VERBOSE
    )

    if not match:
        return None

    sh = int(match.group(1))
    sm = int(match.group(2))
    sap = match.group(3)

    eh = int(match.group(4))
    em = int(match.group(5))
    eap = match.group(6)

    # Only the end has AM/PM, e.g. "01:30 - 02:20 PM"
    if not sap and eap:
        if eap.upper() == "PM":
            sap = "PM" if sh <= 7 else "AM"
        else:
            sap = "AM"

    start_minutes = _time_to_minutes(sh, sm, sap)
    end_minutes = _time_to_minutes(eh, em, eap)

    # College time without AM/PM, e.g. "1:30-2:20"
    if end_minutes <= start_minutes and not eap and 1 <= eh <= 7:
        possible_end = (eh + 12) * 60 + em
        if possible_end > start_minutes:
            end_minutes = possible_end

    if end_minutes <= start_minutes:
        return None

    return start_minutes, end_minutes


def _find_current_timetable_period(db, student_year, department, day_name, current_minutes):
    """Find the timetable period currently active for this year + department."""

    rows = db.execute(
        """
        SELECT *
        FROM timetable
        WHERE LOWER(TRIM(department)) = LOWER(TRIM(?))
          AND LOWER(TRIM(day)) = LOWER(TRIM(?))
        ORDER BY id ASC
        """,
        (str(department), str(day_name))
    ).fetchall()

    wanted_year = normalize_year(student_year)

    for row in rows:

        try:
            row_year = row["year"]
        except Exception:
            row_year = ""

        if normalize_year(row_year) != wanted_year:
            continue

        try:
            time_value = row["time"]
        except Exception:
            time_value = ""

        parsed = _parse_timetable_range(time_value)

        if not parsed:
            continue

        start_minutes, end_minutes = parsed

        if start_minutes <= current_minutes < end_minutes:
            print("ACTIVE TIMETABLE:", dict(row))
            return row

    return None


# =========================================================
# FACE RECOGNITION + AUTOMATIC ATTENDANCE
# =========================================================

@app.route("/api/recognize_face", methods=["POST"])
def api_recognize_face():

    if not kiosk_or_staff():
        return jsonify({
            "success": False,
            "recognized": False,
            "message": "Access Denied"
        }), 403

    db = None

    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get("image", "")

        if not image_data:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "Image is required."
            }), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except Exception:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "Invalid image data."
            }), 400

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "Invalid image."
            }), 400

        # recognize_face() returns (student, confidence, error) - not a dict.
        student, confidence, error = recognize_face(frame)

        print("RECOGNITION RESULT:", student, confidence, error)

        if error:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": error,
                "confidence": confidence
            }), 200

        if not student:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "Face not recognized.",
                "confidence": confidence
            }), 200

        student_id = str(student["student_id"])
        student_name = str(student.get("name", ""))
        student_year = str(student.get("year", ""))
        student_department = str(student.get("department", ""))

        now = now_local()
        today = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        day_name = now.strftime("%A")
        current_minutes = now.hour * 60 + now.minute
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        attendance_marked = False
        attendance_status = None
        attendance_message = ""
        attendance_subject = None
        attendance_time = None

        db = get_db_connection()

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                subject TEXT NOT NULL,
                status TEXT NOT NULL,
                marked_at TEXT
            )
            """
        )
        db.commit()

        current_period = _find_current_timetable_period(
            db, student_year, student_department, day_name, current_minutes
        )

        if current_period is None:
            attendance_message = (
                "Face recognized, but no active timetable period found."
            )

        else:
            attendance_subject = str(current_period["subject"]).strip()
            attendance_time = str(current_period["time"]).strip()

            existing = db.execute(
                """
                SELECT id, status
                FROM attendance
                WHERE student_id = ?
                  AND date = ?
                  AND time = ?
                  AND subject = ?
                LIMIT 1
                """,
                (student_id, today, attendance_time, attendance_subject)
            ).fetchone()

            if existing:
                existing_status = str(existing["status"] or "").strip().lower()

                if existing_status == "present":
                    attendance_marked = True
                    attendance_status = "Present"
                    attendance_message = (
                        f"Already marked Present for {attendance_subject}."
                    )
                else:
                    db.execute(
                        """
                        UPDATE attendance
                        SET status = ?, marked_at = ?
                        WHERE id = ?
                        """,
                        ("Present", now_str, existing["id"])
                    )
                    db.commit()

                    attendance_marked = True
                    attendance_status = "Present"
                    attendance_message = (
                        f"Attendance updated to Present for {attendance_subject}."
                    )

            else:
                db.execute(
                    """
                    INSERT INTO attendance
                    (student_id, date, time, subject, status, marked_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (student_id, today, attendance_time,
                     attendance_subject, "Present", now_str)
                )
                db.commit()

                attendance_marked = True
                attendance_status = "Present"
                attendance_message = (
                    f"Attendance marked Present for {attendance_subject}."
                )

        return jsonify({
            "success": True,
            "recognized": True,
            "student": student,
            "student_id": student_id,
            "name": student_name,
            "year": student_year,
            "department": student_department,
            "confidence": (
                round(float(confidence), 2) if confidence is not None else None
            ),
            "attendance_marked": attendance_marked,
            "attendance_status": attendance_status,
            "attendance_message": attendance_message,
            "attendance_subject": attendance_subject,
            "attendance_time": attendance_time,
            "current_time": current_time,
            "date": today
        })

    except Exception as e:
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass

        print("FACE RECOGNITION / ATTENDANCE ERROR:", e)

        return jsonify({
            "success": False,
            "recognized": False,
            "message": str(e)
        }), 500

    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


# =========================================================
# KIOSK (auto-start camera, no login)
# =========================================================

@app.route("/kiosk")
def kiosk():
    return render_template("kiosk.html")


# =========================================================
# REGISTERED FACES (admin page + APIs)
# =========================================================

FACE_SAMPLES_REQUIRED = 20


def _sample_files(student_id):
    """Sorted list of (number, filename) for a student's saved face images."""
    folder = os.path.join(FACE_DIR, str(student_id))
    if not os.path.isdir(folder):
        return []
    items = []
    for f in os.listdir(folder):
        name, ext = os.path.splitext(f)
        if ext.lower() in (".jpg", ".jpeg", ".png") and name.isdigit():
            items.append((int(name), f))
    return sorted(items)


def _face_status(count):
    if count >= FACE_SAMPLES_REQUIRED:
        return "registered"
    if count > 0:
        return "incomplete"
    return "not_registered"


@app.route("/registered_faces")
def registered_faces():
    if session.get("role") != "admin":
        return "Access Denied", 403
    return render_template("registered_faces.html", required=FACE_SAMPLES_REQUIRED)


@app.route("/api/registered_faces")
def api_registered_faces():
    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403

    db = get_db_connection()
    try:
        rows = db.execute(
            "SELECT student_id, name, year, department FROM students ORDER BY name COLLATE NOCASE"
        ).fetchall()
    except Exception as e:
        print("REGISTERED FACES ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        db.close()

    students = []
    registered = incomplete = total_samples = 0
    for r in rows:
        sid = str(r["student_id"])
        files = _sample_files(sid)
        count = len(files)
        status = _face_status(count)
        registered += status == "registered"
        incomplete += status == "incomplete"
        total_samples += count
        students.append({
            "student_id": sid,
            "name": r["name"],
            "year": normalize_year(r["year"]),
            "department": r["department"],
            "samples": count,
            "status": status,
            "photo": url_for("face_sample", student_id=sid, number=files[0][0]) if files else None,
        })

    return jsonify({
        "success": True,
        "students": students,
        "summary": {
            "total_students": len(students),
            "registered": registered,
            "incomplete": incomplete,
            "total_samples": total_samples,
            "max_samples": len(students) * FACE_SAMPLES_REQUIRED,
        },
    })


@app.route("/api/registered_faces/<student_id>")
def api_registered_face_detail(student_id):
    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403

    db = get_db_connection()
    try:
        s = db.execute(
            "SELECT student_id, name, year, department FROM students WHERE student_id = ?",
            (student_id,)
        ).fetchone()
    finally:
        db.close()

    if not s:
        return jsonify({"success": False, "message": "Student not found."}), 404

    files = _sample_files(student_id)
    return jsonify({
        "success": True,
        "student_id": str(s["student_id"]),
        "name": s["name"],
        "year": normalize_year(s["year"]),
        "department": s["department"],
        "samples": len(files),
        "required": FACE_SAMPLES_REQUIRED,
        "status": _face_status(len(files)),
        "images": [
            {"number": n, "url": url_for("face_sample", student_id=student_id, number=n)}
            for n, _ in files
        ],
    })


@app.route("/face_sample/<student_id>/<int:number>")
def face_sample(student_id, number):
    """Serve one saved face image (admin only)."""
    if session.get("role") != "admin":
        return "Access Denied", 403

    from flask import send_from_directory, abort
    if not re.fullmatch(r"[A-Za-z0-9_-]+", student_id):
        abort(404)

    folder = os.path.join(FACE_DIR, student_id)
    for n, fname in _sample_files(student_id):
        if n == number:
            resp = send_from_directory(folder, fname)
            resp.headers["Cache-Control"] = "no-store"
            return resp
    abort(404)


# =========================================================
# ADMIN ATTENDANCE REPORT (page + API)
# =========================================================

@app.route("/attendance_report")
def attendance_report_page():
    if session.get("role") != "admin":
        return "Access Denied", 403
    return render_template("attendance_report.html")


@app.route("/api/admin_attendance_report")
def api_admin_attendance_report():
    """Per-student attendance totals for a department + year.

    One marked period = one hour (same rule as the student summary API).
    """
    if not require_attendance_role():
        return jsonify({"success": False, "message": "Access Denied"}), 403

    year = normalize_year(request.args.get("year", "")).upper()
    department = request.args.get("department", "").strip()
    month = request.args.get("month", "").strip()          # YYYY-MM (optional)
    month_like = (month + "%") if month else "%"

    year_aliases = {
        "I": ["i", "1st", "1st year", "1"],
        "II": ["ii", "2nd", "2nd year", "2"],
        "III": ["iii", "3rd", "3rd year", "3"],
        "IV": ["iv", "4th", "4th year", "4"],
    }

    if not year or not department:
        return jsonify({
            "success": False,
            "message": "Department and year are required."
        }), 400

    possible_years = year_aliases.get(year, [year.lower()])

    db = None
    try:
        db = get_db_connection()

        # Attendance table may not exist yet on a fresh database.
        db.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                subject TEXT NOT NULL,
                status TEXT NOT NULL,
                marked_at TEXT
            )
        """)

        placeholders = ",".join(["?"] * len(possible_years))

        rows = db.execute(f"""
            SELECT
                s.student_id,
                s.name,
                COALESCE(SUM(CASE WHEN LOWER(TRIM(a.status)) IN ('present', 'absent')
                                  THEN 1 ELSE 0 END), 0) AS total_hours,
                COALESCE(SUM(CASE WHEN LOWER(TRIM(a.status)) = 'present'
                                  THEN 1 ELSE 0 END), 0) AS present_hours
            FROM students s
            LEFT JOIN attendance a ON a.student_id = s.student_id AND a.date LIKE ?
            WHERE LOWER(TRIM(s.year)) IN ({placeholders})
              AND LOWER(TRIM(s.department)) = LOWER(TRIM(?))
            GROUP BY s.student_id, s.name
            ORDER BY s.student_id
        """, [month_like] + possible_years + [department]).fetchall()

        return jsonify({
            "success": True,
            "students": [dict(r) for r in rows]
        })

    except Exception as e:
        print("ADMIN ATTENDANCE REPORT ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500

    finally:
        if db is not None:
            db.close()


# =========================================================
# ADMIN CHANGE PASSWORD
# =========================================================

@app.route("/api/change_password", methods=["POST"])
def api_change_password():
    if session.get("role") != "admin" or not session.get("user_id"):
        return jsonify({"success": False, "message": "Access Denied"}), 403

    data = request.get_json(silent=True) or {}
    current = str(data.get("current_password", ""))
    new = str(data.get("new_password", ""))
    confirm = str(data.get("confirm_password", ""))

    if not current or not new or not confirm:
        return jsonify({"success": False, "message": "Please fill all fields."}), 400

    if new != confirm:
        return jsonify({"success": False, "message": "New passwords do not match."}), 400

    if len(new) < 6:
        return jsonify({
            "success": False,
            "message": "New password must contain at least 6 characters."
        }), 400

    if hmac.compare_digest(new, current):
        return jsonify({
            "success": False,
            "message": "New password must be different from the current password."
        }), 400

    db = None
    try:
        db = get_db_connection()

        user = db.execute(
            "SELECT id, password FROM users WHERE id = ? AND role = 'admin'",
            (session["user_id"],)
        ).fetchone()

        if not user:
            return jsonify({"success": False, "message": "Admin account not found."}), 404

        if not hmac.compare_digest(str(user["password"]), current):
            return jsonify({
                "success": False,
                "message": "Current password is incorrect."
            }), 400

        db.execute("UPDATE users SET password = ? WHERE id = ?", (new, user["id"]))
        db.commit()

        return jsonify({"success": True, "message": "Password changed successfully."})

    except Exception as e:
        print("CHANGE PASSWORD ERROR:", e)
        return jsonify({"success": False, "message": "Unable to change password."}), 500

    finally:
        if db is not None:
            db.close()


# =========================================================
# FACULTY / REPRESENTATIVE PAGES
# =========================================================

def _staff_page(page):
    if not require_attendance_role():
        return redirect(url_for("login"))
    return render_template(
        "staff.html",
        page=page,
        username=session.get("username", ""),
        role=session.get("role", "")
    )


@app.route("/staff")
def staff_home():
    return _staff_page("home")


@app.route("/staff/manual")
def staff_manual():
    return _staff_page("manual")


@app.route("/staff/hourly")
def staff_hourly():
    return _staff_page("hourly")


@app.route("/staff/report")
def staff_report():
    return _staff_page("report")


def _year_variants(year):
    aliases = {
        "I": ["i", "1st", "1st year", "1"],
        "II": ["ii", "2nd", "2nd year", "2"],
        "III": ["iii", "3rd", "3rd year", "3"],
        "IV": ["iv", "4th", "4th year", "4"],
    }
    y = normalize_year(year).upper()
    return aliases.get(y, [y.lower()])


@app.route("/api/departments")
def api_departments():
    if not require_attendance_role():
        return jsonify({"success": False, "message": "Access Denied"}), 403

    db = get_db_connection()
    try:
        rows = db.execute("""
            SELECT DISTINCT TRIM(department) AS department
            FROM students
            WHERE department IS NOT NULL AND TRIM(department) <> ''
            ORDER BY department COLLATE NOCASE
        """).fetchall()
        return jsonify({"success": True,
                        "departments": [r["department"] for r in rows]})
    except Exception as e:
        print("DEPARTMENTS ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/period_attendance")
def api_period_attendance():
    """Every period of a day with the full student list and each status."""
    if not require_attendance_role():
        return jsonify({"success": False, "message": "Access Denied"}), 403

    date_value = request.args.get("date", "").strip()
    year = normalize_year(request.args.get("year", ""))
    department = request.args.get("department", "").strip()

    if not date_value or not year or not department:
        return jsonify({
            "success": False,
            "message": "Date, year and department are required."
        }), 400

    try:
        day_name = datetime.strptime(date_value, "%Y-%m-%d").strftime("%A")
    except ValueError:
        return jsonify({"success": False, "message": "Invalid date."}), 400

    db = None
    try:
        db = get_db_connection()

        variants = _year_variants(year)
        placeholders = ",".join(["?"] * len(variants))

        students = db.execute(f"""
            SELECT student_id, name
            FROM students
            WHERE LOWER(TRIM(year)) IN ({placeholders})
              AND LOWER(TRIM(department)) = LOWER(TRIM(?))
            ORDER BY student_id
        """, variants + [department]).fetchall()

        slots = db.execute("""
            SELECT time, subject, faculty
            FROM timetable
            WHERE UPPER(TRIM(year)) = UPPER(TRIM(?))
              AND LOWER(TRIM(department)) = LOWER(TRIM(?))
              AND LOWER(TRIM(day)) = LOWER(TRIM(?))
        """, (year, department, day_name)).fetchall()

        status_map = {}
        try:
            marked = db.execute("""
                SELECT student_id, time, subject, status
                FROM attendance
                WHERE date = ?
            """, (date_value,)).fetchall()
            for a in marked:
                key = (str(a["student_id"]),
                       str(a["time"]).strip().lower(),
                       str(a["subject"]).strip().lower())
                status_map[key] = str(a["status"]).strip().capitalize()
        except OperationalError:
            pass  # attendance table not created yet

        now = now_local()
        today = now.strftime("%Y-%m-%d")
        now_minutes = now.hour * 60 + now.minute

        periods = []
        seen = set()

        for slot in sorted(slots, key=lambda r: period_start_minutes(r["time"])):
            time_text = str(slot["time"] or "").strip()
            subject = str(slot["subject"] or "").strip()

            if (time_text.lower(), subject.lower()) in seen:
                continue
            seen.add((time_text.lower(), subject.lower()))

            parsed = _parse_timetable_range(time_text)

            if date_value < today:
                state = "ended"
            elif date_value > today:
                state = "upcoming"
            elif not parsed:
                state = "unknown"
            elif now_minutes >= parsed[1]:
                state = "ended"
            elif now_minutes >= parsed[0]:
                state = "ongoing"
            else:
                state = "upcoming"

            rows = []
            present = absent = 0
            for st in students:
                sid = str(st["student_id"])
                status = status_map.get(
                    (sid, time_text.lower(), subject.lower()), "Not marked"
                )
                if status == "Present":
                    present += 1
                elif status == "Absent":
                    absent += 1
                rows.append({"roll_no": sid, "name": st["name"], "status": status})

            periods.append({
                "time": time_text,
                "subject": subject,
                "faculty": slot["faculty"],
                "state": state,
                "present": present,
                "absent": absent,
                "not_marked": len(rows) - present - absent,
                "students": rows
            })

        return jsonify({
            "success": True,
            "day": day_name,
            "total_students": len(students),
            "periods": periods
        })

    except Exception as e:
        print("PERIOD ATTENDANCE ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500

    finally:
        if db is not None:
            db.close()



# =========================================================
# FACULTY PROFILE (name, department, year from Faculty.csv)
# =========================================================

FACULTY_CSV = os.path.join(BASE_DIR, "Faculty.csv")


def _csv_faculty():
    import csv
    data = {}
    try:
        with open(FACULTY_CSV, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                name = (r.get("name") or "").strip()
                if name:
                    data[name.lower()] = {
                        "name": name,
                        "department": (r.get("department") or "").strip(),
                        "year": normalize_year(r.get("year")),
                    }
    except FileNotFoundError:
        print("Faculty.csv not found")
    return data


def load_faculty():
    """Faculty from the database (falls back to Faculty.csv)."""
    data = {}
    try:
        db = get_db_connection()
        try:
            for r in db.execute("SELECT name, department, year FROM faculty").fetchall():
                data[str(r["name"]).strip().lower()] = {
                    "name": str(r["name"]).strip(),
                    "department": str(r["department"] or "").strip(),
                    "year": normalize_year(r["year"]),
                }
        finally:
            db.close()
    except Exception as e:
        print("FACULTY LOAD ERROR:", e)
    return data or _csv_faculty()


@app.route("/api/faculty_profile")
def api_faculty_profile():
    if not require_attendance_role():
        return jsonify({"success": False, "message": "Access Denied"}), 403
    username = session.get("username", "")
    info = load_faculty().get(username.lower(), {})
    return jsonify({
        "success": True,
        "username": username,
        "role": session.get("role", ""),
        "name": info.get("name", username),
        "department": info.get("department", ""),
        "year": info.get("year", ""),
    })


# =========================================================
# ADD FACULTY (admin: upload CSV / Excel)
# =========================================================

@app.route("/add_faculty")
def add_faculty_page():
    if session.get("role") != "admin":
        return "Access Denied"
    return render_template("add_faculty.html")


@app.route("/api/faculty_list")
def api_faculty_list():
    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403
    db = get_db_connection()
    try:
        rows = db.execute(
            "SELECT name, department, year FROM faculty ORDER BY year, department, name"
        ).fetchall()
        return jsonify({"success": True, "faculty": [
            {"name": r["name"], "department": r["department"], "year": r["year"]}
            for r in rows]})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/faculty_set_password", methods=["POST"])
def api_faculty_set_password():
    """Set one password for every faculty login."""
    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403
    pw = str((request.get_json(silent=True) or {}).get("password", "")).strip()
    if len(pw) < 6:
        return jsonify({"success": False,
                        "message": "Password must contain at least 6 characters."}), 400
    db = get_db_connection()
    try:
        n = db.execute("SELECT COUNT(*) AS c FROM users WHERE role = 'faculty'").fetchone()["c"]
        db.execute("UPDATE users SET password = ? WHERE role = 'faculty'", (pw,))
        db.commit()
        return jsonify({"success": True, "count": n,
                        "message": f"Password updated for {n} faculty login(s)."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        db.close()


@app.route("/upload_faculty_csv", methods=["POST"])
def upload_faculty_csv():
    """Rows come from the browser (CSV or Excel parsed there).
    Columns: name, department, year [, password]."""
    if session.get("role") != "admin":
        return jsonify({"success": False, "message": "Access Denied"}), 403

    data = request.get_json(silent=True) or {}
    rows = data.get("rows") or []
    if len(rows) < 2:
        return jsonify({"success": False,
                        "message": "The file must contain a header row and faculty data."}), 400

    headers = [str(h).strip().lower().replace(" ", "_") for h in rows[0]]
    for need in ("name", "department", "year"):
        if need not in headers:
            return jsonify({"success": False,
                            "message": "Missing column. Use: name, department, year (password optional)."}), 400
    idx = {h: headers.index(h) for h in ("name", "department", "year", "password") if h in headers}

    default_pw = str(data.get("default_password") or "").strip() \
        or os.environ.get("FACULTY_DEFAULT_PASSWORD", "faculty123")
    if len(default_pw) < 6:
        return jsonify({"success": False,
                        "message": "Default password must contain at least 6 characters."}), 400
    reset_existing = bool(data.get("reset_existing"))
    added = updated = skipped = 0
    errors = []
    db = get_db_connection()
    try:
        for n, row in enumerate(rows[1:], start=2):
            if not row or not any(str(c).strip() for c in row):
                continue

            def cell(k):
                i = idx.get(k)
                return str(row[i]).strip() if i is not None and i < len(row) and row[i] is not None else ""

            name, dept = cell("name"), cell("department")
            year = normalize_year(cell("year")).upper()
            password = cell("password") or default_pw

            if not (name and dept and year):
                skipped += 1
                errors.append(f"Row {n}: name, department and year are required")
                continue
            if year not in ("I", "II", "III", "IV"):
                skipped += 1
                errors.append(f"Row {n}: year must be I, II, III or IV")
                continue

            exists = db.execute(
                "SELECT id FROM faculty WHERE LOWER(name) = LOWER(?)", (name,)
            ).fetchone()
            if exists:
                db.execute("UPDATE faculty SET department = ?, year = ? WHERE id = ?",
                           (dept, year, exists["id"]))
                updated += 1
            else:
                db.execute("INSERT INTO faculty (name, department, year) VALUES (?, ?, ?)",
                           (name, dept, year))
                added += 1

            user = db.execute("SELECT id, role FROM users WHERE LOWER(username) = LOWER(?)",
                              (name,)).fetchone()
            if not user:
                db.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'faculty')",
                           (name, password))
            elif reset_existing and user["role"] == "faculty":
                db.execute("UPDATE users SET password = ? WHERE id = ?", (password, user["id"]))

        db.commit()
        return jsonify({"success": True, "added": added, "updated": updated,
                        "skipped": skipped, "errors": errors[:50],
                        "message": f"{added} faculty added, {updated} updated."})
    except Exception as e:
        print("UPLOAD FACULTY ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        db.close()


# =========================================================
# TEST
# =========================================================

@app.route("/test")
def test():
    return "FLASK IS WORKING"


# =========================================================
# DATABASE AUTO-SETUP (creates tables + default admin)
# =========================================================

def init_db():
    """Create tables and a default admin if the database is empty."""
    db = get_db_connection()
    try:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                year TEXT,
                department TEXT,
                email TEXT,
                phone TEXT,
                gender TEXT
            );
            CREATE TABLE IF NOT EXISTS faculty (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                department TEXT NOT NULL,
                year TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS timetable (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year TEXT,
                department TEXT,
                day TEXT,
                time TEXT,
                subject TEXT,
                faculty TEXT
            );
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                subject TEXT NOT NULL,
                status TEXT NOT NULL,
                marked_at TEXT
            );
        """)

        has_admin = db.execute(
            "SELECT 1 FROM users WHERE role = 'admin' LIMIT 1"
        ).fetchone()

        if not has_admin:
            admin_user = os.environ.get("ADMIN_USERNAME", "admin")
            admin_pass = os.environ.get("ADMIN_PASSWORD", "admin123")
            db.execute(
                "INSERT OR IGNORE INTO users (username, password, role) "
                "VALUES (?, ?, 'admin')",
                (admin_user, admin_pass)
            )
            print("DEFAULT ADMIN CREATED:", admin_user)

        fac_pass = os.environ.get("FACULTY_DEFAULT_PASSWORD", "faculty123")
        for f in _csv_faculty().values():
            if not db.execute("SELECT 1 FROM faculty WHERE LOWER(name) = LOWER(?)",
                              (f["name"],)).fetchone():
                db.execute("INSERT INTO faculty (name, department, year) VALUES (?, ?, ?)",
                           (f["name"], f["department"], f["year"]))
        for r in db.execute("SELECT name FROM faculty").fetchall():
            db.execute(
                "INSERT OR IGNORE INTO users (username, password, role) "
                "VALUES (?, ?, 'faculty')",
                (r["name"], fac_pass)
            )

        db.commit()
    finally:
        db.close()


# Module level on purpose: gunicorn (Render) never runs the __main__ block.
try:
    init_db()
    ensure_student_contact_columns()
except Exception as e:
    print("DATABASE INIT ERROR:", e)


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("========================================")
    print("SMART ATTENDANCE SERVER STARTING")
    print("DATABASE:", TURSO_URL if USE_TURSO else DATABASE)
    print("TRAINER:", TRAINER_PATH)
    print("FACE LABELS:", FACE_LABELS_PATH)
    print("========================================")

    # Auto-open the kiosk page (camera + scanning start by themselves)
    # 1.5 s after the server starts.
    kiosk_url = f"http://localhost:5000/kiosk?key={KIOSK_KEY}"
    threading.Timer(1.5, lambda: webbrowser.open(kiosk_url)).start()
    print("KIOSK PAGE:", kiosk_url)

    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False
    )