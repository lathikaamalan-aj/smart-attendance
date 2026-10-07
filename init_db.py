import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "attendance.db")

conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

# ==========================================
# USERS TABLE
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL
)
""")

# ==========================================
# STUDENTS TABLE
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    year TEXT,
    department TEXT
)
""")

# ==========================================
# ADMIN ACCOUNT
# ==========================================

cursor.execute("""
INSERT OR IGNORE INTO users
(username, password, role)
VALUES (?, ?, ?)
""", (
    "admin",
    "admin@123",
    "admin"
))

conn.commit()
conn.close()

print("===================================")
print("Database initialized successfully")
print("Database:", DATABASE)
print("Admin username: admin")
print("Admin password: admin@123")
print("===================================")