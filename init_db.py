import sqlite3

conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()


# ==============================
# USERS TABLE
# ==============================

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL
)
""")


# ==============================
# REMOVE OLD STUDENTS TABLE
# ==============================

cursor.execute("DROP TABLE IF EXISTS students")


# ==============================
# CREATE NEW STUDENTS TABLE
# ==============================

cursor.execute("""
CREATE TABLE students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    year TEXT,
    department TEXT
)
""")


# ==============================
# ADMIN ACCOUNT
# ==============================

cursor.execute("""
INSERT OR IGNORE INTO users (username, password, role)
VALUES (?, ?, ?)
""", ("admin", "admin@123", "admin"))


# ==============================
# SAVE
# ==============================

conn.commit()
conn.close()


print("Database created successfully!")
print("Admin account created:")
print("Username: admin")
print("Password: admin@123")