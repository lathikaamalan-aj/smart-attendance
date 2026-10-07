-- ============================================================
-- Smart Attendance - All Years Timetable Setup
-- ============================================================

-- Old timetable table + existing 40 rows remove
DROP TABLE IF EXISTS timetable;

-- New timetable table
CREATE TABLE timetable (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year TEXT NOT NULL,
    department TEXT NOT NULL,
    day TEXT NOT NULL,
    time TEXT NOT NULL,
    subject TEXT NOT NULL,
    faculty TEXT
);

-- Faster timetable filtering
CREATE INDEX idx_timetable_year_department_day
ON timetable(year, department, day);