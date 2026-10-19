"""
Database layer for Smart Attendance.

- If the DATABASE_URL environment variable is set  -> PostgreSQL (Neon, Render, ...)
- Otherwise                                         -> local SQLite file (as before)

The Postgres wrapper accepts the same SQLite-style code used in app.py
(`?` placeholders, PRAGMA table_info, COLLATE NOCASE, AUTOINCREMENT,
INSERT OR IGNORE, executescript) so the routes need no changes.
It also re-raises Postgres errors as sqlite3.IntegrityError /
sqlite3.OperationalError so the existing `except sqlite3...` blocks still work.
"""

import os
import re
import sqlite3

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

USE_PG = bool(DATABASE_URL)

if USE_PG:
    import psycopg2


_PRAGMA = re.compile(r"^\s*PRAGMA\s+table_info\((\w+)\)\s*;?\s*$", re.I)
_NOCASE_ORDER = re.compile(r"ORDER BY\s+([\w.]+)\s+COLLATE\s+NOCASE", re.I)
_INSERT_IGNORE = re.compile(r"^\s*INSERT\s+OR\s+IGNORE\s+INTO", re.I)
_WRITE = re.compile(r"^\s*(INSERT|UPDATE|DELETE)\b", re.I)


def translate(sql, params=()):
    """Convert one SQLite-style statement to PostgreSQL. Returns (sql, params)."""
    m = _PRAGMA.match(sql)
    if m:
        return (
            "SELECT column_name AS name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s",
            (m.group(1).lower(),),
        )

    sql = sql.replace("?", "%s")
    sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")

    if _INSERT_IGNORE.match(sql):
        sql = re.sub(r"INSERT\s+OR\s+IGNORE", "INSERT", sql, count=1, flags=re.I)
        sql = sql.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"

    if re.search(r"SELECT\s+DISTINCT", sql, re.I):
        # Postgres: ORDER BY must be in the select list when DISTINCT is used
        sql = re.sub(r"\s+COLLATE\s+NOCASE", "", sql, flags=re.I)
    else:
        sql = _NOCASE_ORDER.sub(r"ORDER BY LOWER(\1)", sql)

    return sql, params


class Row(dict):
    """Behaves like sqlite3.Row: row["name"], row[0], dict(row), row.keys()."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class Result:
    """Cursor wrapper that returns Row objects."""

    def __init__(self, cur):
        self._cur = cur
        self.rowcount = cur.rowcount
        self.description = cur.description

    def _wrap(self, values):
        cols = [d[0] for d in self._cur.description]
        return Row(zip(cols, values))

    def fetchone(self):
        if self._cur.description is None:
            return None
        values = self._cur.fetchone()
        return None if values is None else self._wrap(values)

    def fetchall(self):
        if self._cur.description is None:
            return []
        return [self._wrap(v) for v in self._cur.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


class PgConnection:
    """Mimics the small part of sqlite3.Connection that app.py uses."""

    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
        self._sp = self.conn.cursor()

    def execute(self, sql, params=()):
        sql, params = translate(sql, params)
        write = bool(_WRITE.match(sql))
        cur = self.conn.cursor()
        try:
            # A savepoint lets one failed INSERT (e.g. duplicate row in a CSV
            # upload) be skipped without aborting the whole transaction.
            if write:
                self._sp.execute("SAVEPOINT stmt")
            cur.execute(sql, params if params else None)
            if write:
                self._sp.execute("RELEASE SAVEPOINT stmt")
            return Result(cur)
        except psycopg2.Error as e:
            if write:
                self._sp.execute("ROLLBACK TO SAVEPOINT stmt")
                self._sp.execute("RELEASE SAVEPOINT stmt")
            else:
                self.conn.rollback()
            if isinstance(e, psycopg2.IntegrityError):
                raise sqlite3.IntegrityError(str(e)) from e
            raise sqlite3.OperationalError(str(e)) from e

    def executescript(self, script):
        for statement in script.split(";"):
            if statement.strip():
                self.execute(statement)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


def connect(sqlite_path):
    if USE_PG:
        return PgConnection()
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn