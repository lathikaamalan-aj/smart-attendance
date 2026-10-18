"""
One-time migration: copy data from local SQLite (attendance.db) to Neon Postgres.

Run from the project folder (where attendance.db is):
    pip install psycopg2-binary
    python migrate_to_neon.py

Do NOT push this file to GitHub.
"""

import sqlite3
import psycopg2

SQLITE_PATH = "attendance.db"
TABLES = ["users", "students", "timetable", "attendance"]

url = input("Neon connection string paste pannunga: ").strip().strip('"').strip("'")
if url.startswith("postgres://"):
    url = "postgresql://" + url[len("postgres://"):]
url = url.replace("&channel_binding=require", "")

src = sqlite3.connect(SQLITE_PATH)
src.row_factory = sqlite3.Row

dst = psycopg2.connect(url)
cur = dst.cursor()

for table in TABLES:
    try:
        rows = src.execute(f"SELECT * FROM {table}").fetchall()
    except sqlite3.OperationalError:
        print(f"{table}: SQLite la table illa, skip")
        continue

    if not rows:
        print(f"{table}: 0 rows, skip")
        continue

    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s",
        (table,),
    )
    pg_cols = {r[0] for r in cur.fetchall()}
    if not pg_cols:
        print(f"{table}: Neon la table illa (Render app oru murai start aagirukkanum)")
        continue

    # timetable / attendance have no unique key, so don't copy twice
    if table in ("timetable", "attendance"):
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        if cur.fetchone()[0] > 0:
            print(f"{table}: Neon la already data irukku, skip")
            continue

    # skip the auto id; Postgres generates new ones
    cols = [c for c in rows[0].keys() if c != "id" and c in pg_cols]
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join(['%s'] * len(cols))}) ON CONFLICT DO NOTHING"
    )

    copied = 0
    for r in rows:
        cur.execute(sql, [r[c] for c in cols])
        copied += cur.rowcount

    print(f"{table}: {copied} of {len(rows)} rows copied")

dst.commit()
dst.close()
src.close()
print("Done.")
