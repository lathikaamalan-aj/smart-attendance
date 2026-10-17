
import os
import sqlite3
import psycopg
from psycopg import sql

# Local SQLite database
SQLITE_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "attendance.db"
)

# Render PostgreSQL connection URL
DATABASE_URL = os.environ.get("DATABASE_URL")

TABLES = [
    "users",
    "students",
    "timetable",
    "attendance",
]


def main():
    if not os.path.exists(SQLITE_DB):
        raise FileNotFoundError(
            f"SQLite database not found: {SQLITE_DB}"
        )

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL environment variable is missing."
        )

    source = sqlite3.connect(SQLITE_DB)
    source.row_factory = sqlite3.Row

    try:
        with psycopg.connect(DATABASE_URL) as target:
            with target.cursor() as cur:

                # Verify the expected tables exist locally.
                existing_tables = {
                    row["name"]
                    for row in source.execute(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'table'
                        """
                    )
                }

                missing = set(TABLES) - existing_tables
                if missing:
                    raise RuntimeError(
                        f"Missing SQLite tables: {sorted(missing)}"
                    )

                # Copy only into empty PostgreSQL tables.
                for table in TABLES:
                    cur.execute(
                        sql.SQL(
                            "SELECT COUNT(*) FROM {}"
                        ).format(sql.Identifier(table))
                    )
                    count = cur.fetchone()[0]

                    if count:
                        raise RuntimeError(
                            f"PostgreSQL table '{table}' "
                            f"already has {count} rows. "
                            "Stopping to avoid duplicate data."
                        )

                for table in TABLES:
                    columns = [
                        row["name"]
                        for row in source.execute(
                            f'PRAGMA table_info("{table}")'
                        )
                    ]

                    rows = source.execute(
                        f'SELECT * FROM "{table}"'
                    ).fetchall()

                    if not columns:
                        raise RuntimeError(
                            f"No columns found for {table}"
                        )

                    # Insert rows using the SQLite column names.
                    query = sql.SQL(
                        "INSERT INTO {} ({}) VALUES ({})"
                    ).format(
                        sql.Identifier(table),
                        sql.SQL(", ").join(
                            sql.Identifier(c) for c in columns
                        ),
                        sql.SQL(", ").join(
                            sql.Placeholder() for _ in columns
                        ),
                    )

                    for row in rows:
                        cur.execute(query, tuple(row))

                    print(
                        f"{table}: {len(rows)} rows copied"
                    )

            # Commit happens when the connection context exits.
        print("\nMigration completed successfully!")

    finally:
        source.close()


if __name__ == "__main__":
    main()
