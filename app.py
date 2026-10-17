        ]):
            return jsonify({
                "success": False,
                "message": "Please fill all fields."
            }), 400

        db = get_db_connection()

        # -----------------------------
        # CHECK STUDENT ID
        # -----------------------------

        existing_student = db.execute("""
            SELECT id
            FROM students
            WHERE student_id = ?
        """, (student_id,)).fetchone()

        if existing_student:
            db.close()

            return jsonify({
                "success": False,
                "message": "Student ID already exists."
            }), 400

        # -----------------------------
        # CHECK USERNAME
        # -----------------------------

        existing_user = db.execute("""
            SELECT id
            FROM users
            WHERE username = ?
        """, (username,)).fetchone()

        if existing_user:
            db.close()

            return jsonify({
                "success": False,
                "message": "Username already exists."
            }), 400

        # -----------------------------
        # INSERT STUDENT
        # -----------------------------

        db.execute("""
            INSERT INTO students
            (
                student_id,
                name,
                username,
                password,
                year,
                department
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            student_id,
            name,
            username,
            password,
            year,
            department
        ))

        # -----------------------------
        # INSERT LOGIN ACCOUNT
        # -----------------------------

        db.execute("""
            INSERT INTO users
            (
                username,
                password,
                role
            )
            VALUES (?, ?, ?)
        """, (
            username,
            password,
            "student"
        ))

        db.commit()
        db.close()

        return jsonify({
            "success": True,
            "message": f"{name} added successfully."
        })

    except Exception as e:

        print("ADD STUDENT ERROR:", e)

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
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
            SELECT
                id,
                student_id,
                name,
                username,
                year,
                department
            FROM students
            ORDER BY id ASC
        """).fetchall()

        db.close()

        students = [dict(student) for student in students]

        return render_template(
            "view_students.html",
            students=students
        )

    except Exception as e:

        print("VIEW STUDENTS ERROR:", e)

        return "Database error: " + str(e)


# =========================================================
# API - GET ALL STUDENTS
# =========================================================

@app.route("/api/students")
def api_students():

    if session.get("role") != "admin":
        return jsonify({
            "success": False,
            "message": "Access Denied"
        }), 403

    try:

        db = get_db_connection()