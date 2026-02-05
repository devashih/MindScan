import sqlite3
import hashlib

DB_NAME = "mindscan.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        emergency_email TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS journal_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        entry_text TEXT,
        emotion TEXT,
        confidence REAL,
        stress_level INTEGER,
        risk_level TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def register_user(username, password, emergency_email):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password, emergency_email) VALUES (?, ?, ?)",
            (username, hash_password(password), emergency_email)
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def get_emergency_email(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT emergency_email FROM users WHERE id=?",
        (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def save_entry(user_id, text, emotion, confidence, stress, risk):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO journal_entries
        (user_id, entry_text, emotion, confidence, stress_level, risk_level)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, text, emotion, confidence, stress, risk))
    conn.commit()
    conn.close()

def fetch_user_entries(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT emotion, confidence, stress_level, risk_level, created_at
        FROM journal_entries
        WHERE user_id=?
        ORDER BY created_at ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows
