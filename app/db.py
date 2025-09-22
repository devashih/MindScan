import os, sqlite3, datetime as dt, hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "mindscan.db")

# Final schema for entries (no mood)
ENTRIES_COLUMNS = ["id", "user_id", "text", "sentiment", "emotion", "created_at"]

def _table_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def _migrate_entries_table(conn):
    cols = _table_columns(conn, "entries")
    needs_rebuild = (
        ("mood" in cols) or
        ("user_id" not in cols) or
        sorted(cols) != sorted(ENTRIES_COLUMNS)
    )
    if not needs_rebuild:
        return
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries_new(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text TEXT,
            sentiment REAL,
            emotion TEXT,
            created_at TEXT
        )
    """)
    src_cols = _table_columns(conn, "entries")
    select_parts = []
    for c in ENTRIES_COLUMNS:
        if c in src_cols:
            select_parts.append(c)
        elif c == "user_id":
            select_parts.append("NULL AS user_id")
        else:
            select_parts.append(f"NULL AS {c}")
    select_sql = ", ".join(select_parts)
    conn.execute(
        f"INSERT INTO entries_new ({', '.join(ENTRIES_COLUMNS)}) "
        f"SELECT {select_sql} FROM entries"
    )
    conn.execute("DROP TABLE entries")
    conn.execute("ALTER TABLE entries_new RENAME TO entries")
    conn.commit()

def get_db():
    data_dir = os.path.join(os.path.dirname(BASE_DIR), "data")
    os.makedirs(data_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment REAL,
            emotion TEXT,
            created_at TEXT
        )
    """)

    _migrate_entries_table(conn)
    return conn

# ---------------- Auth helpers ----------------
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def add_user(username: str, password: str) -> bool:
    conn = get_db()
    try:
        conn.execute("INSERT INTO users(username, password) VALUES (?, ?)", (username, _hash(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def check_user(username: str, password: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, password FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and row[1] == _hash(password):
        return row[0]
    return None

# ---------------- Data I/O ----------------
def save_entry(user_id: int, text: str, sentiment: float, emotion: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO entries(user_id, text, sentiment, emotion, created_at) VALUES (?,?,?,?,?)",
        (user_id, text, sentiment, emotion, dt.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def fetch_entries(user_id: int, days: int = 7):
    conn = get_db()
    cur = conn.cursor()
    since = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()
    cur.execute("""
        SELECT created_at, sentiment, emotion, text
        FROM entries
        WHERE user_id = ? AND created_at >= ?
        ORDER BY created_at ASC
    """, (user_id, since))
    rows = cur.fetchall()
    conn.close()
    return rows
