from fastapi import FastAPI
import sqlite3

app = FastAPI()

DATABASE: str = "bank.db"

def get_db_connection():
    """Get a connection to the current SQLite database."""
    return sqlite3.connect(DATABASE)

def init_db():
    """Initialize the SQLite database and create the table if it doesn't exist."""
    conn: sqlite3.Connection = get_db_connection()
    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bank (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Initialize the database
init_db()
