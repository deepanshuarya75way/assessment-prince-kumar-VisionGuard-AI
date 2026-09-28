import sqlite3
from pathlib import Path 

DB_PATH = Path(__file__).resolve().parent.parent / "quality_loop.db"

def get_connection():
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row 
  return conn

def init_db():
  conn = get_connection()
  cursor = conn.cursor()

  cursor.execute("""
     CREATE TABLE IF NOT EXISTS predictions(
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       image_Path TEXT NOT NULL,
       model_version TEXT NOT NULL,
       predicted_label TEXT NOT NULL,
       confidence REAL,
       created_at TIMESTAMP DEFAULT
CURREENT_TIMESTAMP
    ) 
  """)

  cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews(
    id  INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    review_status TEXT NOT NULL,
    human_label TEXT,
    comment TEXT,
    reviewer TEXT,
    created_at TIMESTAMP DEFAULT
CURRENT_TIMESTAMP,
     FOREIGN KEY (prediction_id)
       REFERENCES predictions(id)
   )
  """)

  cursor.execute("""
   CREATE TABLE IF NOT EXISTS dataset_items(
   id INTEGER PRIMARY KEY AUTOINCREMENT,
   prediction_id INTEGER NOT NULL UNIQUE,
   included INTEGER NOT NULL DEFAULT 0,
   created_at TIMESTAMP DEFAULT
CURRENT_TIMESTAMP,
    FOREIGEN KEY (prediction_id)
      REFERENCES predictions(id)
   )
  """)

  conn.commit()
  conn.close()

if __name__ == "__main__":
  init_db()
  print(f"Database initialized at:{DB_PATH}")