import sqlite3
import os

SQLITE_DB_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/parent_chunks.db"

conn = sqlite3.connect(SQLITE_DB_PATH)
c = conn.cursor()

parent_id = "371143968f4537998af5502c02d2097b_parent_107"
c.execute("SELECT content FROM parent_chunks WHERE id=?", (parent_id,))
row = c.fetchone()
if row:
    print(f"--- Content of {parent_id} ---")
    print(row[0])
else:
    print(f"Parent chunk {parent_id} not found.")

conn.close()
