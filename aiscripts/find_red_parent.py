import sqlite3
import os

DB_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/parent_chunks.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

query = "%256 (19d12 + 133)%"
c.execute("SELECT id, content FROM parent_chunks WHERE content LIKE ?", (query,))
rows = c.fetchall()
for row in rows:
    print(f"ID: {row[0]}")
    # print(row[1][:100])
conn.close()
