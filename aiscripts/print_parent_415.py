import sqlite3
import os

DB_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/parent_chunks.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("SELECT content FROM parent_chunks WHERE id=?", ("371143968f4537998af5502c02d2097b_parent_415",))
row = c.fetchone()
if row:
    print(row[0])
else:
    print("Parent not found")
conn.close()
