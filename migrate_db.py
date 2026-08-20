import sqlite3

con = sqlite3.connect("data/app.db")
cur = con.cursor()
cols = [r[1] for r in cur.execute("PRAGMA table_info(users)").fetchall()]
if "assigned_store" not in cols:
    cur.execute("ALTER TABLE users ADD COLUMN assigned_store VARCHAR(10)")
cur.execute("UPDATE users SET assigned_store = '4' WHERE username = 'manager1'")
cur.execute("UPDATE users SET assigned_store = '1' WHERE username = 'manager'")
con.commit()
con.close()
print("Database schema migration successful!")
