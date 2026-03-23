import sqlite3

conn = sqlite3.connect("mindscan.db")  # your DB file
cursor = conn.cursor()

cursor.execute("SELECT * FROM users")
users = cursor.fetchall()

print("\n--- USERS DATA ---\n")

for user in users:
    print(user)

conn.close()