import pymysql

HOST = "localhost"
USER = "root"
PASSWORD = "root"
DATABASE = "smart_queue"

connection = pymysql.connect(
    host=HOST,
    user=USER,
    password=PASSWORD
)

cursor = connection.cursor()

print("Connected to MySQL")

cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
cursor.execute(f"USE {DATABASE}")


create_users_table = """

CREATE TABLE IF NOT EXISTS users(

id INT AUTO_INCREMENT PRIMARY KEY,
name VARCHAR(100),
email VARCHAR(100) UNIQUE,
password VARCHAR(255),
role VARCHAR(20) DEFAULT 'user',
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

)

"""

cursor.execute(create_users_table)

print("Users table created successfully")



create_table = """

CREATE TABLE IF NOT EXISTS appointments(

id INT AUTO_INCREMENT PRIMARY KEY,

user_id INT,

name VARCHAR(100),
email VARCHAR(100),
phone VARCHAR(20),

appointment_date DATE,
hour INT,

token_number INT,
patients_ahead INT,

day_of_week INT,
month INT,
is_weekend INT,

booking_lead_hours FLOAT,
arrival_delay FLOAT,

queue_length INT,
staff INT,

distance_km FLOAT,
urgency VARCHAR(20),

no_show_prob FLOAT,
no_show_prediction INT,

wait_time FLOAT,

status VARCHAR(20) DEFAULT 'pending',

counter_id INT NULL,

created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

FOREIGN KEY (user_id) REFERENCES users(id)

)

"""

cursor.execute(create_table)

print("Appointments table created successfully")




create_counter_table = """

CREATE TABLE IF NOT EXISTS counters(

id INT AUTO_INCREMENT PRIMARY KEY,
counter_number INT,
status VARCHAR(20) DEFAULT 'active',
current_token INT DEFAULT NULL,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

)

"""

cursor.execute(create_counter_table)

print("Counters table created successfully")




cursor.execute("SELECT COUNT(*) FROM counters")
count = cursor.fetchone()[0]

if count == 0:

    cursor.execute("INSERT INTO counters (counter_number) VALUES (1)")
    cursor.execute("INSERT INTO counters (counter_number) VALUES (2)")
    cursor.execute("INSERT INTO counters (counter_number) VALUES (3)")

    print("Default counters added")



cursor.execute("""
SELECT COUNT(1)
FROM INFORMATION_SCHEMA.STATISTICS
WHERE table_schema = %s
AND table_name = 'appointments'
AND index_name = 'idx_hour'
""",(DATABASE,))

if cursor.fetchone()[0] == 0:
    cursor.execute("CREATE INDEX idx_hour ON appointments(hour)")
    print("Index idx_hour created")


cursor.execute("""
SELECT COUNT(1)
FROM INFORMATION_SCHEMA.STATISTICS
WHERE table_schema = %s
AND table_name = 'appointments'
AND index_name = 'idx_status'
""",(DATABASE,))

if cursor.fetchone()[0] == 0:
    cursor.execute("CREATE INDEX idx_status ON appointments(status)")
    print("Index idx_status created")


cursor.execute("""
SELECT COUNT(1)
FROM INFORMATION_SCHEMA.STATISTICS
WHERE table_schema = %s
AND table_name = 'appointments'
AND index_name = 'idx_user'
""",(DATABASE,))

if cursor.fetchone()[0] == 0:
    cursor.execute("CREATE INDEX idx_user ON appointments(user_id)")
    print("Index idx_user created")


connection.commit()

cursor.close()
connection.close()

print("Database setup completed successfully")