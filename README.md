My personal portfolio website created using the Astro Blog Template!

import pandas as pd
import psycopg2

# Create a pandas dataframe
data = {
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
}
df = pd.DataFrame(data)

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    host="localhost",
    database="mydatabase",
    user="myusername",
    password="mypassword"
)

# Create a new table in the database
cur = conn.cursor()
cur.execute("CREATE TABLE mytable (id SERIAL PRIMARY KEY, name VARCHAR, age INTEGER)")

# Export the dataframe to the database
for index, row in df.iterrows():
    cur.execute("INSERT INTO mytable (name, age) VALUES (%s, %s)", (row['name'], row['age']))

# Commit the changes and close the connection
conn.commit()
cur.close()
conn.close()

