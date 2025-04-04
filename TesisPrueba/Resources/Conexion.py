import psycopg2

host = 'localhost'
port = '5432'
database = 'Operaciones2024'
user = 'postgres'
#password = 'jean31'
password = 'admin123'

def get_connection():
    connection = None
    try:
        connection = psycopg2.connect(
            host = host,
            port = port,
            database = database,
            user = user,
            password = password
        )
        print("Conexión exitosa")
    except psycopg2.Error as e:
        print("Error en la conexión:", e)
    return connection
