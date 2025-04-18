import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

# Configuración de la conexión
host = os.getenv('DB_HOST')
port = '5432'
database = 'operaciones2025'
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
encoding = 'utf8'

def get_connection():
    """Establece y devuelve una conexión a la base de datos PostgreSQL."""
    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            client_encoding=encoding
        )
        print("Conexión exitosa")
        return connection
    except psycopg2.Error as e:
        print(f"Error en la conexión: {e.pgcode} - {e.pgerror}")
        return None