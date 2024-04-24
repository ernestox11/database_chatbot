import os
import pandas as pd
from sqlalchemy import create_engine, text, exc
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection details
username = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
database = os.getenv('DB_DATABASE')
# port = os.getenv('DB_PORT')  # Ensure this is set in your .env file

# Establish database connection
try:
    # engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}?connect_timeout=10', echo=False)
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}/{database}?connect_timeout=10', echo=False)
    logger.info("Database engine created successfully.")
except Exception as e:
    logger.error(f"Error creating database engine: {e}")
    raise

csv_file = 'tourismfinal.csv'  # Replace with the path to your CSV file

def create_table_from_csv(df, conn, table_name):
    dtype_mapping = {
        'int64': 'BIGINT',
        'float64': 'DECIMAL(10,2)',
        'object': 'TEXT',  # Changed from VARCHAR(255) to TEXT for demonstration
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'DATETIME',
        'timedelta[ns]': 'TIME'
    }
    # Drop the existing table if it exists using the text() function
    drop_table_statement = text(f"DROP TABLE IF EXISTS `{table_name}`")
    conn.execute(drop_table_statement)
    logger.info(f"Table {table_name} dropped successfully if it existed.")

    # Create the new table using the text() function
    columns_definitions = ', '.join([f"`{col}` {dtype_mapping[str(dtype)]}" for col, dtype in zip(df.columns, df.dtypes)])
    create_table_statement = text(f"CREATE TABLE IF NOT EXISTS `{table_name}` (id INT AUTO_INCREMENT PRIMARY KEY, {columns_definitions})")
    
    logger.info(f"SQL Command for Table Creation: {create_table_statement}")

    try:
        conn.execute(create_table_statement)
        logger.info(f"Table {table_name} created successfully.")
    except exc.SQLAlchemyError as e:
        logger.error(f"Error creating table {table_name}: {e}")
        raise

def insert_data_from_csv(df, conn, table_name):
    try:
        df.to_sql(table_name, con=conn, if_exists='append', index=False)
        logger.info(f"Data inserted into table {table_name} successfully.")
    except ValueError as e:
        logger.error(f"Error during data insertion into table {table_name}: {e}")
        raise

def read_csv_with_encoding(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8', delimiter=';', on_bad_lines='warn')
    except Exception as e:
        logger.error(f"Failed to read CSV: {str(e)}")
        raise

with engine.begin() as conn:
    try:
        df = read_csv_with_encoding(csv_file)
        table_name = 'tourism_data'
        create_table_from_csv(df, conn, table_name)
        insert_data_from_csv(df, conn, table_name)
    except Exception as e:
        logger.error(f"Error during CSV handling or database operations: {e}")

logger.info("Script execution completed.")
