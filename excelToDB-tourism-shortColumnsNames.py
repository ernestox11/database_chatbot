import os
import re
import pandas as pd
from sqlalchemy import create_engine, text
import logging
from openai import OpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Retrieve environment variables
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_DATABASE = os.getenv('DB_DATABASE')
DB_PORT = os.getenv('DB_PORT', '3306')  # Default MySQL port
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def snake_case_convert(name):
    """ Convert names to snake_case and lowercase, handling spaces and special characters. """
    # Replace spaces and special characters with underscores
    name = re.sub(r'[\s\-]+', '_', name)
    # Handle camel case by inserting underscores.
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    # Convert to lowercase
    name = name.lower()
    # Remove any double underscores caused by previous replacements
    name = re.sub(r'__+', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    return name

def generate_natural_column_names(columns):
    """ Generates natural, database-appropriate column names using OpenAI and converts them to snake case. """
    try:
        prompt_text = (
            "Translate the following database column names into a comma-separated list of natural, descriptive, and unique names. "
            "Each name should be in plain English, easily understandable, start with a letter or an underscore, and contain only letters, numbers, or underscores. "
            "Ensure all names are distinct and descriptive, avoiding technical or abbreviated formats like snake case. The output should strictly be a comma-separated list with no additional text:\n\n" +
            ", ".join(columns)
        )
        logger.info("Sending prompt to OpenAI: %s", prompt_text)
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": prompt_text}]
        )
        natural_names = response.choices[0].message.content.strip().split(", ")
        # Convert each name to snake_case
        snake_case_names = [snake_case_convert(name) for name in natural_names]
        logger.info("Natural column names generated successfully: %s", snake_case_names)
        return snake_case_names
    except Exception as e:
        logger.error("Failed to generate column names: %s", e)
        return [snake_case_convert(name) for name in columns]  # Fallback to original names if there's an error, converted to snake_case

def setup_database_connection():
    """ Set up database connection using environment variables. """
    try:
        db_uri = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}?charset=utf8mb4&connect_timeout=10&read_timeout=10&write_timeout=10"
        engine = create_engine(db_uri, echo=True)
        logger.info("Database engine created successfully.")
        return engine
    except Exception as e:
        logger.error("Error establishing database connection: %s", e)
        raise

def create_table_from_csv(df, conn, table_name):
    """ Creates a table from a DataFrame with appropriate data type mappings. """
    try:
        dtype_mapping = {
            'int64': 'BIGINT',
            'float64': 'DECIMAL(10,2)',
            'object': 'TEXT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'DATETIME',
            'timedelta[ns]': 'TIME'
        }
        conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
        columns_definitions = ', '.join([f"`{col}` {dtype_mapping[str(df.dtypes[col])]}" for col in df.columns])
        create_table_statement = text(f"CREATE TABLE IF NOT EXISTS `{table_name}` (id INT AUTO_INCREMENT PRIMARY KEY, {columns_definitions})")
        conn.execute(create_table_statement)
        logger.info(f"Table {table_name} created successfully.")
    except Exception as e:
        logger.error("Error creating table %s: %s", table_name, e)
        raise

def insert_data_from_csv(df, conn, table_name):
    """ Inserts data from a DataFrame into the specified table. """
    try:
        df.to_sql(table_name, con=conn, if_exists='append', index=False)
        logger.info(f"Data inserted into table {table_name} successfully.")
    except Exception as e:
        logger.error("Error during data insertion into table %s: %s", table_name, e)
        raise

def read_and_prepare_csv(file_path):
    """ Reads a CSV file, applying encoding and delimiter settings, handling data types explicitly to avoid DtypeWarnings. """
    try:
        df = pd.read_csv(file_path, encoding='utf-8', delimiter=';', low_memory=False)
        df.columns = generate_natural_column_names(df.columns.tolist())
        return df
    except Exception as e:
        logger.error("Failed to read CSV: %s", e)
        raise

# Main execution block
if __name__ == "__main__":
    try:
        engine = setup_database_connection()
        with engine.begin() as conn:
            df = read_and_prepare_csv('tourismfinal.csv')
            create_table_from_csv(df, conn, 'tourism_data')
            insert_data_from_csv(df, conn, 'tourism_data')
        logger.info("Script execution completed successfully.")
    except Exception as e:
        logger.error("An error occurred during the script execution: %s", e)
