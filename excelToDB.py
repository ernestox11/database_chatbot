import os
import re
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection details from environment variables
username = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
database = os.getenv('DB_DATABASE')

try:
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}/{database}', echo=True)
    logger.info("Database engine created successfully.")
except Exception as e:
    logger.error(f"Error creating database engine: {e}")

excel_file = 'encuesta.xlsx'
excel_file = 'encuesta_completa.xlsx'

def excel_col_index_to_letter(index):
    letter = ''
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letter = chr(65 + remainder) + letter
    return letter

def clean_question_text(question_text):
    """
    Cleans the question text by removing the initial 'P' followed by digits
    and optionally an underscore and additional characters at the beginning.
    """
    return re.sub(r'^P\d+_?', '', question_text).strip()

def create_tables(conn):
    try:
        # Drop tables if they exist
        conn.execute(text("DROP TABLE IF EXISTS survey_responses;"))
        conn.execute(text("DROP TABLE IF EXISTS questions;"))
        
        # Recreate tables
        conn.execute(text("""
        CREATE TABLE questions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question_text TEXT NOT NULL,
            excel_column_position VARCHAR(5)
        );
        """))
        conn.execute(text("""
        CREATE TABLE survey_responses (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question_id INT,
            response TEXT,
            FOREIGN KEY (question_id) REFERENCES questions(id)
        );
        """))
        logger.info("Tables created successfully.")
    except SQLAlchemyError as e:
        logger.error(f"Error creating tables: {e}")


def insert_questions_get_ids(df, conn):
    question_ids = {}
    data_to_insert = []
    for index, col in enumerate(df.columns):
        col_letter = excel_col_index_to_letter(index + 1)
        original_col = col  # Keep the original column name
        cleaned_col = clean_question_text(col)  # Cleaned column name for insertion
        if re.match(r'^P\d+', col):
            data_to_insert.append({'original_question': original_col, 'question': cleaned_col, 'col_letter': col_letter})

    if data_to_insert:
        try:
            for item in data_to_insert:
                conn.execute(text("""
                INSERT INTO questions (question_text, excel_column_position)
                VALUES (:question, :col_letter)
                ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id), question_text=VALUES(question_text), excel_column_position=VALUES(excel_column_position)
                """), {'question': item['question'], 'col_letter': item['col_letter']})
                result = conn.execute(text("SELECT id FROM questions WHERE question_text = :question"), {'question': item['question']})
                question_id = result.fetchone()[0]
                question_ids[item['original_question']] = question_id
                logger.info(f"Question '{item['question']}' inserted or updated with ID {question_id}.")
        except SQLAlchemyError as e:
            logger.error(f"Error inserting questions: {e}")
    return question_ids

def insert_responses(df, question_ids, conn):
    data_to_insert = []
    for index, row in df.iterrows():
        for question, response in row.items():
            if question in question_ids:
                response = response if not pd.isnull(response) else None
                data_to_insert.append({'question_id': question_ids[question], 'response': response})
                
        # Insert responses in batches to optimize performance
        if data_to_insert:
            try:
                conn.execute(text("""
                INSERT INTO survey_responses (question_id, response)
                VALUES (:question_id, :response)
                """), data_to_insert)
                data_to_insert = []  # Reset for next batch
                logger.info(f"Responses for row {index+1} inserted.")
            except SQLAlchemyError as e:
                logger.error(f"Error inserting responses for row {index+1}: {e}")

with engine.begin() as conn:
    create_tables(conn)
    xls = pd.ExcelFile(excel_file)
    for sheet_name in xls.sheet_names:
        logger.info(f"Starting processing of sheet: {sheet_name}")
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            question_ids = insert_questions_get_ids(df, conn)
            insert_responses(df, question_ids, conn)
            logger.info(f"Completed processing sheet: {sheet_name}")
        except Exception as e:
            logger.error(f"Error processing sheet '{sheet_name}': {e}")
