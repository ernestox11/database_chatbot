import re
import pandas as pd
from sqlalchemy import create_engine, text

# Database connection details
username = 'luis'
password = 'ernecio1234'
host = 'localhost'
database = 'pruebas'
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}/{database}')

excel_file = 'encuesta_completa.xlsx'

def excel_col_index_to_letter(index):
    """Convert a zero-indexed column number to an Excel column letter."""
    letter = ''
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letter = chr(65 + remainder) + letter
    return letter

def create_tables(conn):
    # Adjusted to include the new column
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS questions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        question_text VARCHAR(1024) UNIQUE NOT NULL,
        excel_column_position VARCHAR(5)
    )
    """))
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS survey_responses (
        id INT AUTO_INCREMENT PRIMARY KEY,
        question_id INT,
        response TEXT,
        FOREIGN KEY (question_id) REFERENCES questions(id)
    )
    """))

def insert_questions_get_ids(df, conn):
    question_ids = {}
    for index, col in enumerate(df.columns):
        col_letter = excel_col_index_to_letter(index + 1)
        if re.match(r'^P\d+', col):
            conn.execute(text("""
            INSERT INTO questions (question_text, excel_column_position)
            VALUES (:question, :col_letter)
            ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id), question_text=VALUES(question_text), excel_column_position=VALUES(excel_column_position)
            """), {'question': col, 'col_letter': col_letter})
            result = conn.execute(text("SELECT LAST_INSERT_ID()"))
            question_id = result.fetchone()[0]
            question_ids[col] = question_id
    return question_ids

def insert_responses(df, question_ids, conn):
    for index, row in df.iterrows():
        for question, response in row.items():
            if question in question_ids:  # Ensure the question is one we've inserted
                response = response if not pd.isnull(response) else None
                conn.execute(text("""
                INSERT INTO survey_responses (question_id, response)
                VALUES (:question_id, :response)
                """), {'question_id': question_ids[question], 'response': response})

with engine.begin() as conn:
    create_tables(conn)
    xls = pd.ExcelFile(excel_file)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        question_ids = insert_questions_get_ids(df, conn)
        insert_responses(df, question_ids, conn)
