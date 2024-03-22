import pandas as pd
import re
from sqlalchemy import create_engine, text

# Database connection details
username = 'luis'
password = 'ernecio1234'
host = 'localhost'
database = 'pruebas'
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}/{database}')

excel_file = 'encuesta_completa.xlsx'

def create_tables(conn):
    # Create tables for storing questions and responses
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS questions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        question_text VARCHAR(1024) UNIQUE NOT NULL
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
    # Regex pattern to match 'P' followed by a number
    pattern = r'^P\d+'
    for col in df.columns:
        # Check if column name matches the pattern
        if re.match(pattern, col):
            conn.execute(text("""
            INSERT INTO questions (question_text)
            VALUES (:question)
            ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id), question_text=VALUES(question_text)
            """), {'question': col})
            result = conn.execute(text("SELECT LAST_INSERT_ID()"))
            question_id = result.fetchone()[0]
            question_ids[col] = question_id
    return question_ids

def insert_responses(df, question_ids, conn):
    for index, row in df.iterrows():
        for question, response in row.items(): 
            if question in question_ids:  # Check if the question was inserted
                # Convert NaN values to None
                if pd.isnull(response):
                    response = None
                question_id = question_ids[question]
                conn.execute(text("""
                INSERT INTO survey_responses (question_id, response)
                VALUES (:question_id, :response)
                """), {'question_id': question_id, 'response': response})

with engine.begin() as conn:
    create_tables(conn)
    
    xls = pd.ExcelFile(excel_file)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Insert question names and get their IDs if they match the pattern
        question_ids = insert_questions_get_ids(df, conn)
        
        # Insert responses for each matching question
        insert_responses(df, question_ids, conn)
