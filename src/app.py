from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sqlalchemy.exc import SQLAlchemyError
import os
import logging
import re

# Initialize environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set Streamlit page configuration
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

# Database connection settings
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_DATABASE = os.getenv("DB_DATABASE")

# Function to initialize database connection
@st.cache_resource
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    logging.info(f"Database URI: {db_uri}")
    return SQLDatabase.from_uri(db_uri)

# Establish the database connection
db = init_database(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASE)
if db is not None:
    st.success("Connected to database!")

def clean_sql_query(sql_query):
    # Regex to detect and remove any unwanted 'sql' prefix or surrounding backticks
    cleaned_query = re.sub(r"^\s*sql\s+", "", sql_query, flags=re.IGNORECASE).strip()
    # Remove backticks that might incorrectly wrap the entire query
    cleaned_query = re.sub(r"^`(.*)`$", r"\1", cleaned_query).strip()
    return cleaned_query

def get_sql_chain(db):
    def get_schema(_):
        schema = db.get_table_info()
        logging.info(f"Schema: {schema}")
        return schema

    template = """
    You are a data analyst in a tourism company. Your task involves handling queries about the tourism articles database. This database consists of detailed entries about various articles, each entry encompassing data such as article titles, URLs, domains, sentiments, and more detailed categorizations. Your role is to assist users by retrieving specific information based on their queries related to these articles.

    Use the provided schema information and recent conversation history to interpret the user's query. Generate a relevant SQL query by inferring the required database columns from the user's question.

    Schema details for reference:
    <SCHEMA>{schema}</SCHEMA>

    Recent user interactions for context:
    {chat_history}

    Start your response directly with 'SELECT', 'INSERT', 'UPDATE', or 'DELETE' without any labels or additional text.
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-turbo-preview")

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    template = """
    Translate the SQL query into a response in Spanish that accurately communicates the needed information based on the user's query about the tourism-related data stored in the 'tourism_data' table.

    Detailed Schema Information: {schema}
    Recent Conversation Extract: {chat_history}
    Formulated SQL Query: {query}
    Query Execution Result: {response}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-turbo-preview")

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(clean_sql_query(vars["query"])),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        response = chain.invoke({
            "question": user_query,
            "chat_history": chat_history[-4:]  # To manage token usage
        })
        logging.info(f"Response: {response}")
        return response
    except SQLAlchemyError as e:
        error_message = "An error occurred while processing your query: " + str(e)
        st.error(error_message)
        logging.error(f"SQLAlchemy Error: {error_message}")
        return "An error occurred while processing your query. Please check the query and try again."
    except Exception as e:
        error_message = "An unexpected error occurred: " + str(e)
        st.error(error_message)
        logging.error(f"General Error: {error_message}")
        return "An unexpected error occurred. Please try again later."

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your assistant. Ask me any questions about our tourism database."),
    ]

st.title("Chat with MySQL")

# Chat interface to display messages and handle user input
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    response = get_response(user_query, db, st.session_state.chat_history)
    with st.chat_message("AI"):
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))
