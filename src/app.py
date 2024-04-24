from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

# Initialize environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

# Database connection settings (extracted from environment variables for security)
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_DATABASE = os.getenv("DB_DATABASE")

# Function to initialize database connection
@st.cache_resource
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# Attempt to establish the database connection
db = init_database(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASE)
if db is not None:
    st.success("Connected to database!")
else:
    st.error("Failed to connect to the database.")

def get_sql_chain(db):
    template = """
    You are a data analyst with a bilingual database schema. Based on the conversation history and understanding both English and Spanish column names, formulate a SQL query to retrieve data as requested by the user.

    The table 'tourism_data' contains columns in both English and Spanish. Use this dual-language schema to accurately determine which columns are relevant to the user's question and generate an appropriate SQL query.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Only write the SQL query.
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-turbo-preview")

    def get_schema(_):
        schema = db.get_table_info()
        print("Database Schema:", schema)  # Console log for debugging
        return schema

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    try:
        sql_chain = get_sql_chain(db)
        sql_query = sql_chain.invoke({"chat_history": chat_history})
        print("Generated SQL Query:", sql_query)  # Console log for debugging

        response = db.run(sql_query)
        return response
    except Exception as e:
        print("Error running SQL query:", e)  # Console log for debugging
        return "Sorry, I encountered an issue with your request. Could you please specify it differently?"

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
