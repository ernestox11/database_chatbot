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

# Initialize environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

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
    logging.info(f"Database URI: {db_uri}")
    return SQLDatabase.from_uri(db_uri)

# Attempt to establish the database connection
db = init_database(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASE)
if db is not None:
    st.success("Connected to database!")

def get_sql_chain(db):
    def get_schema(_):
        schema = db.get_table_info()
        logging.info(f"Schema: {schema}")
        return schema

    template = """
        You are a data analyst in a tourism company. Your task involves handling queries about the tourism articles database. This database consists of detailed entries about various articles, each entry encompassing data such as article titles, URLs, domains, sentiments, and more detailed categorizations. Your role is to assist users by retrieving specific information based on their queries related to these articles.

        Schema details for reference:
        <SCHEMA>{schema}</SCHEMA>

        Recent user interactions for context:
        {chat_history}

        The database structure includes a table 'tourism_data' that captures each article's comprehensive details. Use the provided schema information and recent conversation history to interpret the user's query. Please infer the relevant database columns from the user's question, which might use natural language or indirect references to the data stored in the database. When formulating SQL queries, ensure column names with spaces or special characters are enclosed in backticks. This helps to avoid syntax errors and ensures your query is executed correctly. Your response should strictly contain only the SQL query, free of any additional formatting or text.
        """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-turbo-preview")

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def clean_sql_query(sql_query):
    # Strip leading and trailing whitespaces and backticks
    cleaned_query = sql_query.strip().strip('`')
    return cleaned_query

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    template = """
    As a data analyst, you are tasked with translating complex database queries into natural language answers that are easy to understand. Here, the user is inquiring about specific tourism-related data stored in our 'tourism_data' table.

    Based on the table schema, your recent query '{question}', the SQL query you formulated, and the database's response, craft a response in Spanish that accurately and effectively communicates the needed information.

    Detailed Schema Information: {schema}
    Recent Conversation Extract: {chat_history}
    Formulated SQL Query: {query}
    Query Execution Result: {response}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-turbo-preview")

    # Prepare the variables for the prompt
    chain_vars = {
        "question": user_query,
        "chat_history": chat_history[-4:],  # Use only the last 4 messages to keep the token count manageable
    }

    # Assign dynamic variables via a lambda function to ensure they are fetched at runtime
    chain = (
        RunnablePassthrough.assign(query=lambda _: sql_chain, schema=lambda _: db.get_table_info())
        | RunnablePassthrough.assign(response=lambda vars: db.run(clean_sql_query(vars["query"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        # Execute the chain to process and handle the query
        response = chain.invoke(chain_vars)
        logging.info(f"Response: {response}")
        return response
    except SQLAlchemyError as e:
        # Handle SQL execution errors
        error_message = "An error occurred while processing your query: " + str(e)
        st.error(error_message)
        logging.error(f"SQLAlchemy Error: {error_message}")
        return "An error occurred while processing your query. Please check the query and try again."
    except Exception as e:
        # Handle other types of errors
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
