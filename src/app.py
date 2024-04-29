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
    You are a data analyst in a tourism company, responsible for handling queries regarding a database that tracks various articles. Each article entry includes details such as titles, URLs, domains, sentiments, and other categorizations relevant to tourism.

    The structure of the 'tourism_data' table includes various fields that store these details. Your task is to understand user queries, which may not directly mention specific column names, and translate them into SQL queries that fetch the required data.

    Please infer the relevant database columns from the user's question, which might use natural language or indirect references to the data stored in the database. Use your understanding of the schema and the context provided by the user's conversation history to craft precise SQL queries.

    Here is the schema for your reference:
    <SCHEMA>{schema}</SCHEMA>

    Recent user interactions for context:
    {chat_history}

    Respond with the SQL query alone, ensuring it is syntactically correct and relevant to the user's inquiry.
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
    As a data analyst, you are tasked with translating complex database queries into natural language answers that are easy to understand. Here, the user is inquiring about specific tourism-related data stored in our 'tourism_data' table.

    Based on the table schema, your recent query '{question}', the SQL query you formulated, and the database's response, craft a response in Spanish that accurately and effectively communicates the needed information.

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
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        # Execute the chain to process and handle the query
        response = chain.invoke({
            "question": user_query,
            "chat_history": chat_history[-4:],  # Use only the last 4 messages to avoid excessive token usage
        })
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
