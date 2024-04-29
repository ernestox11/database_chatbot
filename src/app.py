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

def get_sql_chain(db):
    template = """
    You are a data analyst tasked with creating SQL queries based on user requests. Each request pertains to data stored in a 'tourism_data' table which includes various columns like 'Article Title', 'Creation Date', etc. This database consists of detailed entries about various articles.

    Use the provided schema information and recent conversation history to interpret the user's query. Generate a relevant SQL query by inferring the required database columns from the user's question. Make sure the query starts directly with 'SELECT', 'INSERT', 'UPDATE', or 'DELETE', and does not include any extraneous prefixes or text.

    **Schema Reference**:
    <SCHEMA>{schema}</SCHEMA>

    **Recent Queries for Context**:
    {chat_history}

    Ensure your response contains only the SQL query, correctly formatted for MySQL, especially ensuring proper use of backticks for column names with spaces or special characters.
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-turbo-preview")

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    template = """
    Translate the SQL query into a response in Spanish that communicates the information clearly based on the user's query about the 'tourism_data' table.

    **Detailed Schema Information**:
    {schema}

    **Recent Conversation Extract**:
    {chat_history}

    **Formulated SQL Query**:
    {query}

    **Query Execution Result**:
    {response}
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
            "chat_history": chat_history,
        })
        return response
    except SQLAlchemyError as e:
        # Handle SQL execution errors
        error_message = "An error occurred while processing your query: " + str(e)
        st.error(error_message)
        return "An error occurred while processing your query. Please check the query and try again."
    except Exception as e:
        # Handle other types of errors
        error_message = "An unexpected error occurred: " + str(e)
        st.error(error_message)
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
