import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sqlalchemy.exc import SQLAlchemyError

# Initialize environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

# Extract environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_DATABASE = os.getenv("DB_DATABASE")

# Function to initialize and cache the database connection
@st.cache_resource
def init_database():
    """Initialize and cache the database connection."""
    db_connection_string = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
    return SQLDatabase.from_uri(db_connection_string)

db = init_database()
if db:
    st.success("Connected to database!")

def get_sql_chain():
    """Generate SQL query based on user input using AI."""
    sql_template = """
    You are a data analyst in a tourism company. Your task involves handling queries about the tourism articles database. This database consists of detailed entries about various articles, each entry encompassing data such as article titles, URLs, domains, sentiments, and more detailed categorizations. Your role is to assist users by retrieving specific information based on their queries related to these articles.

    The database structure includes a table 'tourism_data' that captures each article's comprehensive details. Your task is to formulate SQL queries that precisely fetch the data as per the user's request.

    Based on the table schema below and the conversation history, write a SQL query to answer the user's question.

    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}

    Reply only with the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    """
    schema = db.get_table_info()
    prompt = ChatPromptTemplate.from_template(sql_template)
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    return (RunnablePassthrough.assign(schema=schema) | prompt | llm | StrOutputParser())

def get_response(user_query: str, chat_history: list):
    """Process user query and provide a human-friendly response."""
    sql_chain = get_sql_chain()
    response_template = """
    As a data analyst, you are tasked with translating complex database queries into natural language answers that are easy to understand. Here, the user is inquiring about specific tourism-related data stored in our 'tourism_data' table.

    Based on the table schema, the user's question, the SQL query you formulated, and the database's response, craft a response in Spanish that accurately and effectively communicates the needed information.

    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    response_prompt = ChatPromptTemplate.from_template(response_template)
    llm_response = ChatOpenAI(model="gpt-4-turbo-preview")
    response_chain = (RunnablePassthrough.assign(query=sql_chain) | response_prompt | llm_response | StrOutputParser())
    return response_chain.invoke({
        "schema": db.get_table_info(),
        "chat_history": chat_history,
        "query": sql_chain,
        "question": user_query
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I'm your assistant. Ask me any questions about our tourism database.")]

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
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    response = get_response(user_query, st.session_state.chat_history)
    with st.chat_message("AI"):
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))
