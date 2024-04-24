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

def get_sql_chain(db):
    template = """
    You are a data analyst in a tourism company. Your task involves handling queries about the tourism articles database. This database consists of detailed entries about various articles, each entry encompassing data such as article titles, URLs, domains, sentiments, and more detailed categorizations. Your role is to assist users by retrieving specific information based on their queries related to these articles.

    The database structure includes a table 'tourism_data' that captures each article's comprehensive details. Your task is to formulate SQL queries that precisely fetch the data as per the user's request.

    Based on the table schema below and the conversation history, write a SQL query to answer the user's question.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else.

    For example:
    Question: How many articles mentioned 'sustainability' last month?
    SQL Query: SELECT COUNT(*) FROM tourism_data WHERE topics LIKE '%sustainability%' AND publish_date >= DATE_SUB(NOW(), INTERVAL 1 MONTH);
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
    As a data analyst, you are tasked with translating complex database queries into natural language answers that are easy to understand. Here, the user is inquiring about specific tourism-related data stored in our 'tourism_data' table.

    Based on the table schema, the user's question, the SQL query you formulated, and the database's response, craft a response in Spanish that accurately and effectively communicates the needed information.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""

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

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

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
