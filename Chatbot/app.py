import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load the .env file to retrieve API keys securely
load_dotenv()

# Retrieve API keys from the environment
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the necessary environment variables are set
if langchain_api_key is None:
    st.error("Error: The environment variable 'LANGCHAIN_API_KEY' is not set. Please check your .env file.")
    raise ValueError("LANGCHAIN_API_KEY is missing")
if groq_api_key is None:
    st.error("Error: The environment variable 'GROQ_API_KEY' is not set. Please check your .env file.")
    raise ValueError("GROQ_API_KEY is missing")

# Streamlit UI
st.set_page_config(page_title="Personal Assistant", page_icon="🤖", layout="wide")
st.title("Personal Assistant 🤖")
st.subheader("How can I assist you today?")

# Task Selector
task_type = st.selectbox("Select task type:", ["General Question", "Reminder", "Task Management", "Cooking Advice"])

# Get user input
input_text = st.text_input(f"Ask a {task_type.lower()} question or provide details:", key="user_input")

# Using the Groq inference engine
groqApi = ChatGroq(model="gemma-7b-It", temperature=1)
output_parser = StrOutputParser()

# Prompt-based chain selection
if task_type == "General Question":
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a knowledgeable assistant capable of answering any general questions."),
         ("user", "Question:{question}")]
    )
elif task_type == "Reminder":
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a personal assistant that helps set reminders."),
         ("user", "Reminder:{reminder}")]
    )
elif task_type == "Task Management":
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an assistant that helps manage and organize tasks."),
         ("user", "Task:{task}")]
    )
else:  # Cooking Advice
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a chef, offering cooking advice and tips."),
         ("user", "Cooking question:{question}")]
    )

chain = prompt | groqApi | output_parser

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to add message to chat history
def add_to_chat_history(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

# If user input is provided, process the input using the LLM
if st.button("Enter"):
    if input_text:
        try:
            # Construct the input based on task type
            if task_type == "General Question":
                result = chain.invoke({'question': input_text})
            elif task_type == "Reminder":
                result = chain.invoke({'reminder': input_text})
            elif task_type == "Task Management":
                result = chain.invoke({'task': input_text})
            else:
                result = chain.invoke({'question': input_text})

            # Display the result in Streamlit
            st.write(result)

            # Add user input and result to chat history
            add_to_chat_history("user", input_text)
            add_to_chat_history("assistant", result)

            # Clear the input field after submission
            st.session_state.user_input = ""  # Clear input field

        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
    else:
        st.info(f"Please enter your {task_type.lower()} details.")

# Display chat history
st.markdown("### Chat History")
for message in st.session_state.chat_history:
    st.write(f"**{message['role'].capitalize()}**: {message['content']}")
