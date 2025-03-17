import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Missing API key. Check your .env file.")
    st.stop()

# Set up Streamlit UI
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align: center;'>AI Assistant 🤖</h1>", unsafe_allow_html=True)
st.subheader("How can I assist you today?")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Task selection
task_type = st.selectbox("Select a task:", ["General Question", "Task Management", "Cooking Advice"])

# User input
input_text = st.text_input(f"Enter your {task_type.lower()} request:")

# Set up the AI model
groqApi = ChatGroq(model="gemma2-9b-it", temperature=1)
output_parser = StrOutputParser()

# Define system prompts for each task
task_prompts = {
    "General Question": "You are a knowledgeable AI assistant.",
    "Task Management": "You help manage and organize tasks efficiently.",
    "Cooking Advice": "You are a chef, providing cooking tips and recipes."
}

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", task_prompts[task_type]),
    ("user", "{query}")
])

chain = prompt | groqApi | output_parser

# Process user request
if st.button("Submit"):
    if input_text:
        try:
            result = chain.invoke({"query": input_text})
            st.write(result)

            # Save chat history
            st.session_state.chat_history.append({"role": "User", "content": input_text})
            st.session_state.chat_history.append({"role": "AI", "content": result})
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid input.")

# Display chat history
st.markdown("### Conversation History")
chat_display = "\n".join([f"**{msg['role']}**: {msg['content']}" for msg in st.session_state.chat_history])
st.text_area("Chat History", value=chat_display, height=300, disabled=True)
