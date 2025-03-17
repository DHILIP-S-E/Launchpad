import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# Load environment variables from .env file
load_dotenv()

# Set up the Groq API key from the environment
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing from the environment.")

# ChatGPT-like prompt template
prompt_template = """
    (system: You are a helpful assistant capable of answering a variety of questions. Provide detailed and informative responses.)
    (user: {question})
"""
prompt_instance = ChatPromptTemplate.from_template(prompt_template)

async def send_welcome_message():
    await cl.Message(content="Welcome to the AI Assistant! How can I assist you today?").send()

@cl.on_message
async def assistant(message: cl.Message):
    input_text = message.content
    groq_llm = ChatGroq(model="mixtral-8x7b-32768", temperature=2, api_key=groq_api_key)  # ✅ Updated Model

    output = StrOutputParser()
    chain = prompt_instance | groq_llm | output

    await cl.Message(content="Processing your question...").send()

    try:
        # Asynchronous invocation of the chain
        res = await chain.ainvoke({'question': input_text})
        await cl.Message(content=res).send()
    except Exception as e:
        await cl.Message(content=f"Error processing your request: {str(e)}").send()

# Run the chatbot
if __name__ == "__main__":
    cl.run()
