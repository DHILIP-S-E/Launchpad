import os
import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Set up the Groq API key from the environment
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Updated ChatGPT-like prompt
prompt = """
    (system: You are a helpful assistant capable of answering a variety of questions. Provide detailed and informative responses.),
    (user: {question})
"""
prompt_instance = ChatPromptTemplate.from_template(prompt)

# The app should bind to all interfaces
host = "0.0.0.0"
# Get the PORT from the environment variable or default to 8000
port = int(os.getenv("PORT", 8000))

# Set up the Groq LLM model
groq_llm = ChatGroq(model="gemma-7b-It", temperature=2)
output = StrOutputParser()

# Chain for processing
chain = prompt_instance | groq_llm | output

@cl.on_message
async def assistant(message: cl.Message):
    input_text = message.content

    # Send a message to indicate processing
    await cl.Message(content="Processing your question...").send()
    
    try:
        # Invoke the chain and get the response
        res = await chain.ainvoke({'question': input_text})
        await cl.Message(content=res).send()
    except Exception as e:
        await cl.Message(content=f"Error processing your request: {str(e)}").send()

if __name__ == "__main__":
    # Welcome message displayed when the chatbot starts
    cl.Message(content="Welcome to the Dental Assistant Chatbot! How can I assist you today?").send()

    # Run the Chainlit app with the specified host and port
    print(f"Binding to {host} on port {port}...")
    cl.run(host=host, port=port)
