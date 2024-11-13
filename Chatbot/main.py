import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

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

@cl.on_message
async def assistant(message: cl.Message):
    input_text = message.content
    groq_llm = ChatGroq(model="gemma-7b-It", temperature=2)
    output = StrOutputParser()
    chain = prompt_instance | groq_llm | output

    await cl.Message(content="Processing your question...").send()
    
    try:
        res = await chain.ainvoke({'question': input_text})
        await cl.Message(content=res).send()
    except Exception as e:
        await cl.Message(content=f"Error processing your request: {str(e)}").send()

if __name__ == "__main__":
    # Get the port from the environment variable or use 8000 as a default
    port = int(os.getenv("PORT", "8000"))
    # Print statements to help with debugging in deployment logs
    print("Attempting to bind to host: 0.0.0.0")
    print("Attempting to bind to port:", port)
    
    # Run Chainlit, binding to 0.0.0.0 and using the specified port
    cl.run(host="0.0.0.0", port=port)
    # Send a welcome message after the app starts
    cl.Message(content="Welcome to the Dental Assistant Chatbot! How can I assist you today?").send()
