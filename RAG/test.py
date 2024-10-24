import os
import streamlit as st
from langchain.document_loaders import GoogleDriveLoader
from langchain.llms import OpenAI  # Adjust this based on your LLM
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Set up Streamlit UI
st.title("Document Interaction with LLMs")
st.write("Upload a document from Google Drive and interact with it using LLMs.")

# Google Drive Document ID Input
document_id = st.text_input("Enter Google Drive Document ID:", "")

# Load Document Button
if st.button("Load Document"):
    if document_id:
        with st.spinner("Loading document..."):
            try:
                # Load document from Google Drive
                loader = GoogleDriveLoader(
                    document_ids=[document_id],
                    credentials_path=r"C:\Users\DHILIP\.credentials\client_secret_753799051727-fcsk6q8ckmveisgg1ii3memja9rrbhsj.apps.googleusercontent.com.json"
                )
                docs = loader.load()
                st.success("Document loaded successfully!")

                # Display the loaded document
                for doc in docs:
                    st.write(doc.page_content)

                # Summarization
                if st.button("Summarize Document"):
                    with st.spinner("Summarizing document..."):
                        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Adjust based on your needs
                        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
                        summary = chain.run(docs)
                        st.subheader("Summary:")
                        st.write(summary)

                # Question-Answering
                query = st.text_input("Ask a question about the document:")
                if st.button("Get Answer"):
                    if query:
                        with st.spinner("Getting answer..."):
                            qa_chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
                            answer = qa_chain.run(input_documents=docs, question=query)
                            st.subheader("Answer:")
                            st.write(answer)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid Google Drive document ID.")
