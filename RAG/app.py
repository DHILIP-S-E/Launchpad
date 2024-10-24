import os
from flask import Flask, request, render_template
from langchain_community.document_loaders import GoogleDriveLoader  # Adjusted import for document loading
from langchain_community.chains import load_summarize_chain  # Adjusted import for summarization
from langchain_community.chat_models import ChatGroq  # Adjusted import for the Chat model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Render the main HTML template

@app.route('/upload', methods=['POST'])
def upload():
    document_id = request.form['document_id']
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # Load credentials from environment variable

    try:
        # Load the document from Google Drive
        loader = GoogleDriveLoader(document_ids=[document_id], credentials_path=credentials_path)
        docs = loader.load()

        # Debugging output to check loaded documents
        print(f"Loaded documents: {docs}")
        print(f"Type of first document: {type(docs[0])}")  # Print the type of the first document

        # Initialize the summarization chain
        groqllm = ChatGroq(model="llama3-70b-8192", temperature=0)
        summary_chain = load_summarize_chain(groqllm, chain_type="map_reduce", verbose=True)

        # Check if docs is a list and not empty
        if isinstance(docs, list) and docs:
            # Assuming docs contain document objects with 'page_content'
            page_content = []
            for doc in docs:
                if hasattr(doc, 'page_content'):
                    page_content.append(doc.page_content)
                else:
                    # Handle if the document does not have 'page_content'
                    return render_template('result.html', summary="Document does not have recognized content structure.")

            # Summarize the documents
            summary = summary_chain.run(page_content)  # Summarize page contents
            return render_template('result.html', summary=summary)
        else:
            return render_template('result.html', summary="Loaded documents are not in a recognized format.")

    except Exception as e:
        return render_template('result.html', summary=f"An error occurred: {str(e)}")  # Display error message

if __name__ == '__main__':
    app.run(debug=True)
