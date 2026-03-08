"""
Main Application File - AI Lawyer (RAG based system)

This file performs the complete pipeline for the application:

1. Upload a PDF
2. Extract text from the PDF
3. Split the text into smaller chunks
4. Convert text chunks into embeddings
5. Store embeddings inside a FAISS vector database
6. Retrieve relevant chunks based on user query
7. Send context + query to LLM to generate the answer

Technologies Used:
- Streamlit (Frontend UI)
- LangChain (RAG pipeline)
- Ollama Embeddings (Vector generation)
- FAISS (Vector database)
- OpenAI GPT model (Answer generation)
"""

# -------------------------------------------------------------
# Import required libraries
# -------------------------------------------------------------

import streamlit as st

# Loader to extract text from PDF files
from langchain_community.document_loaders import PDFPlumberLoader

# Used to split large documents into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding model running locally using Ollama
from langchain_ollama import OllamaEmbeddings

# Prompt template used for formatting LLM input
from langchain_core.prompts import ChatPromptTemplate

# Vector database used to store embeddings
from langchain_community.vectorstores import FAISS

# LLM used to generate final answers
from langchain_openai import ChatOpenAI


# -------------------------------------------------------------
# Custom Prompt Template
# -------------------------------------------------------------
# This prompt ensures the LLM answers ONLY using retrieved context
# and does not hallucinate or invent information.

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Question: {question} 
Context: {context} 

Answer:
"""


# -------------------------------------------------------------
# Model and Path Configuration
# -------------------------------------------------------------

# Local embedding model running through Ollama
ollama_model_name = "deepseek-r1:1.5b"#"deepseek-r1:14b"

# Initialize embedding model
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b") #"deepseek-r1:14b")

# Path where FAISS vector database will be stored
FAISS_DB_PATH = "vectorstore/db_faiss"

# Directory where uploaded PDFs will be saved
pdfs_directory = "pdfs/"

# Initialize OpenAI LLM for answer generation
llm_model = ChatOpenAI(model="gpt-4o-mini")


# -------------------------------------------------------------
# Function: Upload PDF
# -------------------------------------------------------------
# Saves the uploaded PDF file to the local directory

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


# -------------------------------------------------------------
# Function: Load PDF
# -------------------------------------------------------------
# Reads the PDF file and converts it into LangChain documents

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


# -------------------------------------------------------------
# Function: Create Text Chunks
# -------------------------------------------------------------
# Large documents are split into smaller pieces so that
# embeddings and retrieval become more efficient.

def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,       # Maximum size of each chunk
        chunk_overlap=20,     # Overlap between chunks to preserve context
        add_start_index=True
    )

    text_chunks = text_splitter.split_documents(documents)

    return text_chunks


# -------------------------------------------------------------
# Function: Get Embedding Model
# -------------------------------------------------------------
# Returns an Ollama embedding model instance

def get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings


# -------------------------------------------------------------
# Function: Create Vector Store
# -------------------------------------------------------------
# Converts document chunks into embeddings and stores them
# inside a FAISS vector database.

def create_vector_store(db_faiss_path, text_chunks, ollama_model_name):

    faiss_db = FAISS.from_documents(
        text_chunks,
        get_embedding_model(ollama_model_name)
    )

    # Save vector database locally
    faiss_db.save_local(db_faiss_path)

    return faiss_db


# -------------------------------------------------------------
# Function: Retrieve Relevant Documents
# -------------------------------------------------------------
# Performs similarity search in FAISS using the user's query

def retrieve_docs(faiss_db, query):
    return faiss_db.similarity_search(query)


# -------------------------------------------------------------
# Function: Build Context
# -------------------------------------------------------------
# Combines retrieved document chunks into one text block
# which will be provided to the LLM as context.

def get_context(documents):

    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    return context


# -------------------------------------------------------------
# Function: Generate Answer using LLM
# -------------------------------------------------------------
# This function:
# 1. Creates the prompt
# 2. Injects context + user question
# 3. Sends it to the LLM
# 4. Returns the generated response

def answer_query(documents, model, query):

    context = get_context(documents)

    prompt = ChatPromptTemplate.from_template(
        custom_prompt_template
    )

    chain = prompt | model

    return chain.invoke({
        "question": query,
        "context": context
    })


# -------------------------------------------------------------
# Streamlit UI Section
# -------------------------------------------------------------

# Upload PDF component
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)


# User query input
user_query = st.text_area(
    "Enter your prompt:",
    height=150,
    placeholder="Ask Anything!"
)


# Button to trigger the AI lawyer
ask_question = st.button("Ask AI Lawyer")


# -------------------------------------------------------------
# Main Application Logic
# -------------------------------------------------------------

if ask_question:

    # Ensure both file and question exist
    if uploaded_file and user_query:

        # Step 1: Save uploaded PDF
        upload_pdf(uploaded_file)

        # Step 2: Load PDF content
        documents = load_pdf(pdfs_directory + uploaded_file.name)

        # Step 3: Split document into chunks
        text_chunks = create_chunks(documents)

        # Step 4: Create FAISS vector database
        faiss_db = create_vector_store(
            FAISS_DB_PATH,
            text_chunks,
            ollama_model_name
        )

        # Step 5: Retrieve relevant document chunks
        retrieved_docs = retrieve_docs(
            faiss_db,
            user_query
        )

        # Step 6: Generate answer using LLM
        response = answer_query(
            documents=retrieved_docs,
            model=llm_model,
            query=user_query
        )

        # Display user question
        st.chat_message("user").write(user_query)

        # Display AI response
        st.chat_message("AI Lawyer").write(response)

    else:
        # Error message if user forgot to upload file or ask question
        st.error("Kindly upload a valid PDF file and/or ask a valid Question!")