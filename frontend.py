"""
Frontend for the AI Lawyer Application

This Streamlit app provides a simple interface where users can:
1. Upload a PDF document
2. Ask questions related to the uploaded document
3. Receive answers generated using a RAG (Retrieval Augmented Generation) pipeline

The RAG pipeline:
User Question -> Retrieve relevant document chunks -> Send to LLM -> Generate answer
"""

# Import functions from the RAG pipeline
# - retrieve_docs : searches the vector database for relevant document chunks
# - answer_query  : generates an answer using the LLM and retrieved context
# - llm_model     : the language model used to generate responses
from rag_pipeline import answer_query, retrieve_docs, llm_model

# Import Streamlit for building the web UI
import streamlit as st


# ------------------------------------------------------------
# Step 1: Upload PDF functionality
# ------------------------------------------------------------
# Allows the user to upload a single PDF document.
# This file will later be processed and indexed into the vector database.

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)


# ------------------------------------------------------------
# Step 2: User Input (Chat Interface)
# ------------------------------------------------------------
# Text area where the user can type their legal question
# related to the uploaded document.

user_query = st.text_area(
    "Enter your prompt:",
    height=150,
    placeholder="Ask Anything!"
)


# Button that triggers the AI response
ask_question = st.button("Ask AI Lawyer")


# ------------------------------------------------------------
# Step 3: Process Question using RAG Pipeline
# ------------------------------------------------------------
# When the user clicks the button, the following steps happen:
# 1. Check if a PDF is uploaded
# 2. Retrieve relevant document chunks from the vector DB
# 3. Send the chunks + user question to the LLM
# 4. Display the generated answer

if ask_question:

    # Ensure a file has been uploaded
    if uploaded_file:

        # Display the user's message in the chat interface
        st.chat_message("user").write(user_query)

        # ----------------------------------------------------
        # RAG Step 1: Retrieve relevant documents
        # ----------------------------------------------------
        # The user's query is used to search the vector database
        # and fetch the most relevant document chunks.
        retrieved_docs = retrieve_docs(user_query)

        # ----------------------------------------------------
        # RAG Step 2: Generate answer using LLM
        # ----------------------------------------------------
        # The retrieved document chunks are provided as context
        # to the language model along with the user's question.
        response = answer_query(
            documents=retrieved_docs,
            model=llm_model,
            query=user_query
        )

        # ----------------------------------------------------
        # Display AI response
        # ----------------------------------------------------
        st.chat_message("AI Lawyer").write(response)

    else:
        # Show error if user asks a question without uploading a file
        st.error("Kindly upload a valid PDF file first!")