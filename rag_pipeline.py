"""
RAG Pipeline Module - AI Lawyer

This module handles the core Retrieval-Augmented Generation (RAG) workflow:

1. Load the LLM (OpenAI in this case)
2. Retrieve relevant document chunks from FAISS
3. Build context from the retrieved documents
4. Generate answers using the LLM based on context and user query

This file is designed to be imported into frontend/main scripts.
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

# LLM used to generate answers
# from langchain_groq import ChatGroq    # Original Groq model (commented)
from langchain_openai import ChatOpenAI

# FAISS vector database instance
from vector_database import faiss_db

# Prompt templates for formatting LLM input
from langchain_core.prompts import ChatPromptTemplate

# Optional: Load environment variables if not using pipenv
# from dotenv import load_dotenv
# load_dotenv()


# -------------------------------------------------------------
# Step 1: Setup LLM
# -------------------------------------------------------------
# Initialize the language model.
# - This is the model that will generate answers using retrieved context.
# - You can replace "gpt-4o-mini" with other OpenAI models if needed.

llm_model = ChatOpenAI(model="gpt-4o-mini")


# -------------------------------------------------------------
# Step 2: Retrieve Relevant Documents
# -------------------------------------------------------------
# Using the FAISS vector database, retrieve documents most similar
# to the user's query.

def retrieve_docs(query):
    """
    Retrieve the top relevant document chunks for a given query.

    Args:
        query (str): The user's question.

    Returns:
        List[Document]: List of retrieved LangChain Document objects.
    """
    return faiss_db.similarity_search(query)


# -------------------------------------------------------------
# Step 3: Build Context from Retrieved Documents
# -------------------------------------------------------------
# Combine the content of retrieved documents into a single text block
# that will be sent to the LLM as context.

def get_context(documents):
    """
    Concatenate page content of retrieved documents into a single string.

    Args:
        documents (List[Document]): List of LangChain Document objects.

    Returns:
        str: Combined text from all retrieved documents.
    """
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


# -------------------------------------------------------------
# Step 4: Generate Answer Using LLM
# -------------------------------------------------------------
# The answer_query function takes:
# - retrieved documents
# - the LLM model
# - the user's query
# It builds a prompt, sends it to the model, and returns the generated answer.

# Custom prompt template ensures:
# 1. LLM uses only the given context
# 2. No hallucination or guessing
# 3. Answers are concise and relevant

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query):
    """
    Generate an answer for the user's query using the provided LLM model
    and context from retrieved documents.

    Args:
        documents (List[Document]): Retrieved documents
        model (LLM): Language model to generate answers
        query (str): User's question

    Returns:
        str: Generated answer from the LLM
    """
    # Step 1: Build context string
    context = get_context(documents)

    # Step 2: Create prompt using template
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)

    # Step 3: Combine prompt with LLM (pipeline)
    chain = prompt | model

    # Step 4: Invoke the model and return the response
    return chain.invoke({"question": query, "context": context})


# -------------------------------------------------------------
# Example usage (commented)
# -------------------------------------------------------------
# Uncomment the following to test the pipeline manually
# question = "If a government bans peaceful protests, which human rights article is affected?"
# retrieved_docs = retrieve_docs(question)
# print("AI Lawyer:", answer_query(documents=retrieved_docs, model=llm_model, query=question))



