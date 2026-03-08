"""
Vector Database Module - AI Lawyer

This module handles:
1. Uploading and loading PDFs
2. Splitting PDFs into smaller text chunks
3. Generating embeddings for chunks using Ollama
4. Storing embeddings in a FAISS vector database for fast retrieval

This file is designed to be imported into the RAG pipeline.
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
# PDF loader to extract text from PDF files
from langchain_community.document_loaders import PDFPlumberLoader

# Used to split large documents into smaller, searchable chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding model running locally via Ollama
from langchain_ollama import OllamaEmbeddings

# Vector database for storing embeddings
from langchain_community.vectorstores import FAISS


# -------------------------------------------------------------
# Step 1: Upload & Load Raw PDF(s)
# -------------------------------------------------------------
# All PDFs will be stored in a dedicated folder
pdfs_directory = 'pdfs/'

def upload_pdf(file):
    """
    Save the uploaded PDF file to the local 'pdfs' directory.

    Args:
        file (UploadedFile): File object uploaded via Streamlit or similar.
    """
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    """
    Load a PDF file and convert it into LangChain Document objects.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        List[Document]: List of documents, one per page.
    """
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


# Example: Load a specific PDF file
file_path = 'Generative_AI_Presentation.pdf'
documents = load_pdf(file_path)
# print("PDF pages:", len(documents))


# -------------------------------------------------------------
# Step 2: Create Text Chunks
# -------------------------------------------------------------
# Splitting large documents into smaller chunks improves:
# 1. Embedding quality
# 2. Retrieval accuracy
# 3. LLM response relevance

def create_chunks(documents):
    """
    Split documents into smaller overlapping text chunks.

    Args:
        documents (List[Document]): List of loaded documents.

    Returns:
        List[Document]: List of text chunks as Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Max characters per chunk
        chunk_overlap=200,     # Overlap to preserve context between chunks
        add_start_index=True   # Keep track of original position in document
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


# Generate text chunks from the loaded documents
text_chunks = create_chunks(documents)
# print("Chunks count:", len(text_chunks))


# -------------------------------------------------------------
# Step 3: Setup Embeddings Model
# -------------------------------------------------------------
# Ollama embeddings will convert text chunks into vector representations.
# These embeddings are later used for similarity search in FAISS.

ollama_model_name = "deepseek-r1:1.5b" #"deepseek-r1:14b"

def get_embedding_model(ollama_model_name):
    """
    Initialize and return Ollama embeddings model.

    Args:
        ollama_model_name (str): Name of the Ollama model.

    Returns:
        OllamaEmbeddings: Embedding model instance
    """
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings


# -------------------------------------------------------------
# Step 4: Create FAISS Vector Database
# -------------------------------------------------------------
# 1. Convert text chunks into embeddings
# 2. Store them in FAISS for fast retrieval during query time
# 3. Save the database locally

FAISS_DB_PATH = "vectorstore/db_faiss"

faiss_db = FAISS.from_documents(
    text_chunks,
    get_embedding_model(ollama_model_name)
)

# Save the FAISS vector database to disk
faiss_db.save_local(FAISS_DB_PATH)