# AI Lawyer – PDF Question Answering with RAG

AI Lawyer is a **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDFs and ask questions. The system retrieves relevant content from the PDF and generates answers using an LLM.  

This project uses:  

- **Streamlit** – Frontend UI for uploading PDFs and chatting with AI  
- **LangChain** – RAG pipeline orchestration  
- **Ollama embeddings** – Convert PDF text chunks into vectors  
- **FAISS** – Vector database for fast document retrieval  
- **OpenAI GPT** – LLM for answer generation  

---

## Features

- Upload PDF files  
- Automatic text extraction from PDF  
- Split large documents into smaller chunks  
- Generate embeddings for fast similarity search  
- Retrieve relevant chunks using FAISS  
- Generate context-aware answers using OpenAI GPT  
- Interactive chat interface  

---

## Project Structure
├── frontend.py # Streamlit UI for uploading PDFs and asking questions
├── main.py # Full RAG pipeline implementation
├── rag_pipeline.py # Functions for retrieval, context creation, and LLM answering
├── vector_database.py # PDF loading, chunking, embeddings, and FAISS indexing
├── pdfs/ # Directory to store uploaded PDFs
├── vectorstore/ # Directory to store FAISS database
└── README.md # Project documentation




---

## Installation and Setup

1. Install the required Python packages using **pipenv**:

```bash
pipenv install streamlit
pipenv install langchain langchain_community langchain_ollama langchain_core langchain_openai faiss-cpu pdfplumber
```

2. Activate the pipenv shell:
```bash
pipenv shell
```

3. Pull and run the Ollama model in a separate terminal:
```bash
ollama pull deepseek-r1:1.5b
ollama run deepseek-r1:1.5b
```

4. Add a PDF file to the pdfs/ directory.

5. Generate the vector database from your PDF:
```bash
python vector_database.py
```

6. Test the RAG pipeline (optional):
```bash
python rag_pipeline.py
```

7. Run the Streamlit frontend:
```bash
streamlit run main.py
```

## How It Works

**Pipeline Overview:**

- **Upload PDF** – User uploads PDF via Streamlit.
- **Load & Chunk** – PDF is loaded and split into smaller text chunks.
- **Embed** – Chunks are converted into vector embeddings using Ollama.
- **Index** – Chunks are stored in FAISS for similarity search.
- **Retrieve** – User query is converted to vector and used to fetch top relevant chunks.
- **Answer** – Retrieved chunks + user query are sent to OpenAI GPT for answer generation.
- **Display** – Answer is displayed in the chat interface.

---

## Configuration

- **Embedding model:** `deepseek-r1:14b` (Ollama)  
- **Vector database path:** `vectorstore/db_faiss`  
- **LLM model:** `gpt-4o-mini` (OpenAI)  
- **Chunk size:** 1000 characters, overlap: 200  

You can adjust these settings in `main.py` or `vector_database.py` as needed.



## Example Usage

Upload a PDF (for example, a presentation about **Generative AI and Large Language Models (LLMs)**), then ask:

```text
What are the limitations of LLMs?
```


## AI Lawyer Response:

Based on the content of the PDF, the limitations of LLMs include:

- Dependence on verified and up-to-date sources to ensure accuracy.
- Challenges in handling multimodal data efficiently (text + images + video).
- Need for optimization for faster performance and edge AI deployment.
- Limited personalization without additional adaptation for specific users or tasks.
- Current AI assistants still require guidance for complex domains like coding, healthcare, and finance.


Notes

- Currently supports one PDF at a time.
- Ollama embeddings must be installed locally.
- OpenAI API key is required for answer generation.
- Ensure the Ollama model (deepseek-r1:1.5b) is running before using the frontend.


# Author
Krishna Varma Vetukuri
