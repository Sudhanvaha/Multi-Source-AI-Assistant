# Multi Source AI Assistant

An agentic AI assistant built using LangGraph that integrates multiple knowledge sources into a single conversational interface.

The assistant can answer questions using uploaded PDF documents, YouTube video transcripts, web search, stock market information, and utility tools while maintaining long-term conversational memory across sessions.

## Features

* Multi-thread conversation support
* Long-term memory with persistent storage
* PDF-based Retrieval Augmented Generation (RAG)
* YouTube transcript question answering
* Hybrid retrieval using FAISS and BM25
* Cross-encoder reranking for improved relevance
* Web search for recent information
* Stock price lookup tool
* Calculator tool
* Conversation summarization for efficient memory usage
* Persistent chat history using PostgreSQL
* Interactive Streamlit frontend

## Architecture

```text
User
 ↓
Streamlit Frontend
 ↓
LangGraph Workflow
 ├── Memory System
 ├── PDF RAG
 ├── YouTube RAG
 ├── Web Search
 ├── Stock Tool
 └── Calculator Tool
 ↓
PostgreSQL + Vector Storage
```

## Tech Stack

* Python
* LangGraph
* LangChain
* Groq LLM
* Streamlit
* PostgreSQL
* FAISS
* BM25
* HuggingFace Embeddings
* Cross Encoder Reranker

## Project Structure

```text
backend/
│
├── main.py
├── graph.py
├── rag.py
├── tools.py
├── database.py
└── config.py

storage/
├── pdf/
└── youtube/

frontend/
└── app.py
```

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd Multi-Source-AI-Assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

Configure PostgreSQL and update database connection strings in the project configuration.

## Running the Application

Start the Streamlit application:

```bash
streamlit run frontend/app.py
```

## Supported Sources

* PDF Documents
* YouTube Videos
* Web Search
* Stock Market Data
* Mathematical Calculations

## Future Improvements

* FastAPI backend
* React frontend
* Multi-document retrieval
* Additional tools and integrations
* Docker deployment
* Evaluation pipeline

## License

This project is intended for educational and learning purposes.
