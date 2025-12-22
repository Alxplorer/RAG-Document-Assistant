# RAG-Document-Assistant
# AI English Tutor (RAG Architecture)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://www.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-Container-blue.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red.svg)](https://streamlit.io/)

> **An intelligent B2 English Tutor that uses Retrieval-Augmented Generation (RAG) to provide context-aware grammar corrections and explanations without hallucinations.**

---

## 🚀 Live Demo
Try the application running in the cloud here:
👉 **[CLICK HERE TO CHAT WITH THE BOT](https://rag-document-assistant-qrakl7xvikpye7hmvl3jbq.streamlit.app/)**

---

##About The Project...

This is not a generic chatbot. It is a specialized **RAG Application** designed to help students improve their English to a B2 level.

Unlike standard ChatGPT, this bot is "grounded" on specific grammar guides and vocabulary lists. It retrieves relevant information from a vector database before answering, ensuring accuracy and pedagogical value.

### Key Features
* **RAG Architecture:** Ingests PDF/TXT study guides, creates embeddings, and retrieves precise context.
* **Hybrid Reasoning:** Prioritizes document context but falls back to general LLM knowledge for casual conversation (Hybrid RAG).
* **Dockerized:** Fully containerized application for consistent deployment across environments.
* **Hallucination Control:** System prompts engineered to strictly follow educational guidelines.
* **Memory:** Maintains conversation history for a natural tutoring experience.

---

## Tech Stack

* **Core:** Python 3.13.2
* **Orchestration:** LangChain
* **LLM:** OpenAI API (GPT-3.5-Turbo / GPT-4)
* **Vector Database:** ChromaDB (Persisted embeddings)
* **Frontend:** Streamlit
* **Containerization:** Docker
* **Data Processing:** PyPDFLoader, RecursiveCharacterTextSplitter

---

## Architecture

The system follows a standard RAG pipeline:

1.  **Ingestion:** Study guides are loaded and split into chunks.
2.  **Embedding:** Chunks are converted into vectors using OpenAI Embeddings.
3.  **Storage:** Vectors are stored in ChromaDB.
4.  **Retrieval:** User queries are converted to vectors to find the most similar chunks (Semantic Search).
5.  **Generation:** The LLM receives the user question + the retrieved context to generate an accurate answer.


##How to Run Locally

### Option 1

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[Alxplorer]/[RAG-Document-Assistant].git
   cd https://github.com/Alxplorer
