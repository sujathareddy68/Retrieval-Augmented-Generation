**Retrieval-Augmented Generation (RAG)**

This project is a RAG-based Question Answering system that lets you upload a PDF, ask questions about its content, and get accurate, context-aware answers.
It combines document retrieval with a language model for more reliable responses.

**Features**

1. Upload PDF documents
2. Extract and index text for retrieval
3. Ask questions in natural language
4. Get answers grounded in the document
5. Flask web interface
6. Supports Ollama with Mistral (local LLM) for private and offline inference

**Installation**

1. Clone the repository:

git clone https://github.com/your-username/your-repo-name.git

cd your-repo-name

2. Create a virtual environment:

python -m venv venv

source venv/bin/activate   # Linux/Mac

venv\Scripts\activate      # Windows

3. Install dependencies:

pip install -r requirements.txt

4. Running the App

python app.py

**Using Ollama + Mistral (Local LLM)**

1. Install Ollama

https://ollama.com/download/windows

2. Pull the Mistral model:

ollama pull mistral

3. Start Ollama in the background:

ollama run mistral
