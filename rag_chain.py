# rag_chain.py

import os
import socket
from typing import Tuple

def load_and_split_pdf(pdf_path):
    # Import locally to avoid heavy imports at module import time
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks


def create_faiss_vectorstore(chunks):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    except Exception as e:
        raise RuntimeError(f"Embedding model failed to load: {e}")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def build_rag_chain(vectorstore):
    from langchain.chains import RetrievalQA
    #from langchain_community.llms import Ollama
    from langchain_ollama import OllamaLLM

    retriever = vectorstore.as_retriever()
    ollama_model = os.environ.get("OLLAMA_MODEL", "mistral")

    #llm = Ollama(model=ollama_model)
    llm = OllamaLLM(model=ollama_model)

    # ✅ force context injection
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # ensures chunks are stuffed into the prompt
        return_source_documents=True
    )
    return qa_chain

def check_ollama_connection(host: str = None, port: int = None, timeout: float = 2.0) -> bool:
    """Return True if a TCP connection to the Ollama host:port succeeds.

    Defaults to reading OLLAMA_HOST/OLLAMA_PORT if host/port are not provided.
    """
    host = host or os.environ.get("OLLAMA_HOST", "localhost")
    port = int(port or os.environ.get("OLLAMA_PORT", "11434"))
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def answer_question_from_pdf(pdf_path, user_query):
    chunks = load_and_split_pdf(pdf_path)
    vectorstore = create_faiss_vectorstore(chunks)
    rag_chain = build_rag_chain(vectorstore)

    result = rag_chain.invoke(user_query)
    answer = result["result"]

    # Optional: show where the answer came from
    sources = [doc.metadata.get("source", "") for doc in result["source_documents"]]

    return {"answer": answer, "sources": sources}




