from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_faiss_from_chunks(chunks):
    return FAISS.from_documents(chunks, embedding_model)

def save_vectorstore(faiss_db, path):
    faiss_db.save_local(path)

def load_vectorstore(path):
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
