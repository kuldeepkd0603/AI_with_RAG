from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_PATH="data"

def documnet_loader(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    document=loader.load()
    return document

document=documnet_loader(DATA_PATH)
print(len(document))

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunlks=text_splitter.split_documents(extracted_data)
    return text_chunlks

text_chunks=create_chunks(document)

print(len(text_chunks))

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

DB_FAIS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,get_embedding_model())
db.save_local(DB_FAIS_PATH)