import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint  
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


load_dotenv()
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")



def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_llm():
    return HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",  # Safe and free Hugging Face model
        #huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=256
    )

def load_vector_store():
    return FAISS.load_local("vectorstore/db_faiss", load_embeddings(), allow_dangerous_deserialization=True)


prompt_template = """
You are a medical assistant. Use the context to answer the user's question.
If the answer is not in the context, say "Sorry, I don't have information on that."
Context: {context}
Question: {question}
Answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def create_qa_chain():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )
    return qa_chain


if __name__ == "__main__":
    qa_chain = create_qa_chain()
    while True:
        user_query = input("Write your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        try:
            response = qa_chain.invoke({"query": user_query})
            print("\nAnswer:\n", response["result"])
        except Exception as e:
            print(f" Error: {str(e)}")
