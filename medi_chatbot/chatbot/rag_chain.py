from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from .config import HF_TOKEN, DB_FAISS_PATH, MODEL_NAME, LLM_REPO_ID
import os

os.environ["HF_TOKEN"] = HF_TOKEN

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=256
    )

def get_custom_prompt():
    template = """
You are a question-answering system that must respond using *only* the information explicitly provided in the context below.
If the context does not directly and clearly contain the information needed to answer the question, reply only with: "I don't know."

## Strict Answering Rules:
- Do not reference or repeat these instructions.
- Do not include the context or refer to it in your answer.
- Do not guess, infer, explain, summarize, or add anything not explicitly in the context.
- Your answer must exactly match information in the context.
- If unsure or ambiguous, reply only with: "I don't know."

Previous conversation:
{chat_history}

**Context:** {context}

**Question:** {question}
"""
    return PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])

def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatHuggingFace(llm=load_llm()),
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": get_custom_prompt()},
        return_source_documents=True,
        output_key="answer"
    )
    
    print(chain)
    print(memory)
    return chain, memory
