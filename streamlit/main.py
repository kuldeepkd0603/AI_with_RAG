import os

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain



from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer",return_messages=True)






os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

def load_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
        task="text-generation",
        #   huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=256
    )

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
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




def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question","chat_history"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
# qa_chain=RetrievalQA.from_chain_type(
#     llm=ChatHuggingFace(llm=load_llm()),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k':3}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatHuggingFace(llm=load_llm()),
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    return_source_documents=True,
    output_key="answer" 
)
# Now invoke with a single query
# user_query=input("Write Query Here: ")
# #user_query="Explain the types of cancer?"
# response=qa_chain.invoke({'query': user_query})
# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])

# llm = load_llm()
# model=ChatHuggingFace(llm=llm)
# res = model.invoke("What is the capital of France?")
# print("LLM Response:", res)


# st.title("RAG Based Medical Chatbot")
# user_query = st.text_input("Enter your query here:", key="user_query")
# if st.button("Ask"):
#     if user_query:
#         print("User Query inside streamlit: ", user_query)
#         response=qa_chain.invoke({'question': user_query})
#         print("response: ", response)
#         print("RESULT: ", response["answer"])
#         st.markdown("### Your answere: ")
#         st.write(response["result"])
#     else:
#         st.error("Please enter a query.")

# st.title("RAG Based Medical Chatbot")

# user_query = st.text_input("Enter your query here:", key="user_query")

# if st.button("Ask"):
#     if user_query:
#         response = qa_chain.invoke({'question': user_query})
#         st.markdown("### Bot Answer:")
#         st.write(response["answer"])
#     else:
#         st.error("Please enter a query.")

# st.title("RAG Based Medical Chatbot")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# user_query = st.text_input("Enter your query here:", key="user_query")

# if st.button("Ask"):
#     if user_query:
#         # Invoke the chain
#         response = qa_chain.invoke({'question': user_query})
#         answer = response["answer"]

#         # Save user input and response to memory manually
#         memory.save_context({"question": user_query}, {"answer": answer})

#         # Show answer
#         st.markdown("### Bot Answer:")
#         st.write(answer)

#         # Optionally show source documents
#         with st.expander("Sources"):
#             for doc in response["source_documents"]:
#                 st.write(doc.page_content)
#     else:
#         st.error("Please enter a query.")




# st.title("RAG-Based Medical Chatbot with Memory")

# # Initialize chat history in session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Input from user
# user_query = st.text_input("Enter your query here:", key="user_query")

# if st.button("Ask"):
#     if user_query:
#         # Add user's message to session chat history
#         st.session_state.chat_history.append({"role": "user", "content": user_query})

#         # Query the RAG chain
#         response = qa_chain.invoke({'question': user_query})

#         # Add bot response to chat history
#         st.session_state.chat_history.append({"role": "bot", "content": response["answer"]})

#         # Display bot response
#         st.markdown("### Bot Answer:")
#         st.write(response["answer"])

#         # Optional: Show chat history
#         with st.expander("Chat History", expanded=True):
#             for msg in st.session_state.chat_history:
#                 role = "You" if msg["role"] == "user" else "Bot"
#                 st.markdown(f"**{role}:** {msg['content']}")
#     else:
#         st.error("Please enter a query.")

# st.title("🧠 RAG-Based Medical Chatbot with Memory")

# # Initialize chat history in session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# user_query = st.text_input("Enter your query here:", key="user_query")

# if st.button("Ask"):
#     if user_query:
#         # Run query through RAG chain
#         response = qa_chain.invoke({"question": user_query})
#         answer = response["answer"]

#         # Save interaction in Streamlit's session
#         st.session_state.chat_history.append(("You", user_query))
#         st.session_state.chat_history.append(("Bot", answer))

#         # Save to memory (for LangChain's internal memory)
#         memory.save_context({"question": user_query}, {"answer": answer})

# # 💬 Display chat history (chat-style)
# if st.session_state.chat_history:
#     st.markdown("### 🗨️ Chat History")
#     for sender, message in st.session_state.chat_history:
#         if sender == "You":
#             st.markdown(f"**👤 You:** {message}")
#         else:
#             st.markdown(f"**🤖 Bot:** {message}")


# import streamlit as st

# st.set_page_config(page_title="Medical Chatbot", page_icon="🧠", layout="centered")
# st.title("🧠 RAG-Based Medical Chatbot with Memory")

# # Initialize chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Text input at bottom (chat-style)
# with st.form(key="chat_form", clear_on_submit=True):
#     user_query = st.text_input("💬 Ask your medical question:", key="user_input", label_visibility="collapsed")
#     submit = st.form_submit_button("Send")

# # Handle message on submit
# if submit and user_query:
#     # Run RAG chain
#     response = qa_chain.invoke({"question": user_query})
#     answer = response["answer"]

#     # Append both messages to chat history
#     st.session_state.chat_history.append(("You", user_query))
#     st.session_state.chat_history.append(("Bot", answer))

#     # Save to LangChain memory
#     memory.save_context({"question": user_query}, {"answer": answer})

# # Display messages in reverse for latest at bottom (chat-style)
# for sender, message in st.session_state.chat_history:
#     with st.chat_message("user" if sender == "You" else "assistant"):
#         st.markdown(message)


import streamlit as st

st.set_page_config(page_title="Medical Chatbot", page_icon="", layout="centered")
st.title("RAG-Based Medical Chatbot with Memory")

# Initialize chat history
if st.button("Clear Chat"):
    st.session_state.chat_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat display area (top)
chat_container = st.container()

# Input area (bottom)
input_container = st.container()

# Show full chat history at the top
with chat_container:
    for sender, message in st.session_state.chat_history:
        with st.chat_message("user" if sender == "You" else "assistant"):
            st.markdown(message)

# Input field stays at the bottom
with input_container:
    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("Type your message...", key="user_input", label_visibility="collapsed")
        submitted = st.form_submit_button("Send")
        print(user_query)

    if submitted and user_query:
        # Get response from the RAG chain
        response = qa_chain.invoke({"question": user_query})
        answer = response["answer"]
        print(response)
        print(answer)

        # Update chat history
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("Bot", answer))

        # Save to memory
        #memory.save_context({"question": user_query}, {"answer": answer})
        print(memory)
