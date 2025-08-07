from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
import os

from chatbot.rag_chain import load_chain
from chatbot.document_handler import load_and_split_documents
from chatbot.vectorstore import create_faiss_from_chunks, save_vectorstore


app = Flask(__name__)
app.secret_key = 'a93f1d4b74b74ed992334abef93c46b1'

UPLOAD_FOLDER = "uploads"
VECTORSTORE_PATH = "vectorstore/db_faiss"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


qa_chain, memory = load_chain()

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_query = request.form.get("user_input")
        if user_query:
            response = qa_chain.invoke({"question": user_query})
            answer = response["answer"]

            session["chat_history"].append(("You", user_query))
            session["chat_history"].append(("Bot", answer))
            session.modified = True

    return render_template("index.html", chat_history=session.get("chat_history", []))


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if file and file.filename.endswith((".pdf", ".docx")):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        
        chunks = load_and_split_documents(file_path)
        new_vectorstore = create_faiss_from_chunks(chunks)
        save_vectorstore(new_vectorstore, VECTORSTORE_PATH)

        session["chat_history"].append(("Bot", f"File '{filename}' uploaded and processed. You can now ask questions about it."))
        session.modified = True
    else:
        session["chat_history"].append(("Bot", "Please upload a PDF or DOCX file."))
        session.modified = True

    return redirect(url_for("index"))

@app.route("/clear")
def clear():
    session["chat_history"] = []
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
