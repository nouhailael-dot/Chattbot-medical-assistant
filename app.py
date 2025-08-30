from flask import Flask, request, jsonify, render_template
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_community.llms import Ollama, Anthropic
import os


app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


embeddings = download_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore(
    index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = Ollama(model="mistral")

# I see where the issue is

# Basically, your AI is set to ONLY respond to medical questions using the PDF you provided

# It needs a way to determine whether or not to answer medical or regular questions

classification_chain = LLMChain(
    llm=chatModel,
    prompt=ChatPromptTemplate.from_messages(
        [("system", router_prompt), ("user", "{input}")]
    ),
)

general_conversation_chain = LLMChain(
    llm=chatModel,
    prompt=ChatPromptTemplate.from_messages([("system", general_conversation_prompt), ("user", "{input}")])
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "what is Acne?"})
print(response["answer"])


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    classification = classification_chain.invoke({"input": msg})
    print(classification)
    if str(classification["text"]).replace(" ", "").upper() == "MEDICAL":
        print("Medical input detected.")
        response = rag_chain.invoke({"input": msg})
        print("Response : ", response["answer"])
        return str(response["answer"])
    response = general_conversation_chain.invoke({"input": msg})
    print("Response: ", response)
    return str(response["text"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
