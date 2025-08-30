from flask import Flask, request, jsonify, render_template
from src.helper import  download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_community.llms import Ollama
import os


app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


embeddings = download_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = Ollama(model="mistral")
prompt= ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt=prompt)  
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "what is Acne?"})
print(response["answer"])

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


