from dotenv import load_dotenv
import os
from pinecone import Pinecone 
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings  
from langchain_pinecone import PineconeVectorStore
load_dotenv()

extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)
embeddings = download_embeddings()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)

Index_name = "medical-chatbot"

if not pc.has_index(Index_name):
    pc.create_index(
        name=Index_name,
        dimension=384,  # Dimension should match the embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(Index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=Index_name,
    pinecone_api_key=pinecone_api_key

)

