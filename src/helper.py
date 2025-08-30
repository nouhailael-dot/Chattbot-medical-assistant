from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

def load_pdf_files(data):
    loader = DirectoryLoader(
        data, 
        glob="*.pdf", 
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents



def filter_to_minimal_docs(documents: List[Document]) -> List[Document]:
    """
    Filters the documents to only include those with minimal content.
    """
    minimal_docs = []
    for doc in documents:
        if len(doc.page_content.strip()) > 0:  # Check if the document has any content
            minimal_docs.append(doc)
    return minimal_docs

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20,
        length_function=len
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

def download_embeddings():
    """
    Downloads and returns the HuggingFace embeddings model.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
       
    )
    return embeddings