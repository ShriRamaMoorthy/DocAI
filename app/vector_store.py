from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

#`
# from langchain_community.embeddings import HuggingFaceEmbeddings
# def get_embeddings():
# return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2") 
# 
# 
# `

def chunk_text(text:str, chunk_size:int=1000, chunk_overlap:int=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n","."," "]
    )
    return splitter.split_text(text)

def create_vector_store(text:str , doc_name:str, persist_dir:str="vectorstore",chunk_size:int=1000):
    chunks = chunk_text(text,chunk_size=chunk_size)

    documents = [
        Document(page_content=chunk,metadata={"source":doc_name,"chunk_id":i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name = doc_name.replace(" ","_").replace(".","_")
    )

    return vectorstore, len(chunks)

def load_vector_store(doc_name:str, persist_dir:str="vectorstore"):
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_metadata=doc_name.replace(" ","_").replace(".","_")
    )

    return vectorstore