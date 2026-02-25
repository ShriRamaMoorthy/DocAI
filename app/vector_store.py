import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from config import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document



def get_embeddings():
    return SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def create_vector_store(text: str, doc_name: str, persist_dir: str = "vectorstore", chunk_size: int = 1000):
    chunks = chunk_text(text, chunk_size=chunk_size)

    documents = [
        Document(page_content=chunk, metadata={"source": doc_name, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    collection_name = doc_name.replace(" ", "_").replace(".", "_")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )

    return vectorstore, len(chunks)


def load_vector_store(doc_name: str, persist_dir: str = "vectorstore"):
    embeddings = get_embeddings()

    collection_name = doc_name.replace(" ", "_").replace(".", "_")

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name        # ← was collection_metadata before, that was the bug
    )

    return vectorstore