import os
import sys
sys.path.append(os.path.dirname(__file__))

from config import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vector_store import load_vector_store


PROMPT_TEMPLATE = """
You are an expert document analyst. Use ONLY the context provided below to answer 
the question. If the answer is not in the context, say "I couldn't find that in 
the document." Do not make up information.

Context:
{context}

Question: {question}

Answer in a clear, structured way. If relevant, mention which part of the document 
supports your answer.
"""


def get_llm():
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1000,
        api_key=OPENAI_API_KEY
    )


def format_docs(docs):
    """Combine retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(doc_name: str):
    """Build the RAG chain using modern LCEL syntax."""
    vectorstore = load_vector_store(doc_name)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    llm = get_llm()

    # Modern LCEL chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, rag_chain


def ask_question(chain_tuple, question: str) -> dict:
    """Run a question through the RAG chain."""
    retriever, rag_chain = chain_tuple

    # Get answer
    answer = rag_chain.invoke(question)

    # Get source chunks separately
    source_docs = retriever.invoke(question)
    sources = [
        {
            "content": doc.page_content,
            "chunk_id": doc.metadata.get("chunk_id", "?"),
            "source": doc.metadata.get("source", "unknown")
        }
        for doc in source_docs
    ]

    return {
        "answer": answer,
        "sources": sources
    }