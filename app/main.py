import streamlit as st
import tempfile
from dotenv import load_dotenv

import sys, os
sys.path.append(os.path.dirname(__file__))

from document_processor import extract_text
from vector_store import create_vector_store
from rag_pipeline import build_qa_chain, ask_question

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


# Page config 
st.set_page_config(
    page_title="DocAI — Document Intelligence",
    page_icon="🧠",
    layout="wide"
)

# Session state init
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# Sidebar
with st.sidebar:
    st.title("DocAI")
    st.markdown("**Document Intelligence Platform**")
    st.divider()

    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
        help="Max 50MB"
    )

    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100,
                           help="Larger = more context per chunk")
    top_k = st.slider("Sources to retrieve", 2, 8, 4,
                      help="How many chunks to fetch per query")

    process_btn = st.button("Process Document", type="primary",
                            disabled=uploaded_file is None)

    if process_btn and uploaded_file:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            suffix = "." + uploaded_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Extract text
            st.info("Extracting text...")
            text = extract_text(tmp_path)

            if len(text.strip()) < 100:
                st.error("Could not extract enough text. Is the PDF scanned?")
            else:
                # Create vector store
                st.info("Creating embeddings...")
                doc_name = uploaded_file.name
                vectorstore, num_chunks = create_vector_store(
                    text, doc_name, chunk_size=chunk_size
                )

                # Build QA chain
                st.info("Building QA chain...")
                st.session_state.qa_chain = build_qa_chain(doc_name)
                st.session_state.doc_name = doc_name
                st.session_state.chat_history = []
                st.session_state.doc_processed = True

                os.unlink(tmp_path)  # cleanup temp file
                st.success(f"Ready! {num_chunks} chunks indexed.")

    if st.session_state.doc_processed:
        st.divider()
        st.success(f"Active: {st.session_state.doc_name}")
        if st.button("Clear & Upload New"):
            st.session_state.qa_chain = None
            st.session_state.doc_name = None
            st.session_state.chat_history = []
            st.session_state.doc_processed = False
            st.rerun()

# Main area
st.title("DocAI — Document Intelligence Platform")

if not st.session_state.doc_processed:
    st.markdown("""
    ### Welcome! Here's how to use DocuMind:
    1. **Upload** a PDF or DOCX in the sidebar
    2. Click **Process Document**
    3. **Ask anything** about your document
    
    #### What you can ask:
    - *"Summarize the key points of this document"*
    - *"What are the payment terms mentioned?"*
    - *"List all risks identified in section 3"*
    - *"What does the document say about termination?"*
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(" **Contracts**\nExtract clauses, terms, obligations")
    with col2:
        st.info(" **Reports**\nSummarize findings, pull key stats")
    with col3:
        st.info(" **Research**\nAsk questions, find methodology")

else:
    # Chat interface
    st.markdown(f"###  Chat with: `{st.session_state.doc_name}`")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander(" View source chunks"):
                    for i, src in enumerate(message["sources"]):
                        st.markdown(f"**Chunk {src['chunk_id']}**")
                        st.text(src["content"][:400] + "...")
                        st.divider()

    # Chat input
    question = st.chat_input("Ask anything about your document...")

    if question:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(st.session_state.qa_chain, question)
                answer = result["answer"]
                sources = result["sources"]

            st.write(answer)

            with st.expander(" View source chunks"):
                for src in sources:
                    st.markdown(f"**Chunk {src['chunk_id']}**")
                    st.text(src["content"][:400] + "...")
                    st.divider()

        # Save to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

