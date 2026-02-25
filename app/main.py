import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

import sys, os
sys.path.append(os.path.dirname(__file__))

from document_processor import extract_text
from vector_store import create_vector_store
from rag_pipeline import build_qa_chain, ask_question

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

import os

# Works both locally AND on Streamlit Cloud
def get_api_key():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    return key

os.environ["OPENAI_API_KEY"] = get_api_key()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DocAI — Document Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ── Session state init ────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("DocAI")
    st.markdown("**Document Intelligence Platform**")
    st.divider()

    st.subheader(" Upload Document")
    uploaded_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
        help="Max 50MB"
    )

    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100,
                           help="Larger = more context per chunk")
    top_k = st.slider("Sources to retrieve", 2, 8, 4,
                      help="How many chunks to fetch per query")

    process_btn = st.button(" Process Document", type="primary",
                            disabled=uploaded_file is None)

    if process_btn and uploaded_file:
        with st.spinner("Processing document..."):
            try:
                # Save uploaded file temporarily
                suffix = "." + uploaded_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Extract text
                st.info(" Extracting text...")
                text = extract_text(tmp_path)

                if len(text.strip()) < 100:
                    st.error("Could not extract enough text. Is the PDF scanned?")
                else:
                    # Create vector store
                    st.info(" Creating embeddings...")
                    doc_name = uploaded_file.name
                    vectorstore, num_chunks = create_vector_store(
                        text, doc_name, chunk_size=chunk_size
                    )

                    # Build QA chain
                    st.info(" Building QA chain...")
                    st.session_state.qa_chain = build_qa_chain(doc_name)
                    st.session_state.doc_name = doc_name
                    st.session_state.chat_history = []
                    st.session_state.doc_processed = True

                    os.unlink(tmp_path)  # cleanup temp file
                    st.success(f" Ready! {num_chunks} chunks indexed.")

            except Exception as e:
                st.error(f" Failed to process document: {str(e)[:300]}")

    if st.session_state.doc_processed:
        st.divider()
        st.success(f" Active: {st.session_state.doc_name}")
        if st.button(" Clear & Upload New"):
            st.session_state.qa_chain = None
            st.session_state.doc_name = None
            st.session_state.chat_history = []
            st.session_state.doc_processed = False
            st.rerun()

# ── Main area ─────────────────────────────────────────────────
st.title(" DocAI — Document Intelligence Platform")

if not st.session_state.doc_processed:
    st.markdown("""
    ### Welcome! Here's how to use DocAI:
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
    # ── Chat interface ────────────────────────────────────────
    st.markdown(f"### Chat with: `{st.session_state.doc_name}`")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View source chunks"):
                    for src in message["sources"]:
                        st.markdown(f"**Chunk {src['chunk_id']}**")
                        st.text(src["content"][:400] + "...")
                        st.divider()

    # Chat input
    question = st.chat_input("Ask anything about your document...")

    if question:
        # Add user message to history and display it
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.write(question)

        # Get and display assistant answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = ask_question(st.session_state.qa_chain, question)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.write(answer)

                    if sources:
                        with st.expander("📎 View source chunks"):
                            for src in sources:
                                st.markdown(f"**Chunk {src['chunk_id']}**")
                                st.text(src["content"][:400] + "...")
                                st.divider()

                    # Save success to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error = str(e)

                    if "429" in error or "quota" in error.lower() or "rate" in error.lower():
                        error_msg = """
                        **Whoops! The AI ran out of juice.**

                        Our OpenAI quota is cooked. The robots are on a coffee break 

                        **What you can do:**
                        - Come back later (quota resets monthly)
                        - Or tell the developer to top up their OpenAI credits 

                        *Error: Insufficient quota (HTTP 429)*
                        """
                    elif "api_key" in error.lower():
                        error_msg = """
                        **The AI forgot its password.**

                        API key is missing or invalid. Someone forgot to feed the robot its key 

                        *Error: Invalid API Key*
                        """
                    else:
                        error_msg = f"""
                        **Something went sideways.**

                        The AI tripped and fell. Here's what it said:

                        `{error[:200]}`
                        """

                    st.error(error_msg)

                    # Save error to history so chat doesn't break
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "An error occurred. Please try again.",
                        "sources": []
                    })