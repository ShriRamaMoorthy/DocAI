# 🧠 DocuAI — Document Intelligence Platform
An end-to-end RAG (Retrieval-Augmented Generation) application that lets you 
upload document and chat with them using AI.
## 🚀 Live Demo
[Coming soon — Streamlit Cloud]
## ⚙️ Tech Stack
- **LangChain** — RAG pipeline & LLM orchestration
- **ChromaDB** — Vector store for embeddings
- **OpenAI** — Embeddings + GPT-3.5 for answers
- **Streamlit** — Frontend UI
- **PyMuPDF** — PDF text extraction
## 🏗️ Architecture
User uploads PDF → Text extracted → Chunked → Embedded → 
Stored in ChromaDB → User asks question → Similar chunks 
retrieved → GPT-3.5 generates answer with sources
## 🛠️ Run Locally
# Clone the repo
 git clone https://github.com/YOUR_USERNAME/DocAI.git <br>
cd DocAI
# Install dependencies
pip install -r requirements.txt
# Set your OpenAI API key
export OPENAI_API_KEY=sk-your-key  # Mac/Linux <br>
$env:OPENAI_API_KEY="sk-your-key"  # Windows PowerShell
# Run the app
streamlit run app/main.py
## 📁 Project Structure
DocAI/ <br>
├── app/ <br>
│   ├── main.py                 # Streamlit UI <br>
│   ├── rag_pipeline.py         # Core RAG logic  <br>
│   ├── document_processor.py   # PDF/DOCX parsing <br>
│   └── vector_store.py         # ChromaDB operations <br>
├── data/ <br>
├── vectorstore/ <br>
├── requirements.txt <br>
└── README.md <br>
## 💡 Features
- Upload a single PDF or DOCX file
- Automatic text chunking and embedding
- Semantic similarity search
- GPT-3.5 powered answers with source citations
- Chat history within session
- Adjustable chunk size and retrieval count
## 🔜 Coming Soon...
- Scaling it to handle multiple docs or PDFs
- Suggested Questions
- Highlighting Source in Document
