import os

try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
