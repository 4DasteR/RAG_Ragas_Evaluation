from pathlib import Path

import streamlit as st

app = Path("app.py")
documents = Path("pages/documents.py")
# logs = Path("pages/logs.py")

def provide_sidebar():
    st.sidebar.title("Menu")
    st.sidebar.page_link(app, label="Evaluate custom query", icon='🔍')
    st.sidebar.page_link(documents, label="Manage documents", icon="📑")
    # st.sidebar.page_link(logs, label="View logs", icon="🖥️")
