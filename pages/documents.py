import os
from typing import Optional

import streamlit as st

from components.models_provider import provide_openai_embeddings
from components.vector_store import VectorStoreProvider
from gui.elements.nav_menu import provide_sidebar

provide_sidebar()

if not st.session_state.embedding:
    st.session_state.embedding = provide_openai_embeddings()
    embedding_model = st.session_state.embedding
        
if not st.session_state.vectorstoreprovider:
    st.session_state.vectorstoreprovider = VectorStoreProvider(embedding_model)

vectorstoreprovider: Optional[VectorStoreProvider] = st.session_state.vectorstoreprovider

if (files := st.file_uploader("Upload new documents", type=["pdf", "txt", "html", "md", "docx", "pptx"], accept_multiple_files=True)):
    for file in files:
        with open(f"{vectorstoreprovider.documents_path}/{file.name}", 'wb') as f:
            f.write(file.getbuffer())
    st.success("Documents saved")

st.markdown('# Available documents')
for doc in vectorstoreprovider.documents_files:
    name, remove = st.columns([4, 1])
    name.markdown(doc.name)
    if remove.button("Remove", key=f"remove-{doc.name}"):
        os.remove(doc) 
        st.rerun()