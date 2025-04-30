from typing import Optional

import streamlit as st

from components.vector_store import VectorStoreProvider
from gui.elements.nav_menu import provide_sidebar

provide_sidebar()

vectorstoreprovider: Optional[VectorStoreProvider] = st.session_state.vectorstoreprovider


if (files := st.file_uploader("Upload new documents", type=["pdf", "txt", "html", "md", "docx", "pptx"], accept_multiple_files=True)):
    for file in files:
        with open(f"{vectorstoreprovider.documents_path}/{file.name}", 'wb') as f:
            f.write(file.getbuffer())
    st.success("Documents saved")
    
st.markdown('# Available documents')
for doc in vectorstoreprovider.documents_path.glob('*.*'):
    name, remove = st.columns(2)
    name.markdown(doc.name)
    remove.button("Remove", key=f"remove-{doc.name}")