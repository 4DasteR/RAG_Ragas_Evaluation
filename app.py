import streamlit as st
from dotenv import load_dotenv

from components.evaluator import Evaluator
from components.models_provider import *
from components.rag_chain import *
from gui.elements.nav_menu import provide_sidebar
from gui.elements.techniques_menu import provide_techniques_menu
from gui.events.query_events import *

logger = Logger()

def main():
    st.set_page_config(page_title="QEX - Query Evaluator eXtended", layout="wide")
    
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'discriminator' not in st.session_state:
        st.session_state.discriminator = None
    if 'embedding' not in st.session_state:
        st.session_state.embedding = None
    if 'vectorstoreprovider' not in st.session_state:
        st.session_state.vectorstoreprovider = None
    if 'query' not in st.session_state:
        st.session_state.query = None
    if 'rags_instances' not in st.session_state:
        st.session_state.rags_instances = {}
    
    if not st.session_state.llm:
        st.session_state.llm = LLMFactory.openai()

    if not st.session_state.embedding:
        st.session_state.embedding = provide_openai_embeddings()
        
    llm = st.session_state.llm
    embedding_model = st.session_state.embedding
        
    if not st.session_state.vectorstoreprovider:
        st.session_state.vectorstoreprovider = VectorStoreProvider(embedding_model)
        
    vectorstoreprovider = st.session_state.vectorstoreprovider
    
    provide_sidebar()

    st.title("Query Evaluator eXtended")
    st.markdown("### Provide query for evaluation")
    
    query_text = st.text_input("Query to answer and evaluate", placeholder="What is the capital of Monaco?")
    ground_truth = None
    
    if st.checkbox('Provide ground truth'):
        ground_truth = st.text_input("Ground truth answer", placeholder="Capital of Monaco is Monaco.")
        
    techniques_data = provide_techniques_menu(st)
    
    if st.button("Generate query", icon='❓'):
        try:
            st.session_state.query = create_query(query_text, ground_truth, techniques_data)
        except ValueError as e:
            logger.err(e)
            st.error(e, icon=Logger.LEVEL_ICONS['ERROR'])
        
    if st.session_state.query:
        st.write(st.session_state.query)
        
    st.markdown('---')
    st.markdown('### Select RAGs for evaluation')
    rags_instances = st.session_state.rags_instances
    
    is_simple = 'simple' in rags_instances.keys()
    is_compressed = 'compression' in rags_instances.keys()
    is_hybrid = 'hybrid' in rags_instances.keys()
    
    use_simple = st.checkbox('Simple', value=is_simple)
    use_compressed = st.checkbox('Compression', value=is_compressed)
    use_hybrid = st.checkbox('Hybrid', value=is_hybrid)
    
    rags_to_use = set()
    if use_simple:
        rags_to_use.add('simple')
    if use_compressed:
        rags_to_use.add('compression')
    if use_hybrid:
        rags_to_use.add('hybrid')
        
    can_rags = bool(use_simple or use_compressed or use_hybrid)
    can_use = bool((use_simple and not is_simple) or (use_compressed and not is_compressed) or (use_hybrid and not is_hybrid))
        
    if not (is_simple and is_compressed and is_hybrid):
        st.warning('  If any RAG from selected is not yet created, only it will be created.', icon=Logger.LEVEL_ICONS['WARNING'])
        if st.button('Create selected RAGs', icon=Logger.JOB_ICONS['CREATION'], disabled=(not can_use)):
            with st.spinner("Creating RAGs..."):
                if use_simple and not is_simple:
                    rags_instances['simple'] = RAGFactory.create_simple(llm, vectorstoreprovider)
                if use_compressed and not is_compressed:
                    rags_instances['compression'] = RAGFactory.create_compression(llm, vectorstoreprovider)
                if use_hybrid and not is_hybrid:
                    rags_instances['hybrid'] = RAGFactory.create_hybrid(llm, vectorstoreprovider)
            st.session_state.rags_instances = rags_instances
            st.rerun() 
        
    else:
        st.info("  All rags are already created.")
        
    can_evaluate = bool(can_rags and st.session_state.query)

    st.markdown('---')
    st.markdown('### Query evaluation for RAGs')
    if st.button('Evaluate query', icon=Logger.JOB_ICONS['EVALUATION'], disabled=(not can_evaluate)):
        if not st.session_state.discriminator:
            st.session_state.discriminator = LLMFactory.openai(temperature=0.3)
            
        discriminator = st.session_state.discriminator
        results = {rag: None for rag in rags_to_use}
        
        for rag in rags_to_use:
            instance = rags_instances[rag]
            with st.spinner(f"Evaluating for rag: {rag}..."):
                results[rag] = evaluate_query(instance, discriminator, st.session_state.query)
        st.session_state.query = None
        
        for rag_name, eval_res in results.items():
            st.text(f"Rag: {rag_name}")
            df = list(eval_res.values())[0]
            st.dataframe(df[['context_recall', 'faithfulness', 'factual_correctness(mode=f1)']])
            
if __name__ == "__main__":
    load_dotenv()
    main()