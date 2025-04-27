import streamlit as st

from components.query_builder import ALL_TECHNIQUES, SEMANTIC_UNITS


def render_settings_section(anchor: st, technique: str):
    if technique == "Chain of Thought":
        return {"chain_of_thought": True}
    
    with anchor.expander(f"{technique} Settings"):
        if technique == "Zero-Shot":
            zero_col1, zero_col2 = anchor.columns(2)
            with zero_col1:
                zero_min_val = anchor.number_input("Min", min_value=1, max_value=10)
                
            with zero_col2:
                zero_max_val = anchor.number_input("Max", min_value=zero_min_val+2, max_value=20)
                
            zero_semantic_unit = anchor.radio("Semantic unit", options=SEMANTIC_UNITS, horizontal=True)
            
            return {"zero_shot": (zero_min_val, zero_max_val, zero_semantic_unit)}
            
        elif technique == "Role Prompting":
            role_knowledge = anchor.text_input("You are an expert in...", placeholder="e.g., mathematics and machine learning")
            
            return {"role_prompting": role_knowledge}
            
        elif technique == "Self Consistency":
            n_answers = anchor.number_input("Number of answers to consider", min_value=2)
            
            return {"self_consistency": n_answers}
            
        elif technique == "Directional Stimulus":
            hints = anchor.text_input("Hints...", placeholder="Monkeys are mammals, trees are used to make wood")
            
            return {"directional_stimulus": hints.split(',')}
            
def provide_techniques_menu(anchor: st):
    col_num = 5
    anchor.markdown("### Select prompt optimization techniques")
    cols = anchor.columns(col_num) 
    selected_techniques = []
    selected_techniques_data = {}

    for idx, technique in enumerate(ALL_TECHNIQUES):
        if cols[idx % col_num].checkbox(technique):
            selected_techniques.append(technique)

    for technique in selected_techniques:
        if technique in ALL_TECHNIQUES:
            selected_techniques_data.update(render_settings_section(anchor, technique))
            
    return selected_techniques_data