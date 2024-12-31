import streamlit as st
from feature.patient import patient_form_page
from feature.symptom import symptom_form_page
from feature.diagnosis import diagnosis_result_page
from feature.reference_letter import reference_letter_page
from feature.result_detail import result_detail_page

def main():
    st.set_page_config(
        page_title="Aizen",
        page_icon=":hospital:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("AIZEN")
    st.markdown("""
            AI-powered disease diagnosis and treatment recommendation
            """)
    
    # Initialize active tab if not exists
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'Patient'
    
    # Initialize session state for completion
    if 'completed_tabs' not in st.session_state:
        st.session_state.completed_tabs = {
            "Patient": False, 
            "Symptom": False,
            "Diagnosis": False,
            "Result Detail": False,
            "Reference Letter": False,
        }

    # Create navigation buttons
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        if st.button("Patient📋", use_container_width=True):
            st.session_state.active_tab = "Patient"
            st.session_state.completed_tabs["Patient"] = False
    with col2:
        if st.button("Symptom 🤒", use_container_width=True):
            st.session_state.active_tab = "Symptom"
            st.session_state.completed_tabs["Symptom"] = False
    with col3:
        if st.button("Diagnosis 🩺", use_container_width=True):
            st.session_state.active_tab = "Diagnosis"
            st.session_state.completed_tabs["Diagnosis"] = False
    with col4:
        if st.button("Data Detail 📝", use_container_width=True):
            st.session_state.active_tab = "Result Detail"
            st.session_state.completed_tabs["Result Detail"] = False
    with col5:
        if st.button("Download 📥", use_container_width=True):
            st.session_state.active_tab = "Reference Letter"
            st.session_state.completed_tabs["Reference Letter"] = False


    # Display content based on active tab
    if st.session_state.active_tab == "Patient":
        patient_form_page()
        if st.session_state.completed_tabs["Patient"]:
            st.session_state.active_tab = "Symptom"
            
    elif st.session_state.active_tab == "Symptom":
        symptom_form_page()
        if st.session_state.completed_tabs["Symptom"]:
            st.session_state.active_tab = "Diagnosis"
            
    elif st.session_state.active_tab == "Diagnosis":
        diagnosis_result_page()
        if st.session_state.completed_tabs["Diagnosis"]:
            st.session_state.active_tab = "Result Detail"
            
    elif st.session_state.active_tab == "Result Detail":
        result_detail_page()
        if st.session_state.completed_tabs["Result Detail"]:
            st.session_state.active_tab = "Reference Letter"
            
    elif st.session_state.active_tab == "Reference Letter":
        reference_letter_page()
        if st.session_state.completed_tabs["Reference Letter"]:
            st.session_state.active_tab = "Patient"
            

if __name__ == "__main__":
    main()
