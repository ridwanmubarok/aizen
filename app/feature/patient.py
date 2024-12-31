import streamlit as st
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from app.inference.location_utils import get_language, get_season
import torch

def patient_form_page():
    st.info("""
            ### Patient Information
            Please enter the patient data to be diagnosed. This information will be used for the referral letter and diagnosis results related to this system.
            """)
    
    # Retrieve existing data from session state or set default values
    patient_name = st.text_input(
        "Patient Full Name", 
        placeholder="Enter full name", 
        key="patient_name", 
        value=st.session_state.get('patient_name', '')
    )
    patient_age = st.number_input(
        "Age", 
        min_value=0, 
        max_value=120, 
        step=1, 
        format="%d", 
        key="patient_age", 
        value=st.session_state.get('patient_age', 0)
    )
    
    gender_options = ["Male", "Female"]
    default_gender_index = gender_options.index(st.session_state.get('patient_gender', 'Male'))
    gender = st.radio(
        "Gender", 
        options=gender_options, 
        key="patient_gender", 
        index=default_gender_index
    )
    
    contact_info = st.text_input(
        "Contact Information", 
        placeholder="Enter phone or email", 
        key="patient_contact_info", 
        value=st.session_state.get('patient_contact_info', '')
    )
    patient_location = st.text_input(
        "Location", 
        placeholder="Enter patient location", 
        key="patient_location", 
        value=st.session_state.get('patient_location', '')
    )
    st.divider()
    
    if st.button("Save", type="primary", use_container_width=True, key="patient_save_button"):
        with st.spinner("Saving patient data..."):
            if patient_location:
                try:
                    geolocator = Nominatim(user_agent="aizen")
                    location_data = geolocator.geocode(patient_location)
                    if location_data:
                        date = datetime.now()
                        country = location_data.address
                        st.session_state.location = location_data.address
                        st.session_state.date = date
                        language = get_language(country)
                        season = get_season(country, date)
                        st.success(f"Country detected: {country}")
                        st.success(f"Language detected: {language}")
                        st.success(f"Season: {season}")
                        st.session_state.patient_data = {
                            "full_name": patient_name,
                            "age": patient_age,
                            "gender": gender,
                            "contact_info": contact_info,
                            "location": location_data.address,
                            "country": country,
                            "date": date,
                            "language": language,
                            "season": season
                        }
                        st.session_state.completed_tabs["Patient"] = True
                        st.success("Patient data saved successfully")
                        st.session_state.active_tab = "Symptom"
                        st.rerun()
                    else:
                        st.warning("Location not found. Using default weights.")
                except GeocoderTimedOut:
                    st.warning("Location service timed out. Using default weights.")
