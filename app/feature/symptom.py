import streamlit as st
from PIL import Image
from torchvision import transforms

def symptom_form_page():
    # Collect symptom information
    st.info("""
            ### Symptom Information
            Enter the symptoms of the disease you are experiencing along with a relevant image. Please ensure the image uploaded is of high resolution.
            """)

    # Symptom name
    symptom = st.text_area("The symptoms you are experiencing", placeholder="Ex: Itchy, Red, Swollen, etc.", key="symptom")
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], 
                                     label_visibility="collapsed")
    
    patient_data = st.session_state.get('patient_data', None)
    
    if patient_data is None:
        st.warning("Please complete the patient information first.")
        st.stop()
    
    if uploaded_file:
        # Display original image
        original_image = Image.open(uploaded_file).convert('RGB')
        original_image.thumbnail((320, 250))
        
        # Process the image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Create a copy for processing visualization
        processed_image = original_image.copy()
        processed_image = transforms.Resize(256)(processed_image)
        processed_image = transforms.CenterCrop(224)(processed_image)
        processed_image = transforms.ColorJitter(brightness=0.1, contrast=0.1)(processed_image)
        
        # Display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image")
        with col2:
            st.image(processed_image, caption="Processed Image")
        
        # Save to session state
        st.session_state.image = original_image
        
        st.session_state.symptom_data = {
            "symptom": symptom,
            "original_image": original_image,
            "processed_image": processed_image
        }
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            if st.button("Back", use_container_width=True, key="symptom_back_button"):
                st.session_state.completed_tabs["Patient"] = False
                st.rerun()
        
        with col_right:
            if st.button("Save", type="primary", use_container_width=True, key="symptom_save_button"):
                with st.spinner('Saving symptom data...'):
                    st.session_state.completed_tabs["Symptom"] = True
                    st.success('Symptom data saved successfully!')
                    st.session_state.active_tab = "Diagnosis"
                    st.rerun()

    return uploaded_file


