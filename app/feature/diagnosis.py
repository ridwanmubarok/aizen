import streamlit as st
from app.inference.model_utils import predict_disease, load_model

def diagnosis_result_page():
    st.info("""
            ### Diagnosis Result
            The following is the diagnosis result of the disease you are experiencing. This result is based on the symptoms you provided and the image you uploaded.
            """)
    model, vectorizer, label_encoder, device = load_model()
    symptom_data = st.session_state.get('symptom_data', None)
    
    if symptom_data is not None:
        with st.spinner("Diagnosing..."):
            try:
                predictions = predict_disease(
                    model, vectorizer, label_encoder, 
                    symptom_data["processed_image"], 
                    symptom_data["symptom"],
                    device=device
                )
                
                # Store predictions in session state for later use
                st.session_state.diagnosis_results = predictions
                
                # Access primary prediction
                primary_prediction = predictions[0]
                primary_disease = primary_prediction['disease']
                primary_confidence = primary_prediction['confidence']
                
                if primary_disease == "Uncertain":
                    st.warning("⚠️ Low confidence prediction. Please consider the top possibilities below:")
                else:
                    st.success(f"🔍 Primary Diagnosis: {primary_disease.title()}")
                
                st.metric("Confidence", f"{primary_confidence:.1%}")
                st.progress(int(primary_confidence * 100))
                st.markdown("### Top 3 Possibilities:")
                cols = st.columns(3) 
                
                for idx, pred in enumerate(predictions[:3]):
                    with cols[idx]:
                        with st.container():
                            disease = pred['disease']
                            confidence = pred['confidence']
                            st.markdown(f"#### {disease.title()}")
                            st.markdown(f"**Confidence**: {confidence:.1%}")
                            
                            st.markdown("**Detected Symptoms:**")
                            for symptom in pred['detected_symptoms']:
                                st.markdown(f"- {symptom}")
                            
                            st.markdown("**Visual Analysis:**")
                            img_cols = st.columns(2)
                            with img_cols[0]:
                                st.image(symptom_data["processed_image"], caption="Original Image")
                            with img_cols[1]:
                                st.image(pred['visual_explanation']['heatmap'], caption="Model Focus")
                            st.markdown(pred['visual_explanation']['highlighted_regions'])
                
                if st.button("Next", type="primary", use_container_width=True, key="diagnosis_next_button"):
                    
                    st.session_state.completed_tabs["Result Detail"] = True
                    st.session_state.active_tab = "Result Detail"
                    st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("No symptom data available. Please fill in the symptom section first.")
    
