import streamlit as st
from app.inference.generative_utils import generate_diagnosis_explanation
import json
import pandas as pd

def result_detail_page():
    st.info("Result Detail")

    # Retrieve diagnosis results from session state
    diagnosis_results = st.session_state.get('diagnosis_results', None)
    patient_data = st.session_state.get('patient_data')
    prediction = st.session_state.get('diagnosis_results')
    symptom_data = st.session_state.get('symptom_data')

    if prediction:
        primary_prediction = prediction[0]
        primary_disease = primary_prediction['disease']
        primary_confidence = primary_prediction['confidence']
        
        list_predictions = []
        for idx, pred in enumerate(prediction[:3]):
            disease = pred['disease']
            confidence = pred['confidence']
            data = {
                "disease": disease.title(),
                "confidence": f"{confidence:.1%}",
            }
            list_predictions.append(data)

        season = patient_data.get('season')
        country = patient_data.get('country')
        age = patient_data.get('age')
        gender = patient_data.get('gender')
        language = patient_data.get('language')
        symptoms = symptom_data.get('symptom')
        
        # Generate explanation
        explanation_data = generate_diagnosis_explanation(
            primary_disease, 
            primary_confidence, 
            symptoms, 
            country, 
            season, 
            language
        )
        
        explanation_data = explanation_data.strip()
        
        # Parse the explanation data as JSON
        parsed_explanation = parse_explanation_data(explanation_data)

        if parsed_explanation:
            st.title("Diagnosis Explanation")
            st.markdown(f"**Explanation:** {parsed_explanation['explanation']}")

            # Create two columns
            col1, col2 = st.columns(2)

            with col1:
                # Display recommendations
                st.subheader("Recommendations")
                for recommendation in parsed_explanation['recommendations']:
                    st.markdown(
                        f"""
                        <div style='
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 10px;
                            margin-bottom: 10px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                        '>
                            <p style='margin: 0;'>{recommendation}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                # Display first aid tips
                st.subheader("First Aid Tips")
                for tip in parsed_explanation['first_aid']:
                    st.markdown(
                        f"""
                        <div style='
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 10px;
                            margin-bottom: 10px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                        '>
                            <p style='margin: 0;'>{tip}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                # Display over-the-counter options
                st.subheader("Over-the-Counter Options")
                for otc in parsed_explanation['over_the_counter']:
                    st.markdown(
                        f"""
                        <div style='
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 10px;
                            margin-bottom: 10px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                        '>
                            <p style='margin: 0;'>{otc}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                # Display seasonal advice
                st.subheader("Seasonal Advice")
                st.markdown(f"**Season:** {parsed_explanation['season']}")

            with col2:
                # Display traditional medicine options
                st.subheader("Traditional Medicine")
                for medicine in parsed_explanation['traditional_medicine']:
                    st.markdown(
                        f"""
                        <div style='
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 10px;
                            margin-bottom: 10px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                        '>
                            <strong>{medicine}</strong>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                # Display modern medicine advice
                st.subheader("Modern Medicine")
                for medicine in parsed_explanation['modern_medicine']:
                    st.markdown(
                        f"""
                        <div style='
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 10px;
                            margin-bottom: 10px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                        '>
                            <strong>{medicine}</strong>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                # Display nearby health centers
                st.subheader("Nearby Health Centers")
                health_centers_df = pd.DataFrame(parsed_explanation['health_centers'])

                # Convert latitude and longitude to float
                health_centers_df['latitude'] = health_centers_df['latitude'].astype(float)
                health_centers_df['longitude'] = health_centers_df['longitude'].astype(float)

                # Add a Google Maps link column
                health_centers_df['Google Maps'] = health_centers_df.apply(
                    lambda row: f"https://www.google.com/maps/search/?api=1&query={row['latitude']},{row['longitude']}",
                    axis=1
                )

                for index, row in health_centers_df.iterrows():
                    st.markdown(
                        f"""
                        <div style='
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 15px;
                            margin-bottom: 15px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                        '>
                            <h4 style='margin: 0;'>{row['name']}</h4>
                            <p style='margin: 5px 0;'>Opening Hours: {row['opening_hours']}</p>
                            <a href='{row['Google Maps']}' target='_blank' style='color: #007BFF; text-decoration: none;'>Open in Google Maps</a>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
            if st.button("Next to Generate Reference Letter", type="primary", use_container_width=True):
                st.write("Reference Letter Generated")

def parse_explanation_data(explanation_data):
    try:
        if explanation_data.startswith("```") and explanation_data.endswith("```"):
            explanation_data = explanation_data[3:-3].strip()
        explanation_data = explanation_data.replace("json", "").strip()
        parsed_data = json.loads(explanation_data)
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        return None


