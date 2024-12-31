import streamlit as st
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
import io
from streamlit_pdf_viewer import pdf_viewer
import tempfile
import numpy as np
from PIL import Image as PILImage
import os


def create_reference_letter(disease, patient_name, patient_age, patient_gender, doctor_name, predictions, symptom_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("<b>AIZEN Diagnosis Result</b>", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Patient and Doctor Information
    patient_info = Paragraph(f"<b>Patient Name:</b> {patient_name}", styles['Normal'])
    patient_info_age = Paragraph(f"<b>Patient Age:</b> {patient_age}", styles['Normal'])
    patient_info_gender = Paragraph(f"<b>Patient Gender:</b> {patient_gender}", styles['Normal'])
    doctor_info = Paragraph(f"<b>Referred to:</b> {doctor_name}", styles['Normal'])
    condition_info = Paragraph(f"<b>Condition:</b> {disease}", styles['Normal'])
    symptom_info = Paragraph(f"<b>Symptom:</b> {symptom_data['symptom']}", styles['Normal'])
    story.extend([patient_info, patient_info_age, patient_info_gender, doctor_info, condition_info, symptom_info, Spacer(1, 12)])

    # Body of the Letter
    body = f"""
    Dear Dr. {doctor_name},<br/><br/>
    
    We are writing to refer {patient_name}, a {patient_age}-year-old {patient_gender}, to your esteemed care for further evaluation and management of {disease}. 
    Our preliminary diagnosis suggests that the patient may benefit from your specialized expertise in this area.<br/><br/>
    
    Enclosed with this letter, you will find detailed diagnostic results, including relevant medical images and symptom analysis, 
    which we hope will assist you in your assessment. We have conducted a thorough examination and believe that your insights 
    will be invaluable in determining the most appropriate course of action for {patient_name}.<br/><br/>
    
    Please do not hesitate to contact us should you require any additional information or clarification regarding the patient's medical history or current condition. 
    We are committed to ensuring a seamless transition of care and are available for any discussions that may aid in the patient's treatment.<br/><br/>
    
    Thank you for your attention to this matter and for your continued dedication to patient care.<br/><br/>
    
    Sincerely,<br/>
    Your Medical Team
    """
    body_paragraph = Paragraph(body, styles['Normal'])
    story.append(body_paragraph)
    story.append(Spacer(1, 12))

    # Add a page break to start a new page for the analysis images
    story.append(PageBreak())
    
    story.append(Paragraph("<b>Diagnosis Images and Details</b>", styles['Normal']))

    # Add Diagnosis Images and Details
    for idx, pred in enumerate(predictions[:3]):
        disease = pred['disease']
        confidence = pred['confidence']
        detected_symptoms = pred['detected_symptoms']
        visual_explanation = pred['visual_explanation']

        # Add disease and confidence
        story.append(Paragraph(f"<b>Diagnosis {idx + 1}: {disease.title()}</b>", styles['Normal']))
        story.append(Paragraph(f"Confidence: {confidence:.1%}", styles['Normal']))
        story.append(Spacer(1, 6))

        # Add detected symptoms
        story.append(Paragraph("<b>Detected Symptoms:</b>", styles['Normal']))
        for symptom in detected_symptoms:
            story.append(Paragraph(f"- {symptom}", styles['Normal']))
        story.append(Spacer(1, 6))

        # Add some padding before the visual analysis section
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Visual Analysis:</b>", styles['Normal']))
        story.append(Spacer(1, 6))  # Add some padding after the title

        # Check the type of processed image
        processed_image = symptom_data["processed_image"]
        if isinstance(processed_image, np.ndarray):
            pil_image = PILImage.fromarray(processed_image)
        elif isinstance(processed_image, PILImage.Image):
            pil_image = processed_image
        else:
            raise ValueError(f"Processed image is not in a supported format: {type(processed_image)}")

        # Save processed image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            pil_image.save(temp_file.name)
            original_image = Image(temp_file.name, width=140, height=105)

        # Convert heatmap to a PIL image
        heatmap = visual_explanation['heatmap']
        if isinstance(heatmap, np.ndarray):
            pil_heatmap = PILImage.fromarray(heatmap)
        elif isinstance(heatmap, PILImage.Image):
            pil_heatmap = heatmap
        else:
            raise ValueError(f"Heatmap is not in a supported format: {type(heatmap)}")

        # Save heatmap image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            pil_heatmap.save(temp_file.name)
            heatmap_image = Image(temp_file.name, width=140, height=105)

        # Create a table to align images side by side
        image_table = Table([[original_image, heatmap_image]], colWidths=[150, 150])
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(image_table)
        story.append(Spacer(1, 12))  # Add some padding after the images

    # Build the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def reference_letter_page():
    
    patient_data = st.session_state.get('patient_data', {})
    prediction = st.session_state.get('diagnosis_results')
    symptom_data = st.session_state.get('symptom_data')
    
    if patient_data and prediction and symptom_data:
        primary_prediction = prediction[0]
        
        disease = primary_prediction['disease']
        patient_name = patient_data.get('full_name', '')
        patient_age = patient_data.get('age', '')
        patient_gender = patient_data.get('gender', '')
        
        # Input field for doctor's name
        doctor_name = st.text_input("Enter the doctor's name", "Dr. Smith")

        # Create PDF
        pdf_buffer = create_reference_letter(disease, patient_name, patient_age, patient_gender, doctor_name, prediction, symptom_data)

        # Create two columns
        col1, col2 = st.columns(2)

        with col2:
            # Display PDF using Streamlit PDF Viewer
            pdf_bytes = pdf_buffer.getvalue()
            pdf_viewer(pdf_bytes)

        with col1:
            # Provide a download button
            st.info("""
                    ### Reference Letter
                    Please download the diagnosis referral letter data from AIZEN. 
                    This data is expected to help and shorten further diagnosis by medical personnel or doctors who handle patients.
                    """)
            st.download_button(
                label="Download Reference Letter",
                data=pdf_buffer,
                file_name="reference_letter.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("Please complete the previous steps before generating a reference letter.")
