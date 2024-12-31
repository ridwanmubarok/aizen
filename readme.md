# AIZEN

**Brief Description:**
***AIZEN*** is a prototype artificial intelligence (AI)-based solution designed to assist healthcare professionals in the diagnostic process of various diseases. In its initial phase, this application focuses on helping diagnose skin diseases such as acne, dermatitis, eczema, melanoma, and psoriasis. However, the long-term goal of this application is to expand its capabilities to diagnose a wider range of diseases, both dermatological and non-dermatological, targeting more complex medical conditions such as tuberculosis (TB), cancer, tumors, and other chronic diseases.

***AIZEN*** leverages AI technology to analyze symptoms reported by patients, either through descriptions or images, and provides preliminary diagnoses that can assist healthcare professionals in determining treatment steps or further follow-up. By utilizing AI, this prototype application aims to expedite the process of accurate diagnosis, especially in areas with limited access to medical facilities or advanced diagnostic tools.

As the prototype develops, we plan to integrate radiology technology into the system, allowing users to analyze medical images such as X-rays, CT scans, and MRIs. This AI-based radiology technology will enable the detection of abnormalities or early signs of serious diseases such as cancer and tumors, facilitating early detection, which is crucial for treatment and recovery.

It is important to note that this application is not intended to replace healthcare professionals but rather to serve as a supportive tool that can assist them in providing faster and more accurate diagnoses. This prototype aims to help healthcare professionals, especially in situations where time is critical and access to medical facilities is limited.

With ongoing development, we hope this prototype can further expand its capabilities, provide broader access for the community to receive faster medical diagnoses, and support healthcare professionals worldwide, particularly in underserved areas. This application is committed to continuous improvement by adding various types of diseases and more advanced diagnostic technologies, focusing on enhancing accuracy and ease of use.

## Key Features

This prototype application offers several key features designed to support healthcare professionals in the diagnostic process of diseases. The available features include:

- **AI-Based Disease Diagnosis**  
  Utilizing artificial intelligence technology, this prototype can assist in diagnosing various diseases based on symptoms reported by patients. Users can upload images or describe symptoms to receive an accurate preliminary analysis, which will support medical decision-making.

- **Medical Image Analysis**  
  This feature allows users to upload and analyze medical images, to detect abnormalities or early signs of diseases.

- **Referral for Follow-Up Steps**  
  After a preliminary diagnosis is provided, this prototype offers referral recommendations for appropriate follow-up. These referrals may include suggestions for further treatment, additional examinations, or consultations with specialists, depending on the type of disease detected.

- **Reports and Statistics**  
  This prototype provides reports and statistics that are useful for healthcare professionals to monitor disease trends and treatment effectiveness, as well as assist in data-driven decision-making.


## Data Validity

In this prototype, the validity of the data may not be fully established as we have not yet found a reliable dataset based on medical records. For image datasets, we have sourced some images from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). However, for symptom data, we have not obtained a valid dataset and have only compiled information based on a review of several articles.


## Technologies Used

This prototype application is built using the latest technologies in software development and artificial intelligence, including:

- **Python Streamlit**: For creating an interactive and user-friendly web interface.
- **OpenAI**: For implementing the language model used in text analysis and symptom interpretation.
- **Machine Learning**: To analyze data and provide accurate diagnoses.
- **Deep Learning**: For image processing and radiology analysis.


## Installation Instructions

1. **Clone the Repository**  
   Clone the repository to your local machine using the following command:
   ```bash
   git clone https://github.com/your-repository-url.git

2. **Copy the .env.example file**  
   Copy the .env.example file to .env and fill in the necessary variables:
   ```bash
   cp .env.example .env
   ```

3. **Navigate to the Project Directory**  
   Navigate to the project directory:
   ```bash
   cd your-project-directory
   ```

4. **Virtual Environment**  
   Create a virtual environment:
   ```bash
   python -m venv venv
   ```

5. **Activate the Virtual Environment**  
   Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

6. **Install Dependencies**  
   Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

7. **Run the Application**  
   Start the application by running the following command:
   ```bash
   streamlit run app/main.py
   ```


### Troubleshooting
if you encounter an error related to `No module named 'app'`, you can try to running the following commands:

```powershell
$env:PYTHONPATH="$env:PYTHONPATH;your_directory_project"
```

```bash
export PYTHONPATH=$PYTHONPATH:your_directory_project
```

## Acknowledgments

We would like to express our gratitude to the following individuals and organizations for their contributions to this project:

- [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) for providing the image dataset.
- [OpenAI](https://openai.com/) for providing the language model used in text analysis.
- [Streamlit](https://streamlit.io/) for providing the web framework used in this project.


## DEMO PREVIEW

- [https://drive.google.com/file/d/1OYpmklPnWu5JDZgBUqf6pES_RckxcCt3/view?usp=sharing](https://drive.google.com/file/d/1OYpmklPnWu5JDZgBUqf6pES_RckxcCt3/view?usp=sharing)

## DEMO LINKS

- [https://aizen.cloud-ind.my.id/](https://aizen.cloud-ind.my.id/)
