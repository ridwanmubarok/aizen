import os
import openai
from app.const.prompt_data import PROMPT_LANG_FORMAT
from dotenv import load_dotenv

load_dotenv()

def generate_diagnosis_explanation(disease, confidence, symptoms, location=None, season=None, language='en'):
    """
    Generates a detailed explanation of the diagnosis using OpenAI's GPT-4o
    """
    # Format the prompt using the selected language template
    prompt = PROMPT_LANG_FORMAT[language].format(
        disease=disease,
        confidence=confidence,
        symptoms=symptoms if symptoms else "Not specified",
        location=location if location else "Not specified",
        season=season if season else "Not specified",
    )
    # Use environment variable for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    # Set the API key
    openai.api_key = api_key
    
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a very helpful assistant for medical diagnosis. Your task is to provide comprehensive recommendations based on existing data from the system as well as your own knowledge."
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nPlease respond in the following JSON format:\n"
                               "{\n"
                               "  \"explanation\": \"<explanation>\",\n"
                               "  \"recommendations\": \"List<recommendations>\",\n"
                               "  \"first_aid\": \"List<first_aid>\",\n"
                               "  \"over_the_counter\": \"List<over_the_counter>\",\n"
                               "  \"season\": \"<season>\",\n"
                               "  \"traditional_medicine\": \"List<traditional_medicine>\",\n"
                               "  \"modern_medicine\": \"List<modern_medicine>\",\n"
                               "  \"health_centers\": \"List<health_centers, latitude, longitude, opening_hours>\",\n"
                               "}"
                }
            ],
        )
    except Exception as e:
        # Handle exceptions from the API call
        print(f"An error occurred: {e}")
        return None

    return completion.choices[0].message.content
