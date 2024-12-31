import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from app.training.train_model import MultimodalMLP
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

model_path = "app/model/aizen_model.pth"

def load_model(language='en'):
    """
    Loads the trained model and required preprocessing components
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)

        text_size = checkpoint['vectorizer'].get_feature_names_out().shape[0]
        model = MultimodalMLP(
            text_size=text_size,
            hidden_size_1=2048,
            hidden_size_2=1024,
            output_size=len(checkpoint['label_encoder'].classes_)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        return model, checkpoint['vectorizer'], checkpoint['label_encoder'], device
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def generate_occlusion_map(model, image_tensor, text_tensor, target_idx, device, window_size=64, stride=32):
    """
    Generate sensitivity map using occlusion method (optimized version)
    """
    width = image_tensor.shape[2]
    height = image_tensor.shape[3]
    
    sensitivity_map = torch.zeros((1, 1, width, height))
    
    batch_size = 16
    occluded_images = []
    positions = []
    
    for x in range(0, width-window_size, stride):
        for y in range(0, height-window_size, stride):
            occluded_image = image_tensor.clone()
            occluded_image[0, :, x:x+window_size, y:y+window_size] = 0
            occluded_images.append(occluded_image)
            positions.append((x, y))
            
            if len(occluded_images) == batch_size:
                batch_tensor = torch.cat(occluded_images, dim=0)
                batch_text = text_tensor.repeat(batch_size, 1)
                
                with torch.no_grad():
                    outputs = model(batch_tensor.to(device), batch_text)
                    probs = torch.softmax(outputs, dim=1)[:, target_idx]
                
                for idx, (x_pos, y_pos) in enumerate(positions):
                    sensitivity_map[0, 0, x_pos:x_pos+window_size, y_pos:y_pos+window_size] += (1 - probs[idx].item())
                
                occluded_images = []
                positions = []
    
    if occluded_images:
        batch_tensor = torch.cat(occluded_images, dim=0)
        batch_text = text_tensor.repeat(len(occluded_images), 1)
        
        with torch.no_grad():
            outputs = model(batch_tensor.to(device), batch_text)
            probs = torch.softmax(outputs, dim=1)[:, target_idx]
        
        for idx, (x_pos, y_pos) in enumerate(positions):
            sensitivity_map[0, 0, x_pos:x_pos+window_size, y_pos:y_pos+window_size] += (1 - probs[idx].item())
    
    sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min())
    return sensitivity_map

def predict_disease(model, vectorizer, label_encoder, image, symptoms=None, language='en', device=None):
    model.eval()
    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        original_image = image.copy()
        
        image_tensor = transform(image).unsqueeze(0)
        if device:
            image_tensor = image_tensor.to(device)

        text_feature = vectorizer.transform([symptoms]).toarray()
        feature_names = vectorizer.get_feature_names_out()
        important_symptoms = [
            feature_names[i] for i in range(len(text_feature[0]))
            if text_feature[0][i] > 0
        ]

        text_tensor = torch.FloatTensor(text_feature)
        if device:
            text_tensor = text_tensor.to(device)

        outputs = model(image_tensor, text_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        top3_confidences, top3_indices = torch.topk(probabilities, 3, dim=1)
        top3_diseases = label_encoder.inverse_transform(top3_indices.cpu().numpy()[0])
        top3_confidences = top3_confidences.cpu().numpy()[0]

        predictions_with_explanation = []
        for idx, (disease, confidence) in enumerate(zip(top3_diseases, top3_confidences)):
            sensitivity_map = generate_occlusion_map(
                model, 
                image_tensor, 
                text_tensor, 
                top3_indices[0][idx].item(), 
                device
            )
            
            heatmap = sensitivity_map[0, 0].cpu().numpy()
            heatmap = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            
            original_np = np.array(original_image)
            overlay = cv2.addWeighted(original_np, 0.7, heatmap_colored, 0.3, 0)
            
            explanation = {
                'disease': disease,
                'confidence': confidence,
                'detected_symptoms': important_symptoms,
                'visual_explanation': {
                    'original_image': original_image,
                    'heatmap': overlay,
                    'highlighted_regions': (
                        "Red area indicates the most influential part of the image in diagnosis. "
                        "When this area is occluded, the model's confidence in diagnosis decreases significantly."
                    )
                }
            }
            predictions_with_explanation.append(explanation)

        return predictions_with_explanation

def suggest_symptoms(user_input, vectorizer, language='en'):
    """
    Suggest symptoms based on user input using a TF-IDF vectorizer and cosine similarity.
    Supports multiple languages (en, id, ja).
    """
    if not user_input:
        return []

    symptoms_df = pd.read_csv("app/training/data/symptoms.csv")
    symptoms_col = f'Symptoms_{language}'
    if symptoms_col not in symptoms_df.columns:
        raise ValueError(f"Unsupported language: {language}")
    all_symptoms = []
    for symptoms_text in symptoms_df[symptoms_col]:
        symptoms = [s.strip() for s in symptoms_text.split(',')]
        all_symptoms.extend(symptoms)
    symptom_list = list(dict.fromkeys(all_symptoms))
    lang_vectorizer = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 3),
        stop_words='english' if language == 'en' else None,
        min_df=1
    )
    
    symptom_vectors = lang_vectorizer.fit_transform(symptom_list)
    user_vector = lang_vectorizer.transform([user_input.lower()])

    if user_vector.size == 0 or symptom_vectors.size == 0:
        return []

    similarities = cosine_similarity(user_vector, symptom_vectors).flatten()
    threshold = 0.1
    top_indices = np.where(similarities > threshold)[0]
    
    suggested_symptoms = [symptom_list[i] for i in top_indices]
    suggested_symptoms = sorted(
        suggested_symptoms, 
        key=lambda x: similarities[symptom_list.index(x)], 
        reverse=True
    )
    
    return suggested_symptoms[:5]