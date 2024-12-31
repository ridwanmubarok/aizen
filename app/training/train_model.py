import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd

class MultimodalDataset(Dataset):
    def __init__(self, image_paths, text_features, labels, transform=None):
        self.image_paths = image_paths
        self.text_features = text_features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        text_feature = torch.FloatTensor(self.text_features[idx])
        label = self.labels[idx]
        return image, text_feature, label

class MultimodalMLP(nn.Module):
    def __init__(self, text_size, hidden_size_1, hidden_size_2, output_size):
        super(MultimodalMLP, self).__init__()
        self.image_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.image_model.parameters():
            param.requires_grad = False
        num_ftrs = self.image_model.fc.in_features
        self.image_model.fc = nn.Linear(num_ftrs, hidden_size_2)
        self.text_fc1 = nn.Linear(text_size, hidden_size_2)
        self.combined_fc = nn.Linear(hidden_size_2 * 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, image, text):
        img_feat = self.image_model(image)
        img_feat = self.dropout(img_feat)
        text_feat = self.relu(self.text_fc1(text))
        text_feat = self.dropout(text_feat)
        combined = torch.cat((img_feat, text_feat), dim=1)
        output = self.combined_fc(combined)
        return output

def load_dataset(data_dir, symptoms_csv_path):
    image_paths = []
    symptoms_texts = []
    labels = []
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")
    
    # Read symptoms from CSV file
    symptoms_df = pd.read_csv(symptoms_csv_path)
    
    # Combine symptoms from all language columns
    language_columns = [col for col in symptoms_df.columns if col.startswith('Symptoms_')]
    data_dict = {}
    for _, row in symptoms_df.iterrows():
        # Concatenate symptoms from all language columns
        symptoms_text = ' '.join([row[col] for col in language_columns if pd.notna(row[col])])
        data_dict[row['Disease'].lower()] = symptoms_text
    
    for disease in os.listdir(data_dir):
        disease_path = os.path.join(data_dir, disease)
        if os.path.isdir(disease_path):
            symptoms = data_dict.get(disease.lower(), "")
            image_count = 0
            for img_name in os.listdir(disease_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(disease_path, img_name))
                    symptoms_texts.append(symptoms)
                    labels.append(disease)
                    image_count += 1
            print(f"Loaded {image_count} images from {disease}")
    if not image_paths:
        raise ValueError("No images found in directory")
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Unique labels: {set(labels)}")
    return image_paths, symptoms_texts, labels

def train_model(data_dir, symptoms_csv_path):
    print(f"Loading data from: {os.path.abspath(data_dir)}")
    batch_size = 16
    learning_rate = 0.0001
    num_epochs = 100
    hidden_size = 512
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset with both image paths and symptoms data
    image_paths, symptoms_texts, labels = load_dataset(data_dir, symptoms_csv_path)
    
    vectorizer = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 3),
        stop_words='english',
        min_df=2
    )
    text_features = vectorizer.fit_transform(symptoms_texts).toarray()
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(encoded_labels),
        y=encoded_labels
    )
    class_weights = torch.FloatTensor(class_weights)
    X_img_train, X_img_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
        image_paths, text_features, encoded_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=encoded_labels
    )
    train_dataset = MultimodalDataset(X_img_train, X_text_train, y_train, transform=transform)
    test_dataset = MultimodalDataset(X_img_test, X_text_test, y_test, transform=transform)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=[class_weights[label] for label in y_train],
        num_samples=len(y_train),
        replacement=True
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    model = MultimodalMLP(
        text_size=text_features.shape[1],
        hidden_size_1=2048,
        hidden_size_2=1024,
        output_size=len(le.classes_)
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.02,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=8,
        verbose=True,
        min_lr=1e-6
    )
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_val_accuracy = 0.0
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        for images, texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, texts, labels in test_loader:
                outputs = model(images, texts)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(test_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        scheduler.step(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        save_path = "app/model/aizen_model.pth"
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'vectorizer': vectorizer,
                'label_encoder': le,
                'val_accuracy': val_accuracy,
                'epoch': epoch
            }, save_path)
            print(f"New best model saved! Validation Accuracy: {val_accuracy:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    print(f"Training set size: {len(X_img_train)}")
    print(f"Test set size: {len(X_img_test)}")
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")
    return model, vectorizer, le

if __name__ == "__main__":
    data_dir = "app/training/dataset"
    symptoms_csv_path = "app/training/data/symptoms.csv"
    model, vectorizer, label_encoder = train_model(data_dir, symptoms_csv_path)
