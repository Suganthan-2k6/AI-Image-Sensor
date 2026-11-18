import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np

class AIImageDetector:
    def __init__(self, model_path=None):
        # Load pre-trained EfficientNet-B0
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Replace classifier for binary classification (real vs AI)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2)  # 2 classes: [real, ai]
        )
        
        # Load custom weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """Predict if image is Real or AI-generated"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction and confidence
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            
            # Class 0 = Real, Class 1 = AI
            label = "🟢 Real" if pred_class == 0 else "🔴 AI-Generated"
            
            return {
                "label": label,
                "confidence": round(confidence * 100, 2),
                "prediction": "real" if pred_class == 0 else "ai",
                "score": {
                    "real": round(probabilities[0][0].item() * 100, 2),
                    "ai": round(probabilities[0][1].item() * 100, 2)
                }
            }