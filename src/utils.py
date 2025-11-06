import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def load_class_names(path):
    """
    Load class names from JSON file
    
    Args:
        path: Path to JSON file containing class names
        
    Returns:
        dict: Dictionary mapping integer class indices to class names
    """
    with open(path, 'r') as f:
        class_names = json.load(f)
    # Convert string keys to integers
    return {int(k): v for k, v in class_names.items()}

def preprocess_image(image):
    """
    Preprocess PIL image for model input
    
    Args:
        image: PIL Image object
        
    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 3, H, W)
    """
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, top_k=5):
    """
    Make prediction and return top k results
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        top_k: Number of top predictions to return
        
    Returns:
        tuple: (top_probabilities, top_class_indices)
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, top_k)
    
    return top_probs[0], top_classes[0]