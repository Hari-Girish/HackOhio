#server_7P44XNWJQNLJ7SOEYJ7MNXOR-5HG5PP5NOSGRV2R6
from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import PIL.Image as Image
from PIL import Image
import anvil.server

# Load the model (this assumes your model is saved in a file called 'model.pth')
model = torch.load('model.pth')
model.eval()  # Set the model to evaluation mode

# Preprocessing transformations for the uploaded image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

# This function is called from the client-side
@anvil.server.callable
def compare_to_user(uploaded_image):
    # Load the image from the uploaded file
    image = Image.open(io.BytesIO(uploaded_image.get_bytes()))
    
    # Apply transformations
  Testing Locally: Before deploying, run your server code locally to ensure that it works as expected. If there are any issues, fix them before trying to deploy again.Testing Locally: Before deploying, run your server code locally to ensure that it works as expected. If there are any issues, fix them before trying to deploy again.Testing Locally: Before deploying, run your server code locally to ensure that it works as expected. If there are any issues, fix them before trying to deploy again.  image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get the predicted class
    predicted_class = torch.argmax(output, dim=1)

    # Map predicted class to label
    label_map = {0: 'MRI_BRAIN', 1: 'MRI_SPINE', 2: 'CT_CHEST'}
    predicted_label = label_map[predicted_class.item()]
    
    return predicted_label
