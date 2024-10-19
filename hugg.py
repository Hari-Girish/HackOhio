from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
import PIL.Image as Image
brain = load_dataset("TrainingDataPro/brain-mri-dataset")
brainFrame = brain['train'].to_pandas()
brainFrame['image'] = brainFrame['image'].astype(str)
# Ensure that only the first 22 characters are removed
brainFrame['image'] = brainFrame['image'].str[24:]
brainFrame['image'] = brainFrame['image'].str[:-1]

# Check the first few rows after slicing
print(brainFrame['image'].head())
brainFrame.to_csv('./MRI_BRAIN/brainFrame.csv', index=False)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the model's expected input size
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
])