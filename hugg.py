from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
import PIL.Image as Image

#Loading Brain Data
brain = load_dataset("TrainingDataPro/brain-mri-dataset")
brainFrame = brain['train'].to_pandas()
brainFrame['image'] = brainFrame['image'].astype(str)
# Ensure that only the first 22 characters are removed
brainFrame['image'] = brainFrame['image'].str[24:]
brainFrame['image'] = brainFrame['image'].str[:-1]
brainFrame.drop(columns=['label'], inplace=True)
# Check the first few rows after slicing
brainFrame['category'] = 'MRI_BRAIN'
brainFrame.to_csv('./brainFrame.csv', index=False)

#Loading Lumbar Spine Data
spine = load_dataset("UniDataPro/lumbar-spine-mri")
spineFrame = spine['train'].to_pandas()
spineFrame['image'] = spineFrame['image'].astype(str)
spineFrame['image'] = spineFrame['image'].str[24:]
spineFrame['image'] = spineFrame['image'].str[:-1]
spineFrame.drop(columns=['label'], inplace=True)
spineFrame['catgeory'] = 'MRI_SPINE'
spineFrame.to_csv('./spineFrame.csv', index=False)

#Loading Chest CT data
chest = load_dataset("Mahadih534/Chest_CT-Scan_images-Dataset")
chestFrame = chest['train'].to_pandas()
chestFrame['image'] = chestFrame['image'].astype(str)
chestFrame['image'] = chestFrame['image'].str[24:]
chestFrame['image'] = chestFrame['image'].str[:-1]
chestFrame.drop(columns=['label'], inplace=True)
chestFrame['catgeory'] = 'CT_CHEST'
chestFrame.to_csv('./chestFrame.csv', index=False)

# Merging the DataFrames
mergedFrame = pd.concat([brainFrame, spineFrame, chestFrame], ignore_index=True)

# Save the merged DataFrame to a CSV file
mergedFrame.to_csv('./merged_mri_ct_data.csv', index=False)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the model's expected input size
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
])