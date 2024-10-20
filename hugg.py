from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import PIL.Image as Image
from torchvision.models import resnet18, ResNet18_Weights

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
    spineFrame['category'] = 'MRI_SPINE'
    spineFrame.to_csv('./spineFrame.csv', index=False)

    #Loading Chest CT data
    chest = load_dataset("Mahadih534/Chest_CT-Scan_images-Dataset")
    chestFrame = chest['train'].to_pandas()
    chestFrame['image'] = chestFrame['image'].astype(str)
    chestFrame['image'] = chestFrame['image'].str[24:]
    chestFrame['image'] = chestFrame['image'].str[:-1]
    chestFrame.drop(columns=['label'], inplace=True)
    chestFrame['category'] = 'CT_CHEST'
    chestFrame.to_csv('./chestFrame.csv', index=False)

    # Cleaning image paths to remove extra quotes or unexpected characters
    brainFrame['image'] = brainFrame['image'].str.strip("'")  # Strip any leading/trailing single quotes
    spineFrame['image'] = spineFrame['image'].str.strip("'")
    chestFrame['image'] = chestFrame['image'].str.strip("'")

    # Merging the DataFrames
    mergedFrame = pd.concat([brainFrame, spineFrame, chestFrame], ignore_index=True)

    # Save the merged DataFrame to a CSV file
    mergedFrame.to_csv('./merged_mri_ct_data.csv', index=False)

    # Dataset class
    class MRI_CT_Dataset(Dataset):
        def __init__(self, dataframe, transform=None):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            img_path = self.dataframe.iloc[idx, 0]  # Assuming 'image' column is the first
            label = self.dataframe.iloc[idx, 1]     # Assuming 'category' column is second

            # Load the image
            image = Image.open(img_path).convert("RGB")  # Convert grayscale to RGB if needed

            # Apply any transformations
            if self.transform:
                image = self.transform(image)

            # Convert label to a numerical value (for multi-class classification)
            label_map = {'MRI_BRAIN': 0, 'MRI_SPINE': 1, 'CT_CHEST': 2}
            label = label_map[label]

            return image, label

    # Transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained models
    ])

    def main():
        dataset = MRI_CT_Dataset(mergedFrame, transform=transform)

        # Create DataLoader for training and validation
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        # Load the pre-trained ResNet18 model with updated weights argument
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)

        # Replace the final fully connected layer to match your number of output classes (3)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 3)  # 3 output classes: MRI_BRAIN, MRI_SPINE, CT_CHEST

        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 10

        for epoch in range(num_epochs):
            running_loss = 0.0
            model.train()  # Set model to training mode

            for images, labels in dataloader:
                # Move data to the appropriate device (CPU or GPU)
                images, labels = images.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Track loss
                running_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
        torch.save(model, ".//model.pth")

    if __name__ == '__main__':
        main()
