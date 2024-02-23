#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install datasets
#!pip install --upgrade --force-reinstall huggingface_hub


# In[12]:


import torch
import torchvision
from torchvision.transforms import transforms, Lambda, Resize
from torchvision import datasets, transforms, models
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class CustomDataLoader:
    def __init__(self, data_path, batch_size, dataset_type):
        self.data_path = data_path
        self.batch_size = batch_size
        self.dataset_type = dataset_type

        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),  # Resize to size 48x48
            transforms.Grayscale(),  # Convert all images to grayscale format
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.2295])
        ])

        self.dataset = torchvision.datasets.ImageFolder(root=f"{self.data_path}/{dataset_type}", transform=self.transform)

        if self.dataset_type == 'train':
            shuffle = True
        else:
            shuffle = False

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


# In[3]:


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# In[7]:


data_dir = 'FER2013'

# Create data loaders
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
#image_datasets


# In[8]:


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
print(dataset_sizes)

class_names = image_datasets['train'].classes
class_names


# In[13]:


# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Freeze all layers except the final classification layer
for name, param in model.named_parameters():
    if "fc" in name:  # Unfreeze the final classification layer
        param.requires_grad = True
    else:
        param.requires_grad = False

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Use all parameters


# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[15]:


num_epochs = 10
for epoch in range(num_epochs):
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print("Training complete!")


# In[16]:


# Save the model
torch.save(model.state_dict(), 'Image_classification_model.pth')


# In[ ]:




