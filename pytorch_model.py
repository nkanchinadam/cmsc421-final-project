import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchinfo import summary
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import time
import os
from PIL import Image
import pandas as pd

train_path = "./PathAndClassTrain.csv"
val_path = "./PathAndClassVal.csv"
# define the image transformations 
IMAGE_SIZE = 224
data_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
                                             transforms.ToTensor()])
data_transform_val = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),transforms.ToTensor()])

class CustomDataSet(Dataset):
    def __init__(self, csv_file, class_list=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.class_list = class_list

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = Image.open(self.df.path[index])
        label = self.df.label[index]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data(batch_size=64):
    dataset_train = CustomDataSet(csv_file=train_path, transform=data_transform)
    dataset_val = CustomDataSet(csv_file=val_path, transform=data_transform_val)
    
    # Create data loaders.
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_val, 
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, valid_loader

def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_loss.png'))

# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = [[float(val) for val in label[1:-1].split(', ')] for label in labels]
        labels = torch.tensor(labels)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        #_, preds = torch.max(outputs.data, 1)
        #print(preds)
        diffs = torch.sub(outputs.data, labels)
        accuracy = torch.sum(diffs*diffs, dim=1)
        vals = torch.tensor([1.0 if acc <= 0.05 else 0.0 for acc in accuracy])
        train_running_correct += vals.sum().item()
        # Backpropagation

        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = [[float(val) for val in label[1:-1].split(', ')] for label in labels]
            labels = torch.tensor(labels)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            #_, preds = torch.max(outputs.data, 1)
            diffs = torch.sub(outputs.data, labels)
            accuracy = torch.sum(diffs*diffs, dim=1)
            vals = torch.tensor([1.0 if acc <= 0.05 else 0.0 for acc in accuracy])
            valid_running_correct += vals.sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


# Pre-trained resnet that we will freeze
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False

# our model that we will train
our_layers = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 5),
    nn.Sigmoid()
)

model.fc = our_layers

summary(model, input_size=(1, 3, 224, 224))

epochs=20
batch_size=64
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader, valid_loader = get_data(batch_size=batch_size)

# Optimizer.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# Loss function.
criterion = nn.BCEWithLogitsLoss()

# Lists to keep track of losses and accuracies.
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# Start the training.
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(
        model, 
        train_loader, 
        optimizer, 
        criterion,
        device
    )
    valid_epoch_loss, valid_epoch_acc = validate(
        model, 
        valid_loader, 
        criterion,
        device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)
    
# Save the loss and accuracy plots.
save_plots(
    train_acc, 
    valid_acc, 
    train_loss, 
    valid_loss, 
    name="Random"
)
print('TRAINING COMPLETE')