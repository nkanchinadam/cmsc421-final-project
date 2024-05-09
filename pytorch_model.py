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
from torcheval.metrics import BinaryAccuracy
from torchvision.transforms import ToTensor, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score

import os
from PIL import Image
import pandas as pd

train_path = "./PathAndClassTrain.csv"
val_path = "./PathAndClassVal.csv"
# define the image transformations 
IMAGE_SIZE = 224
data_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomRotation(degrees=45),
                                     transforms.ToTensor()])
data_transform_val = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                         transforms.ToTensor()])

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
    train_running_count = 0
    train_total_count = 0
    # metric = BinaryAccuracy(threshold=0)
    counter = 0
    total_labels = []
    total_outputs = []
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = [[float(val) for val in label[1:-1].split(', ')] for label in labels]
        for l in labels:
            total_labels.append(l)
        labels = torch.tensor(labels)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        #_, preds = torch.max(outputs.data, 1)
        #print(preds)
        
        #for i in range(len(outputs.data)):
            #metric.update(outputs.data[i, :], labels[i, :])
        

        # Accuracy + F1 Score
        for i in range(len(outputs.data)):
            output = outputs.data[i, :]
            label = labels[i,:]
            output = [1.0 if o >= 0 else 0.0 for o in output]
            total_outputs.append(output)
            output = torch.tensor(output)
            train_total_count += 1
            # if torch.all(torch.eq(output, label)):
            #     train_running_count += 1
            if label[np.argmax(output)] == 1:
                train_running_count += 1

        # Backpropagation

        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    # epoch_acc = metric.compute()
    epoch_acc = train_running_count / train_total_count
    epoch_f1 = f1_score(total_labels, total_outputs, average='samples', zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    #metric = BinaryAccuracy(threshold=0)
    counter = 0
    valid_running_count = 0
    total_running_count = 0
    total_labels = []
    total_outputs = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = [[float(val) for val in label[1:-1].split(', ')] for label in labels]
            for l in labels:
                total_labels.append(l)
            labels = torch.tensor(labels)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            #_, preds = torch.max(outputs.data, 1)

            # [0, 1, 0, 0, 0] and [1, 0, 0, 0, 0] accuracy of 60% 
            #for i in range(len(outputs.data)):
                #metric.update(outputs.data[i, :], labels[i, :])

            # Accuracy + F1 Score
            for i in range(len(outputs.data)):
                output = outputs.data[i, :]
                label = labels[i,:]
                output = [1.0 if o >= 0 else 0.0 for o in output]
                total_outputs.append(output)
                output = torch.tensor(output)
                total_running_count += 1
                # if torch.all(torch.eq(output, label)):
                #     valid_running_count += 1
                if label[np.argmax(output)] == 1:
                    valid_running_count += 1

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    #epoch_acc = metric.compute()
    epoch_acc = valid_running_count / total_running_count
    epoch_f1 = f1_score(total_labels, total_outputs, average='samples', zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


# Pre-trained resnet that we will freeze
#model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
model = torchvision.models.googlenet(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False

#model.fc = nn.Linear(512, 5)
model.fc = nn.Linear(1024, 5)
summary(model, input_size=(1, 3, 224, 224))

epochs=4
batch_size=64
learning_rate = 0.1
label_name = ['Crime', 'Action', 'Romance', 'Comedy', 'Drama']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader, valid_loader = get_data(batch_size=batch_size)

# Optimizer.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# Loss function.
criterion = nn.BCEWithLogitsLoss()

# Lists to keep track of losses and accuracies.
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
train_f1, valid_f1 = [], []
# Start the training.
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc, train_epoch_f1 = train(
        model, 
        train_loader, 
        optimizer, 
        criterion,
        device
    )
    valid_epoch_loss, valid_epoch_acc, valid_epoch_f1 = validate(
        model, 
        valid_loader, 
        criterion,
        device
    )
    
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    train_f1.append(train_epoch_f1)
    valid_f1.append(valid_epoch_f1)
    print(f"Training loss: {train_epoch_loss:.3f}, Training acc: {train_epoch_acc:.3f}, Training F1 Score: {train_epoch_f1:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, Validation acc: {valid_epoch_acc:.3f}, Validation F1 Score: {valid_epoch_f1:.3f}")
    
    print('-'*50)
    
torch.save(model.state_dict(), "./ResNetModel.pt")
# Save the loss and accuracy plots.
plt.plot(train_loss)
plt.plot(valid_loss)
plt.show()
plt.plot(train_f1)
plt.plot(valid_f1)
plt.show()
plt.plot(train_acc)
plt.plot(valid_acc)
plt.show()
print('TRAINING COMPLETE')

