import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchinfo import summary
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image

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
    nn.Linear(128, 5)
)

summary(model, input_size=(10, 3, 224, 224))
summary(model.fc)
model.fc = our_layers
summary(our_layers)
summary(model, input_size=(10, 3, 224, 224))


