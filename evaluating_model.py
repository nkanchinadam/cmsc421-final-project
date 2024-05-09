import torch 
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

IMAGE_SIZE = 224
test_path = './PathAndClassTest.csv'
data_transform_test = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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

model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
names = ['Crime', 'Action', 'Romance', 'Comedy', 'Drama']
model.fc = nn.Linear(512, 5)
model.load_state_dict(torch.load('./ResNetModel.pt'))

dataset = CustomDataSet(csv_file=test_path, transform=data_transform_test)

test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

# true positive - false positive - true negative - false negative
metric = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
counter = 0
for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
  counter+=1
  image, labels = data 
  labels = [[float(val) for val in label[1:-1].split(', ')] for label in labels][0]
  outputs = model(image)
  outputs = [0.0 if data < 0 else 1.0 for data in outputs.data[0, :]]

  for ind in range(len(outputs)):
    if outputs[ind] == labels[ind]:
      if outputs[ind] == 1:
        metric[ind][0] += 1
      else:
        metric[ind][2] += 1
    elif outputs[ind] == 1:
      metric[ind][1] += 1
    else:
      metric[ind][3] += 1
  if counter == 500:
    break
    
print(metric)
for ind in range(len(metric)):
  metrics = metric[ind]
  true = [1]*metrics[0] + [0]*metrics[1] + [0]*metrics[2] + [1]*metrics[3]
  pred = [1]*metrics[0] + [1]*metrics[1] + [0]*metrics[2] + [0]*metrics[3]
  matrix = confusion_matrix(y_true=true, y_pred=pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
  disp.plot()
  plt.title(names[ind])
  plt.show()


