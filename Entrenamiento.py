import os
from pyexpat import model
from turtle import forward
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torchvision.io import read_image
import matplotlib.pyplot as plt
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cpu')
print(device)

TRAIN_PATH = '.\Validacion\Prueba'
TRAIN_LABELS = '.\Validacion\datos.csv'

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path+".jpg")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class NNConvolucional(nn.Module):
    def __init__(self, entradas, capa1, capa2):
        super().__init__(),
        self.conv1 = nn.Conv2d(in_channels = entradas,out_channels = capa1,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=capa1, out_channels=capa2,
                               kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(in_features=100*100*16, out_features=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu( self.conv1(x))
        X = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x

def accuracy(model, loader):
    num_correct = 0
    num_total = 0
    model.eval()
    model = model.to(device = device)
    with torch.no_grad():
        for xi, yi in loader:
            xi = xi.to(device = device, dtype = torch.float32)
            yi = yi.to(device = device, dtype = torch.long)
            scores = model(xi)
            _, pred = scores.max(dim=1)
            num_correct += (pred == yi).sum()
            num_total += pred.size(0)
        return float(num_correct)/num_total


def train(model, optimiser, epochs):
    cost_history = []
    acc_history = []
    model = model.to(device = device)
    for epoch in range(epochs):
        for i, (xi, yi) in enumerate(train_loder):
            model.train()
            xi = xi.to(device = device, dtype = torch.float32)
            yi = yi.to(device = device, dtype = torch.long)
            print(xi.shape)
            prediccion = model(xi)
            cost = F.cross_entropy(input=prediccion, target=yi)
            optimiser.zero_grad()
            cost.backward()
            optimiser.step()
        acc = accuracy(model, val_loder)
        print(f'Epoch_ {epoch}, costo: {cost.item()}, accuracy: {acc}')
        cost_history.append(cost)
        acc_history.append(acc)
    '''plt.figure()
    plt.plot(epochs, cost_history, label='Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Gráfica de la precisión
    plt.figure()
    plt.plot(epochs, acc_history, label='Training accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()'''

datasetCompleto = CustomImageDataset(TRAIN_LABELS,TRAIN_PATH,transform=None, target_transform=None)


BATCH_SIZE = 20
TRAIN_SIZE = int(datasetCompleto.__len__() * 0.8)
VAL_SIZE = datasetCompleto.__len__() - TRAIN_SIZE

train_dataset, val_dataset = random_split(datasetCompleto, [TRAIN_SIZE,VAL_SIZE])

train_loder = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True)
val_loder = DataLoader(val_dataset,batch_size=BATCH_SIZE, shuffle=True)


numFiltrosCan1 = 16
numFiltrosCan2 = 32
epochs = 50
lr = 0.0001
modelCNN3 = NNConvolucional(3, numFiltrosCan1, numFiltrosCan2)
optimiser = torch.optim.Adam(modelCNN3.parameters(), lr)
train(modelCNN3,optimiser,epochs)

PATH ='.\modelo1.pt'
torch.save(modelCNN3.state_dict(),PATH)


'''def plot_mini_bach(train_features, train_labels):
    plt.figure(figsize=(10,10))
    for i in range(BATCH_SIZE):
        plt.subplot(5,4,i+1)
        img = train_features[i,...].permute(1,2,0).numpy()

        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
plot_mini_bach(train_features=train_features,train_labels=train_dataset)'''