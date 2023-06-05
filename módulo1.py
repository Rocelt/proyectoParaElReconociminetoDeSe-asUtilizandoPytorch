
import mediapipe as mp
import cv2
import numpy as np
from torchvision.io import read_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os

device = torch.device('cpu')
print(device)

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

#----------------------------- Creamos la carpeta donde almacenaremos el entrenamiento ---------------------------------
'''nombre = 'LetraB'
direccion = './Validacion'
carpeta = direccion + '/' + nombre
if not os.path.exists(carpeta):
    print('Carpeta creada: ',carpeta)   
    os.makedirs(carpeta)'''

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

PATH ='.\modelo.pt'

modelo = NNConvolucional(3,16,32)

modelo.load_state_dict(torch.load(PATH, map_location=device))
modelo.eval()

tensor = transforms.ToTensor()


with mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.5 ) as hands:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        dedos_reg = frame.copy()
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        #print("Hand landmarks", results.multi_hand_landmarks)

        x_coords = []
        y_coords = []

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for idx, landmark in enumerate(hand_landmarks.landmark):
                        x_coords.append(int(landmark.x * width))
                        y_coords.append(int(landmark.y * height))
            x_max = max(x_coords)+30
            y_max = max(y_coords)+30
            x_min = min(x_coords)-30
            y_min = min(y_coords)-30
            cv2.rectangle(frame,(x_min,y_max),(x_max,y_min),(0,255,0),2)
            

            dedos_reg = frame[y_min:y_max, x_min:x_max]
            dedos_reg = cv2.resize(dedos_reg,(200,200), interpolation = cv2.INTER_CUBIC)

            img_tensor = tensor(dedos_reg)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device=device)

            with torch.no_grad():
                output = modelo(img_tensor)
                print(output)
            _, predicted = torch.max(output, 1)

            print(predicted.item())

            if predicted.item():
                cv2.putText(frame,"Letra B",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
            else:
                cv2.putText(frame,"Letra A",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)

            x_max = []
            y_max = []
            x_min = []
            y_min = []
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0XFF ==27:
            break
cap.release()
cv2.destroyAllWindows