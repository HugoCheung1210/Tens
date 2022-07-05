import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# from Resnet10 import train_resnet, res10_model
# from LSTM import train_lstm, BILSTM
from res10_model import ResNet10, ResBlock
from BILSTM import BiLSTM
from CBAMv2 import CBAM_CNN

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.BILSTM = BiLSTM(input_dim = 44,hidden_dim = 128,batch_size = 16)
        self.resnet = ResNet10(ResBlock)
        self.CBAM_CNN = CBAM_CNN(6)      
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(13568, 768)
        self.fc2 = nn.Linear(768,512)
        self.fc_out = nn.Linear(512, 3)
        
    def forward(self, x1, x2, x3):
        
        x1 = self.BILSTM(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.drop(x1)
        
        x2 = self.resnet(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.drop(x2)
        
        x3 = self.CBAM_CNN(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.drop(x3)

        
        # Concatenate in dim1 (feature dimension)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fc1(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.fc_out(x)
        
        return x

# model = MyModel(ResBlock)
# x1 = torch.randn(16, 20, 50)
# x2 = torch.randn(16, 20, 100, 100)
# x3 = torch.randn(16, 16, 28)
# output = model(x1, x2, x3)
# summary(output)