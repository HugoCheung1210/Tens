import torch.nn as nn
import torch
from torchinfo import summary



class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  

    def forward(self, x):
        
        x = self.Cam(x)
        x = self.Sam(x)

        return x


class ChannelAttentionModul(nn.Module):  
    def __init__(self, in_channel, r=0.5): 
        super(ChannelAttentionModul, self).__init__()
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(1, int(in_channel * r)),  
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(1, int(in_channel * r)),  
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        max_branch = self.MaxPool(x)
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        avg_branch = self.AvgPool(x)
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        h, w = weight.shape
   
        Mc = torch.reshape(weight, (h, w, 1))

        x = Mc * x

        return x


class SpatialAttentionModul(nn.Module): 
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv1d(in_channel, 1, 7, padding=3)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        MaxPool = torch.max(x, dim=2).values  
        AvgPool = torch.mean(x, dim=2)


        MaxPool = torch.unsqueeze(MaxPool, dim=2)
        AvgPool = torch.unsqueeze(AvgPool, dim=2)

        
        x_cat = torch.cat((MaxPool, AvgPool), dim=2)  

    
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)
        x = Ms * x

        return x







class model(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.layer0 = nn.Conv1d(in_channel, 64, kernel_size=1)
        self.layer1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2 = nn.ReLU(inplace=True)
        self.CBAM1 = CBAM(64)
        self.layer3 = nn.Conv1d(64, 128, kernel_size=1)
        self.layer4 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer5 = nn.ReLU(inplace=True)
        self.CBAM2 = CBAM(128)
        self.f = nn.Flatten()
        self.fc = nn.Linear(256, 256)
        self.drop = nn.Dropout()
        self.output= nn.Linear(256,3)
    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.drop(input)
        input = self.CBAM1(input)

        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)
        input = self.drop(input)

        input = self.CBAM2(input)
        input = self.f(input)
        input = self.fc(input)
        input = self.output(input)
        return input
