import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import ConcatDataset,DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from TextDataset_CDAM import TextFolder_CDAM
from TextDataset_LSTM import TextFolder_LSTM

from fusion_model import MyModel



# Resnet10 dataloader
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.ToTensor()
    ]),
}
img_data_dir = './Resnet10/dataset'
img_train_dir = './train_processed'

test_dir = './test_processed'

image_datasets = {x: datasets.ImageFolder(os.path.join(img_data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

res_dataloaders = {x: DataLoader(image_datasets[x],batch_size= 8,
                                             shuffle=True, num_workers=0, drop_last = True)
              for x in ['train', 'test']}

res_class = image_datasets['train'].classes


# LSTM dataloader
data_dir = './LSTM/data'

text_datasets = {x: TextFolder_LSTM(os.path.join(data_dir, x))
                 for x in ['train', 'test']}
lstm_dataloaders = {x: DataLoader(text_datasets[x], batch_size=8,
                                             shuffle= True, num_workers=0,  drop_last = True)
              for x in ['train', 'test']}
text_dataset_sizes = {x: len(text_datasets[x]) for x in ['train', 'test']}
    
lstm_class = text_datasets['test'].classes

# CDAM dataloader
motion_dir = "./CDAM/data"

motion_dataset = {x: TextFolder_CDAM(os.path.join(motion_dir, x))
                 for x in ['train', 'test']}
cdam_dataloaders = {x: DataLoader(motion_dataset[x], batch_size=8,
                                             shuffle= True, num_workers=0,drop_last=True)
              for x in ['train', 'test']}
cdam_dataset_sizes = {x: len(motion_dataset[x]) for x in ['train', 'test']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.load_state_dict(torch.load("./trained_fusion.pt"))
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # if phase == 'train': dataloader = trainloader
            # else: dataloader = testloader
            res_dataloader_iterator = iter(res_dataloaders[phase])
            cdam_dataloader_iterator = iter(cdam_dataloaders[phase])
            for i, (input_lstm, label_lstm) in enumerate(lstm_dataloaders[phase]):
                try:
                   input_res, label_res = next(res_dataloader_iterator)
                   input_cdam, label_cdam = next(cdam_dataloader_iterator)
                except StopIteration:
                    res_dataloader_iterator = iter(res_dataloaders[phase])
                    cdam_dataloader_iterator = iter(cdam_dataloaders[phase])
                    input_res, label_res = next(res_dataloader_iterator)
                    input_cdam, label_cdam = next(cdam_dataloader_iterator)




                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    input_cdam = torch.unsqueeze(input_cdam,2)
                    outputs = model(input_lstm,input_res,input_cdam)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, label_lstm)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # running_loss += loss.item() * inputs.size(0)
                
                running_loss += loss.item()
                running_corrects += torch.sum(preds == label_lstm.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    # torch.save(best_model_wts, "./trained_fusion.pt")

    model.load_state_dict(best_model_wts)
    
    return model


if __name__ == '__main__': 
    model = MyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_conv = train_model(model, criterion, optimizer_ft,
                            exp_lr_scheduler, num_epochs=5)
    
            