import torch
from torch import nn
from torch.utils.data import DataLoader
from TextDataset import TextFolder
import os
import torch.optim as optim
from CBAMv2 import CBAM
import time
import copy
from torch.optim import lr_scheduler


data_dir = './data'

text_dataset = {x: TextFolder(os.path.join(data_dir, x))
                 for x in ['train', 'test']}
dataloaders = {x: DataLoader(text_dataset[x], batch_size=16,
                                             shuffle= True, num_workers=0,drop_last=True)
              for x in ['train', 'test']}
dataset_sizes = {x: len(text_dataset[x]) for x in ['train', 'test']}
    
class_names = text_dataset['test'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, n_epochs):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
                
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device)

                labels = labels.to(device)
               
                model.zero_grad()
                    
                with torch.set_grad_enabled(phase == 'train'):
                   
                    inputs = torch.unsqueeze(inputs,2)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_function(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
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
    
    
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__': 
    model = CBAM(6)
    model_cbam = train(model, 20)