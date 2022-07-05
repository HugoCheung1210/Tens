import torch
from torchvision import transforms
import os
import cv2
from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip

p = transforms.Compose([transforms.Resize((100,100))])
folder = 'VID_19082021_145939_data'
i = 1
clip = VideoFileClip("./20210719_120125/20210719_120125.mp4")
vid_len = np.floor(clip.duration)
ct = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
for fname in os.listdir(folder):
    if fname == '.DS_Store' : ct -= 1
    if fname == 'txt' : ct -= 1
FPS = np.floor(ct/vid_len)
stk = []

# Resize and output as binary file
for fname in os.listdir(folder):
    # Sometime there is some DS_Store file to affect the reading of data
    if fname == '.DS_Store' : continue
    if fname == 'txt' : continue   
     
    # img = cv2.imread(os.path.join(folder,fname))
    img = Image.open(os.path.join(folder,fname))
    re_img = p(img)
    # Turn the plt file to numpy array to output it again
    oput = np.array(re_img)
    # stk.append(oput)
    
    # if len(stk)%FPS == 0:
    #     output = np.stack(stk)
    #     stk = []
    #     with open('data.bin', 'wb') as f:
    #         arr=output.tobytes()
    #         f.write(arr)
    #         f.close()
       
       
    cv2.imwrite('./train_processed/raise_arm/image' + str(i) + '.jpg', oput)
        
    i +=1
            