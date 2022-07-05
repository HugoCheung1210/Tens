import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz, resample
import scipy
import os



x1 = []
x2 = []

y1 = []
y2 = []

z1 = []
z2 = []

class motion:
  def __init__(self):
    self.x1 = []
    self.x2 = []
    self.y1 =[]
    self.y2 =[]
    self.z1 = []
    self.z2 = []

file = 'TENS_Data_19082021_154944'
with open(file+'.txt') as fp:
    for line in fp:
        line = line.split(" ")
        line1 = line[1:4]
        line2 = line[14:17]
        line_m = line1+line2
        cor = np.empty(6)
        for item, value in enumerate(line_m):
            value = float(value)
            if type(value) is not float: continue
            cor[item] = value
        x1.append(cor[0])
        x2.append(cor[3])
        y1.append(cor[1])
        y2.append(cor[4])
        z1.append(cor[2])  
        z2.append(cor[5])  
    
#  5845
#  5984,11705
#  11816


# 6177
# 6168,11759
# 12127

# 2397,8362
# 8587,14201
# 14366,20090

# 6001
# 6089,11856
# 12022,17682

front = motion()
front2side = motion()
side = motion()
for i in range(6054):
    front.x1.append(x1[i])
    front.x2.append(x2[i])

    front.y1.append(y1[i])
    front.y2.append(y2[i])
    
    front.z1.append(z1[i])
    front.z2.append(z2[i])

for i in range(6282,11944):
    front2side.x1.append(x1[i])
    front2side.x2.append(x2[i])

    front2side.y1.append(y1[i])
    front2side.y2.append(y2[i])

    front2side.z1.append(z1[i]) 
    front2side.z2.append(z2[i]) 

for i in range(12247, 17766):
    side.x1.append(x1[i])
    side.x2.append(x2[i])

    side.y1.append(y1[i])
    side.y2.append(y2[i])

    side.z1.append(z1[i])
    side.z2.append(z2[i])

# front.x1 = np.array(front.x1)
fx1 = scipy.signal.resample(front.x1,int(len(front.x1)/10))
fx2 = scipy.signal.resample(front.x2,int(len(front.x2)/10))
fy1 = scipy.signal.resample(front.y1,int(len(front.y1)/10))
fy2 = scipy.signal.resample(front.y2,int(len(front.y2)/10))
fz1 = scipy.signal.resample(front.z1,int(len(front.z1)/10))
fz2 = scipy.signal.resample(front.z2,int(len(front.z2)/10))
f2sx1 = scipy.signal.resample(front2side.x1,int(len(front2side.x1)/10))
f2sx2 = scipy.signal.resample(front2side.x2,int(len(front2side.x2)/10))
f2sy1 = scipy.signal.resample(front2side.y1,int(len(front2side.y1)/10))
f2sy2 = scipy.signal.resample(front2side.y2,int(len(front2side.y2)/10))
f2sz1 = scipy.signal.resample(front2side.z1,int(len(front2side.z1)/10))
f2sz2 = scipy.signal.resample(front2side.z2,int(len(front2side.z2)/10))
sx1 = scipy.signal.resample(side.x1,int(len(side.x1)/10))
sx2 = scipy.signal.resample(side.x2,int(len(side.x2)/10))
sy1 = scipy.signal.resample(side.y1,int(len(side.y1)/10))
sy2 = scipy.signal.resample(side.y2,int(len(side.y2)/10))
sz1 = scipy.signal.resample(side.z1,int(len(side.z1)/10))
sz2 = scipy.signal.resample(side.z2,int(len(side.z2)/10))



b, a = scipy.signal.butter(2, .8, 'lowpass',fs = 10)
f_x1 = scipy.signal.lfilter(b, a, fx1)
f_x2 = scipy.signal.lfilter(b, a, fx2)
f_y1 = scipy.signal.lfilter(b, a, fy1)
f_y2 = scipy.signal.lfilter(b, a, fy2)
f_z1 = scipy.signal.lfilter(b, a, fz1)
f_z2 = scipy.signal.lfilter(b, a, fz2)

f2s_x1 = scipy.signal.lfilter(b, a, f2sx1)
f2s_x2 = scipy.signal.lfilter(b, a, f2sx2)
f2s_y1 = scipy.signal.lfilter(b, a, f2sy1)
f2s_y2 = scipy.signal.lfilter(b, a, f2sy2)
f2s_z1 = scipy.signal.lfilter(b, a, f2sz1)
f2s_z2 = scipy.signal.lfilter(b, a, f2sz2)

s_x1 = scipy.signal.lfilter(b, a, sx1)
s_x2 = scipy.signal.lfilter(b, a, sx2)
s_y1 = scipy.signal.lfilter(b, a, sy1)
s_y2 = scipy.signal.lfilter(b, a, sy2)
s_z1 = scipy.signal.lfilter(b, a, sz1)
s_z2 = scipy.signal.lfilter(b, a, sz2)

# Front motion
# Split the data in accord to len of video data 
hi = os.path.join("./front")
if not os.path.exists(hi):
          os.mkdir(hi)

# Length of video data
# Will be optimized to 1 sec window later   
# dataf_len = 1746

# f_len = int(len(fx1)/dataf_len)
    
for j in range (len(f_x1)):  
    path_front = os.path.join(hi,hi+str(j+1)) + file + ".txt"
    f = open (path_front,"w")      
    text1 = str(f_x1[j])+" "+str(f_y1[j])+" "+ str(f_z1[j])+" "
    text1 = text1.rstrip('\n')
    text2 = str(f_x2[j])+" "+str(f_y2[j])+" "+ str(f_z2[j])+" "
    text = text1+text2
    text = text.rstrip('\n')
    f.write(text)
    f.close()

# Front to side motion
# Split the data in accord to len of video data 
hi = os.path.join("./front2side")
if not os.path.exists(hi):
          os.mkdir(hi)

# Length of video data
# Will be optimized to 1 sec window later   
# dataf2side_len = 1769

# f2s_len = int(len(f2sx1)/dataf2side_len)

for j in range (len(f2s_x1)):
    path_f2side = os.path.join(hi,hi+str(j+1)) + file + ".txt"
    f = open (path_f2side,"w")        
    text1 = str(f2s_x1[j])+" "+str(f2s_y1[j])+" "+ str(f2s_z1[j])+" "
    text1 = text1.rstrip('\n')
    text2 = str(f2s_x2[j])+" "+str(f2s_y2[j])+" "+ str(f2s_z2[j])+" "
    text = text1 + text2
    text = text.rstrip('\n')
    f.write(text)
    f.close()


# Side motion
# Split the data in accord to len of video data 
hi = os.path.join("./side")
if not os.path.exists(hi):
          os.mkdir(hi)

# Length of video data
# Will be optimized to 1 sec window later   
# data_s_len = 1747

# s_len = int(len(sx1)/data_s_len)
# for i in range(data_s_len):
    
for j in range (len(s_x1)):
    path_side = os.path.join(hi,hi+str(j+1))+ file  + ".txt"
    f = open (path_side,"w")        
    text1 = str(s_x1[j])+" "+str(s_y1[j])+" "+ str(s_z1[j])+" "
    text1 = text1.rstrip('\n')
    text2 = str(s_x2[j])+" "+str(s_y2[j])+" "+ str(s_z2[j])+" "
    text = text1 + text2
    text = text.rstrip('\n')
    f.write(text)
    f.close()






# if __name__ == "__main__":
#     fig, ax = plt.subplots()
#     # ax.plot(front.x1,label ="x axis")
#     ax.plot(x1,label ="filter_x axis")
#     # ax.plot(f2sx1,label ="filter_x axis")

#     # ax.plot(front2side.x1,label ="filter_y axis")
#     # ax.plot(front2side.x2,label ="y axis")
#     # ax.plot(front.z,label ="y axis")
#     # ax.plot(z,label ="filter_z axis")

#     ax.set_xlabel('DATA_NUM')
#     ax.set_ylabel('sensor')
#     ax.set_title('Motion ')
    
#     plt.show()
        
    
    # fig.savefig('f.jpg')