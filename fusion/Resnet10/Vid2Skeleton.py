import cv2
import mediapipe as mp
import uuid
import os
import numpy as np
from absl.flags import FLAGS
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from torchvision import transforms
from PIL import Image

# set up 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# For static images:
# IMAGE_FILES = []
# with mp_pose.Pose(
#     static_image_mode=True,
#     model_complexity=2,
#     min_detection_confidence=0.5) as pose:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     image_height, image_width, _ = image.shape
#     # Convert the BGR image to RGB before processing.
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if not results.pose_landmarks:
#       continue
#     print(
#         f'Nose coordinates: ('
#         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
#         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
#     )

i =0 # this is used as txt files index

# For webcam input:
# cap = cv2.VideoCapture(1)

# For video input:
vid_name = 'VID_19082021_141315_side'
cap = cv2.VideoCapture(vid_name+".mp4") # open one video in mp4 format to get list of frames

  
# cap.open(0, cv2.CAP_DSHOW)


text = "" # used to store the result and write in the file
# Curl counter variables
counter = None 
stage = None
flag = False
angle = 0
test = 0

counter_right = None 
stage_right = None
flag_right = False
rightside = 0
test_right = 0


with mp_pose.Pose(
  # these two values are default setting already work pretty well
  # higher confidence = higher accuracy of the detection model
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read() #get the webcam open
    if not success:
      # print("Ignoring empty camera frame.")
      # For webcam input:
      # cap = cv2.VideoCapture(FLAGS.video)
      # If loading a video, use 'break' instead of 'continue'.
      break

    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  
    image.flags.writeable = False

    # make detection
    results = pose.process(image)

    
    if results.pose_landmarks:
      for idx,pose_result in enumerate(results.pose_landmarks.landmark):
        # 11-32 is the nodes of body except the face
        if( idx >=11 and idx<=32):
          # print coordinate in console
          # print(idx)
          # print("x coordinate: " ,pose_result.x)
          # print("y coordinate: " ,pose_result.y)
          text = text + str(pose_result.x) + " " + str(pose_result.y) + " "
          
      # store in txt file independently
      hi = os.path.join("./",vid_name+"_txt")
      if not os.path.exists(hi):
          os.mkdir(hi)
      path = os.path.join(hi,hi+str(i+1)) + ".txt"
      f = open (path,"w")
      f.write(text)
      f.close()
      i = int(i) +1
      text = ""   

    # Draw the pose annotation on the image.
    image.flags.writeable = True
  
    
    # recolor the image to BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    # render detections
    # mp_drawing.draw_landmarks(
    #     image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #     mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
    #     mp_drawing.DrawingSpec(color=(0,0,240), thickness=2, circle_radius=4))
    #     # this drawingSpec is to change color and visualizing effect

    img = cv2.imread('./black.jpg')
    annotated_image = img.copy()
    mp_drawing.draw_landmarks(
    img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,255), thickness=5, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0,255,255), thickness=5, circle_radius=4))

    dir = os.path.join("./",vid_name+"_data")
    if not os.path.exists(dir):
        os.mkdir(dir)
    p = transforms.Compose([transforms.Resize((100,100))])
    hi = Image.fromarray(img)
    img = p(hi)
    oput = np.array(img)
    cv2.imwrite(os.path.join(dir, vid_name)+ "_"+str(i) +'.jpg', oput)
    
    # print("os.path: ", os.path)    
    # cv2.imwrite(os.path.join('Output', '{}.jpg'.format(uuid.uuid1())), image)
    
    # cv2.imshow('MediaPipe Pose', image)
  
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break         
cap.release() 



# cv2.destroyAllWindows()
