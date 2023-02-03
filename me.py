from ultralytics import YOLO
import cv2
import numpy as np
import requests
from base64 import b64decode
from os import environ
IMAGE_FILE = "./6.jpg"
CLIP_DROP_API_KEY = environ['CLIP_DROP']
MODEL = YOLO("yolov8l-seg.pt")
BGD_MODEL = np.zeros((1,65),np.float64)
FGD_MODEL = np.zeros((1,65),np.float64)
cv_image = cv2.imread(IMAGE_FILE)
cv_height,cv_width,_ = cv_image.shape
res = MODEL.predict(IMAGE_FILE)
human_mark = []
real_mark = []
for obj in res:
    temp = obj.boxes.xyxy
    temp2 = obj.masks.segments
    for i,human_checked in enumerate(obj.numpy().boxes.cls):
        if int(human_checked) == 0 :
            human_mark.append(temp[i].numpy().astype(int))
            real_mark.append(temp2[i])
blank = np.zeros(cv_image.shape,np.uint8)
for each_mark in real_mark:
    for j in range(0,len(each_mark)):
        each_mark[j][0] *= cv_width
        each_mark[j][1] *= cv_height
    contours = np.array(each_mark.astype(int))
    cv2.fillPoly(blank,pts=[contours],color=(255,255,255))

all_mask = cv2.cvtColor(blank,cv2.COLOR_RGB2GRAY)
all_mask = np.where(all_mask == 255 ,cv2.GC_BGD,cv2.GC_FGD).astype('uint8')
final_full_mask,_,_ = cv2.grabCut(cv_image,all_mask,None,BGD_MODEL,FGD_MODEL,3,cv2.GC_INIT_WITH_MASK)
final_full_mask = np.where((final_full_mask==2)|(final_full_mask==0),0,1).astype('uint8')
final_full_img = cv_image*final_full_mask[:,:,np.newaxis]
cv2.imwrite("mask.png",blank)
for i,obj in enumerate(human_mark):
    cropped = cv_image[obj[1]:obj[3],obj[0]:obj[2]]
    cropped_mask = blank[obj[1]:obj[3],obj[0]:obj[2]]
    cropped_mask = cv2.cvtColor(cropped_mask,cv2.COLOR_RGB2GRAY)
    cropped_mask = np.where(cropped_mask == 255,cv2.GC_FGD,cv2.GC_BGD).astype('uint8')
    final_mask,_,_ = cv2.grabCut(cropped,cropped_mask,None,BGD_MODEL,FGD_MODEL,3,cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((final_mask==2)|(final_mask==0),0,1).astype('uint8')
    final_img = cropped*final_mask[:,:,np.newaxis]
    tmp = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(final_img)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    #print(dst)
    # cv2.imshow("EXP "+str(i),dst)
    cv2.imwrite("EXP"+str(i)+".png",dst)

print("Finished Extraction...")
"""
img1 = cv2.imencode('.jpg',cv_image)[1].tobytes()
img2 = cv2.imencode('.png',blank)[1].tobytes()
r = requests.post('https://clipdrop-api.co/cleanup/v1',
  files = {
    'image_file': ('image.jpg', img1, 'image/jpeg'),
    'mask_file': ('mask.png', img2, 'image/png')
    },
  headers = { 'x-api-key': CLIP_DROP_API_KEY}
)

if (r.ok):
    buff = np.frombuffer(r.content,dtype=np.uint8)
    buff = cv2.imdecode(buff,flags=1)
    cv2.imwrite("KEK.jpg",buff)
  # r.content contains the bytes of the returned image
else:
  r.raise_for_status()
"""