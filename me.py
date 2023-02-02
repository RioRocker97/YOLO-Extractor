from ultralytics import YOLO
import cv2
import numpy as np

IMAGE_FILE = "./2.jpg"

model = YOLO("yolov8l-seg.pt")
cv_image = cv2.imread(IMAGE_FILE)
cv_height,cv_width,_ = cv_image.shape
res = model.predict(IMAGE_FILE,show=True)
human_mark = []
real_mark = []
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
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


for i,obj in enumerate(human_mark):
    cropped = cv_image[obj[1]:obj[3],obj[0]:obj[2]]
    cropped_mask = blank[obj[1]:obj[3],obj[0]:obj[2]]
    cropped_mask = cv2.cvtColor(cropped_mask,cv2.COLOR_RGB2GRAY)
    cropped_mask = np.where(cropped_mask == 255,cv2.GC_FGD,cv2.GC_BGD).astype('uint8')
    final_mask,_,_ = cv2.grabCut(cropped,cropped_mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

    final_mask = np.where((final_mask==2)|(final_mask==0),0,1).astype('uint8')
    final_img = cropped*final_mask[:,:,np.newaxis]
    cv2.imshow("EXP "+str(i),final_img)
    cv2.imwrite("EXP"+str(i)+".png",final_img)


cv2.waitKey(0)
cv2.destroyAllWindows()