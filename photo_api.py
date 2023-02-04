from flask import Flask,request
from flask import jsonify
from asgiref.wsgi import WsgiToAsgi
from ultralytics import YOLO
from os import environ
from base64 import b64encode
import cv2
import numpy as np
import requests
###### GLOBAL FUNCTION #############
CLIP_DROP_API_KEY = environ['CLIP_DROP']
MODEL = YOLO(environ['YOLO_MODEL'])
BGD_MODEL = np.zeros((1,65),np.float64)
FGD_MODEL = np.zeros((1,65),np.float64)
APP = Flask(__name__)
ASGI_APP = WsgiToAsgi(APP)
######## Image Segmentation ##############
async def detectAndExtract(binary_base_image):
    image = byteToImageArray(binary_base_image)
    img_height,img_width,_ = image.shape
    human_mark = []
    real_mark = []
    human_img = []
    blank = np.zeros(image.shape,np.uint8)
    res = MODEL.predict(byteToImageArray(binary_base_image),stream=True)
    # Get Prediction Result
    for obj in res:
        temp = obj.boxes.xyxy
        temp2 = obj.masks.segments
        for i,human_checked in enumerate(obj.numpy().boxes.cls):
            if int(human_checked) == 0 :
                human_mark.append(temp[i].numpy().astype(int))
                real_mark.append(temp2[i])
    # Prepare Human Mask
    for each_mark in real_mark:
        for j in range(0,len(each_mark)):
            each_mark[j][0] *= img_width
            each_mark[j][1] *= img_height
        contours = np.array(each_mark.astype(int))
        cv2.fillPoly(blank,pts=[contours],color=(255,255,255))
    # Extract Human Mask
    for obj in human_mark:
        cropped = image[obj[1]:obj[3],obj[0]:obj[2]]
        cropped_mask = blank[obj[1]:obj[3],obj[0]:obj[2]]
        final_cut = makeMask(cropped,cropped_mask)
        final_cut = makeBlackTransparent(final_cut)
        human_img.append(imageArrayToBase64String(final_cut))
    return blank,human_img
######## Background Removal ##############
async def cleanUpOriginalImage(binary_base_image,human_mask_all):
    #return imageArrayToBase64String(human_mask_all)
    image = byteToImageArray(binary_base_image)
    image = cv2.imencode('.jpg',image)[1].tobytes()
    blank = cv2.imencode('.png',human_mask_all)[1].tobytes()
    r = requests.post('https://clipdrop-api.co/cleanup/v1',
    files = {
        'image_file': ('image.jpg', image, 'image/jpeg'),
        'mask_file': ('mask.png', blank, 'image/png')
        },
    headers = { 'x-api-key': CLIP_DROP_API_KEY}
    )
    if (r.ok):
        return imageArrayToBase64String(byteToImageArray(r.content),'.jpg')
    else:
        r.raise_for_status()
######## Byte Data To NP Array Helper ############
def byteToImageArray(byte_data):
    buff = np.frombuffer(byte_data,dtype=np.uint8)
    buff = cv2.imdecode(buff,flags=1)
    return buff
######## NP Array To Byte Data Helper ############
def imageArrayToByte(image_array,file_type):
    return cv2.imencode(file_type,image_array)[1].tobytes()
def imageArrayToBase64String(image_array,file_type='.png'):
    prefix = 'data:image/png;base64,' if file_type == '.png' else 'data:image/jpeg;base64,'
    return prefix+b64encode(cv2.imencode(file_type,image_array)[1]).decode()
######## Image Crop with mask Helper ############
def makeMask(image,mask,alpha=255):
    tmp = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    tmp = np.where(tmp == alpha ,1,0).astype('uint8')
    return cv2.bitwise_and(image,image,mask=tmp)
######## Transparent BG Helper ##############
def makeBlackTransparent(image):
    tmp = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b,g,r = cv2.split(image)
    rgba = [b,g,r,alpha]
    return cv2.merge(rgba,4)
@APP.route("/",methods=['GET','POST'])
async def photo_api():
    if request.method == 'GET' :
        return 'OK'
    elif request.method == 'POST' :
        image_file = request.files['original'].read()
        human_mask_all,human_pose = await detectAndExtract(image_file)
        image_no_human = await cleanUpOriginalImage(image_file,human_mask_all)
        return jsonify({
            'base':image_no_human,
            'human_pose':human_pose
        })
    else :
        return "Not Found",404
