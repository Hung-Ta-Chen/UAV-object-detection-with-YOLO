#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import cv2
import numpy as np
import secrets


# In[2]:


def brightness(img, low, high):
    value = secrets.SystemRandom().uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img
def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img
def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img
def rotation(img, angle):
    angle = int(secrets.SystemRandom().uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img
#img = rotation(img, 30)    
#img = vertical_flip(img, True)
#img = horizontal_flip(img, True)
#img = brightness(img, 0.5, 1.5)


# In[20]:


# k=0
# for path in glob.glob("./string_450/*.jpg"):
#     img = cv2.imread(path)
#     #print (os.path.split(path)[1])
    
#     #rotation
#     img_r = rotation(img, 30)    
#     cv2.imwrite('./string/string_'+str(k)+'.jpg',img_r)
#     k+=1
    
#     #vertical flip
#     img_r = vertical_flip(img, True)   
#     cv2.imwrite('./string/string_'+str(k)+'.jpg',img_r)
#     k+=1
    
#     #horizontal flip
#     img_r = horizontal_flip(img, True)
#     cv2.imwrite('./string/string_'+str(k)+'.jpg',img_r)
#     k+=1
    
#     #brightness
#     img_r = brightness(img, 0.5, 1.5)   
#     cv2.imwrite('./string/string_'+str(k)+'.jpg',img_r)
#     k+=1
    

