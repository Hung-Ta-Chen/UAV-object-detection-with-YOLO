#python3 detect_uav.py --model_def config/yolov3-custom.cfg --weights_path checkpoints/yolov3_ckpt_200.pth --class_path data/custom/classes.names
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *

import os
import sys
import time
import datetime
import argparse
import pyscreenshot

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import matplotlib; matplotlib.use('agg')    
import cv2

import threading
import socket

# cap=cv2.VideoCapture(0)
'''
sock.sendto("command".encode(encoding="utf-8"), tello_address)
time.sleep(0.5)
sock.sendto("streamon".encode(encoding="utf-8"), tello_address)
time.sleep(0.5)
sock.sendto("battery?".encode(encoding="utf-8"), tello_address)
time.sleep(0.5)'''
cap=cv2.VideoCapture("udp://192.168.10.1:11111")	


#get frame in background
current_frame=None 
class cam_thread(threading.Thread):
    def __init__(self):	
        global cap
        threading.Thread.__init__(self)
    def run(self):			
        global cap			
        global current_frame			
        global current_ret			
        while True:			
            current_ret, current_frame=cap.read()			
            time.sleep(0.03)		

cam_t=cam_thread()	
cam_t.start()	

host = ''			
recv_data=""			
port = 9000			
locaddr = (host,port)		
			
# Create a UDP socket			
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)			
tello_address = ('192.168.10.1', 8889)			
sock.bind(locaddr)			
			

#hold UAV per 10 seconds
class stop_thread(threading.Thread):			
    def __init__(self):			
        #print("stop thread init")			
        threading.Thread.__init__(self)			
			
                    			
    def run(self):			
        global tello_address			
        while True:			
            print("send stop")			
            sock.sendto("stop".encode(encoding="utf-8"), tello_address)			
            time.sleep(10)			
stop_t=stop_thread()			
			
def recv():						
    global recv_data			
    while True:			
        try:			
            data, server = sock.recvfrom(1518)			
            recv_data=data.decode(encoding="utf-8")			
            print("{}: {}".format(server, recv_data))			
            			
        except Exception:			
            print ('\nExit . . .\n')			
            break			

#todo: connect to UAV and open video stream
#(you can use lab1/example.py to send "streamon" to UAV first)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")


    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))

    model.eval()  

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # Tensor = torch.FloatTensor

    
    fr = 0
    r = 1
    max_x = 960
    max_y = 720	

    target="Virgo"
    scrn = False
    cnt = 0	
    #cap=cv2.VideoCapture('pred2.mp4')			
    sock.sendto("takeoff".encode(encoding="utf-8"), tello_address)
    time.sleep(4)				
    sock.sendto("up 80".encode(encoding="utf-8"), tello_address)
    time.sleep(4)	
    print('up')	
    stop_t.start()

    #cap=cv2.VideoCapture('udp://196.168.10.1:11111')
    #cap=cv2.VideoCapture(0)
    while (cap.isOpened()):
        time.sleep(0.05)
        #fr+=1
        #ret, frame = cap.read()
        frame=current_frame

        if (fr % r == 0):
            input_imgs, _ = transforms.Compose([DEFAULT_TRANSFORMS, Resize(opt.img_size)])((frame,np.zeros((1, 5))))
            input_imgs = Variable(input_imgs.type(Tensor)).unsqueeze(0)

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            # Create plot
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            
            print ('-----------------------------------------')

            if detections is not None:
                sock.sendto("stop".encode(encoding="utf-8"), tello_address)
                # Rescale boxes to original image

                detections = rescale_boxes(detections[0], opt.img_size, img.shape[:2])

                for x1, y1, x2, y2, conf, cls_pred in detections:
                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = (0.2235294117647059, 0.23137254901960785, 0.4745098039215686, 1.0)
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )


                    
                    x_center=(x1+x2)/2
                    y_center=(y1+y2)/2
                    print("class: {}, center: ({},{})".format(classes[int(cls_pred)],x_center, y_center))
                    print("box_w: {}, box_y: {}, ratio: {}".format(box_w, box_h, (box_w*box_h/max_x/max_y)))
                    
                    
                    #-----
                    
                    if classes[int(cls_pred)] == target:	
                        #move the uav
                        go_x = 0
                        go_y = 0
                        if x_center+100 < max_x/2:
                            go_x += 20
                            print("left ")
                        elif x_center-100 > max_x/2:
                            go_x -= 20
                            print("right ")
                        if y_center+100 < max_y/2:
                            go_y += 20
                            print("up ")
                        elif y_center-100 > max_y/2:
                            go_y -= 20
                            print("down ")
                        if go_x != 0 and go_y != 0:
                            send_str = "go 0 " + str(go_x) + " " + str(go_y)+ " 50"
                            sock.sendto(send_str.encode(encoding="utf-8"), tello_address)
                            time.sleep(3)
                        elif go_x > 0:
                            sock.sendto("left 20".encode(encoding="utf-8"), tello_address)
                            time.sleep(3)
                        elif go_x < 0:
                            sock.sendto("right 20".encode(encoding="utf-8"), tello_address)
                            time.sleep(3)
                        elif go_y > 0:
                            sock.sendto("up 20".encode(encoding="utf-8"), tello_address)
                            time.sleep(3)
                        elif go_y < 0:
                            sock.sendto("down 20".encode(encoding="utf-8"), tello_address)
                            time.sleep(3)
                        
                        if box_w * box_h < max_x * max_y * 0.045 and x_center+200>max_x/2 and x_center-200<max_x/2 and y_center+150 > max_y/2 and y_center-150 < max_y/2:
                            sock.sendto("speed 70".encode(encoding="utf-8"), tello_address)
                            time.sleep(0.5)	
                            sock.sendto("forward 35".encode(encoding="utf-8"), tello_address)	
                            print("forward 35")	
                            time.sleep(3)
                            if y_center > max_y/2:
                                sock.sendto("down 20".encode(encoding="utf-8"), tello_address)	
                                print("down ")	
                                time.sleep(0.5)
                                sock.sendto("stop".encode(encoding="utf-8"), tello_address)
                        elif box_w * box_h < max_x * max_y * 0.2 and x_center+200>max_x/2 and x_center-200<max_x/2 and y_center+150 > max_y/2 and y_center-150 < max_y/2:
                            sock.sendto("speed 70".encode(encoding="utf-8"), tello_address)
                            time.sleep(0.5)		
                            sock.sendto("forward 20".encode(encoding="utf-8"), tello_address)	
                            print("forward 20")	
                            time.sleep(3)	
                            #sock.sendto("stop".encode(encoding="utf-8"), tello_address)
                            if y_center > max_y/2:
                                sock.sendto("down 20".encode(encoding="utf-8"), tello_address)	
                                print("down ")	
                                time.sleep(0.5)
                                sock.sendto("stop".encode(encoding="utf-8"), tello_address)
                            if box_w * box_h > max_x * max_y * 0.04:
                                time.sleep(1)
                        #if the uav is at the center and close enough, take a screenshot and land
                        elif  x_center+200>max_x/2 and x_center-200<max_x/2 and y_center+150 > max_y/2 and y_center-150 < max_y/2:
                            filename = './screenshots/'+target+'_'+str(cnt)+'.jpg'
                            cv2.imwrite(filename, img)
                            cnt = cnt + 1
                            sock.sendto("land".encode(encoding="utf-8"), tello_address)
                            time.sleep(10)
                            print("land ")


                    #------
                    	
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
                img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                cv2.imshow("plot",img)

                #press 'esc' to land, 'space' to take screenshots
                k = cv2.waitKey(20)
                if k%256 == 27:
                    print('esc')
                    sock.sendto("land".encode(encoding="utf-8"), tello_address)
                    time.sleep(15)
                if k%256 == 32:
                    scrn = True
                if scrn:
                    filename = './screenshots/'+target+'_'+str(cnt)+'.jpg'
                    image = pyscreenshot.grab(bbox=(80, 110, 800, 800))
                    #cv2.imwrite(filename, img)
                    image.save(filename)
                    cnt = cnt + 1
                    scrn = False
            plt.close('all')
                

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

    cap.release()
    cv2.destroyAllWindows()

