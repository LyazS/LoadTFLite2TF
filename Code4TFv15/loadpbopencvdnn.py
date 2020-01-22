import cv2
from cv2 import dnn
import numpy as np 
import time

pbfile1="palm_detection.pb"
pbfile2="hand_landmark.pb"
# pbtxtfile="hand_landmark.pbtxt"
net1=dnn.readNetFromTensorflow(pbfile1,)
net2=dnn.readNetFromTensorflow(pbfile2,)
# net.setPreferableBackend(dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(dnn.DNN_TARGET_OPENCL)
im=np.load("testnp.npy")

while True:
    starttime=time.time()
    # net1.setInput(dnn.blobFromImage(im,size=(256,256)))
    net2.setInput(dnn.blobFromImage(im,size=(256,256)))
    # out1=net1.forward(["concat","concat_1"])
    out2=net2.forward(["Reshape","Reshape_1"])

    endtime=time.time()
    print(endtime-starttime)
print(out1[0].shape,out1[1].shape)
print(out2[0].shape,out2[1].shape)

