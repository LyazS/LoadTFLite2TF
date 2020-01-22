import os
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm as tqdm
import pdb
import time

ptfl1="palm_detection_without_custom_op.tflite"
ptfl2="hand_landmark.tflite"
testnpint=np.load("testnp.npy")[np.newaxis].astype(np.int32)
testnp=np.load("testnp.npy")[np.newaxis]
print(testnp.dtype)

# testlite1=tf.lite.Interpreter(ptfl1)
# testlite1.allocate_tensors()
# output_details1 = testlite1.get_output_details()
# input_details1= testlite1.get_input_details()

testlite2=tf.lite.Interpreter(ptfl2)
testlite2.allocate_tensors()
output_details2 = testlite2.get_output_details()
input_details2= testlite2.get_input_details()

# print(input_details,output_details)

# inidx=input_details[0]['index']
# outidx=output_details[0]['index']

while True:
    starttime=time.time()
    # testlite1.set_tensor(0, testnpint)
    # testlite1.invoke()
    testlite2.set_tensor(0, testnp)
    testlite2.invoke()
    endtime=time.time()
    print(endtime-starttime)

# out_reg = testlite.get_tensor(outidx)
# print(out_reg.shape)