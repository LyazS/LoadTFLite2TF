import cv2
import numpy as np
import tensorflow as tf
import time 

testimg=np.load("testnp.npy")
print(testimg.shape)

interp_palm = tf.lite.Interpreter("palm_detection.tflite")
interp_palm.allocate_tensors()

output_details = interp_palm.get_output_details()
input_details = interp_palm.get_input_details()
in_idx = input_details[0]['index']
out_reg_idx = output_details[0]['index']
out_clf_idx = output_details[1]['index']

strattime=time.time()
interp_palm.set_tensor(in_idx, testimg[np.newaxis])
interp_palm.invoke()
endtime=time.time()
print("time : ",endtime-strattime)

out_reg = interp_palm.get_tensor(out_reg_idx)[0]
out_clf = interp_palm.get_tensor(out_clf_idx)[0,:,0]
# print(interp_palm.get_tensor(378))
# print(type(out_clf))

