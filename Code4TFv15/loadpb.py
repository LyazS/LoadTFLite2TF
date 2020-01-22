# -*- coding: UTF-8 -*-
from __future__ import division
import os
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm as tqdm
import pdb
import time

testnp="testnp.npy"
print(tf.__version__)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    with open("hand_landmark.pb", "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    inputs = sess.graph.get_tensor_by_name("inputs:0")
    outputs1 = sess.graph.get_tensor_by_name("Reshape_1:0")
    outputs2 = sess.graph.get_tensor_by_name("Reshape:0")


    print(inputs)
    print(outputs1)
    print(outputs2)
    while True:
        starttime=time.time()
        sess.run([outputs1,outputs2],feed_dict={inputs:np.load(testnp)[np.newaxis]})
        endtime=time.time()
        print(endtime-starttime)
