#!/usr/bin/env python
# coding: utf-8

# -*- coding: UTF-8 -*-
from __future__ import division
import os
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm as tqdm
import pdb
from Operators import tf_op_dict

print(tf.__version__)

testnp = "testnp.npy"
Json_path = "hand_landmark_new.json"
pb_path = "hand_landmark.pb"
# Json_path="palm_detection_new.json"
# pb_path="palm_detection.pb"
# # Config TF

# # Convert TFLite
#
# * convert tflite to json
#
# * using google flatbuffer -- flatc，flatc can convert the tflite file to json format
#
# * flatbuffer：https://github.com/google/flatbuffers
#
# * Install it：
#     1. download the git
#     2. cmake -G "Unix Makefiles" //create the MakeFile
#     3. make //create the flatc
#     4. make install //安裝flatc
#
# * Convert:
#     1. copy the structure file 'schema.fbs' from tensorflow to the root of flatbuffer
#     2. #./flatc -t schema.fbs -- xxxxx.tflite
#     3. and you get the json

# # Data Process

# load the json
with open(Json_path, 'r') as f:
    load_dict = json.load(f)

# ## json to numpy

import struct


def four_int2bin2float(i1, i2, i3, i4, dtype='f'):
    s = int.to_bytes(int(i1), 1, 'little') + int.to_bytes(
        int(i2), 1, 'little') + int.to_bytes(
            int(i3), 1, 'little') + int.to_bytes(int(i4), 1, 'little')
    return struct.unpack(dtype, s)


def getOneBuffers(buffers, dtype='f'):
    """
    arg:
    load_dict['buffers'][index]['data']
    """
    if buffers.__len__() % 4 != 0:
        print("buffers length error:", buffers.__len__())
        raise
    data_list = []
    for i in range(buffers.__len__() // 4):
        data_list.append(
            four_int2bin2float(buffers[i * 4 + 0], buffers[i * 4 + 1],
                               buffers[i * 4 + 2], buffers[i * 4 + 3], dtype))
    return np.array(data_list)


def getAllBuffers(buffers):
    """
    arg:
    load_dict['buffers']
    
    convert the json format(uint8) to numpy.array(float)
    """
    data_list = []
    for data_dict in tqdm(buffers,
                          total=buffers.__len__(),
                          desc="TFLite Buffers"):
        if 'data' not in data_dict:
            data_list.append([])
            continue
        data_tmp = data_dict['data']
        data_tmp_list = []
        for i in range(data_tmp.__len__() // 4):
            data_tmp_list.append(
                four_int2bin2float(data_tmp[i * 4 + 0], data_tmp[i * 4 + 1],
                                   data_tmp[i * 4 + 2], data_tmp[i * 4 + 3]))
        data_list.append(np.array(data_tmp_list))

    return np.array(data_list)


# testallbfs = getAllBuffers(load_dict['buffers'])
# print(testbfs.shape)
# testonebfs = getOneBuffers(load_dict['buffers'][101]['data'],'i')
# print(testonebfs,testonebfs.shape)
# testonebfs = getOneBuffers(load_dict['buffers'][102]['data'],'f')
# print(testonebfs,testonebfs.shape)

# ## what is in the tflite

# 网络结构参数
load_dict['subgraphs'][0]['operators']

# 参数大小
load_dict['subgraphs'][0]['tensors']

print(load_dict.keys())
print('\noperator_codes :', load_dict['operator_codes'])
print("\nsubgraphs keys :", load_dict['subgraphs'][0].keys(), end="\n\n")

print(load_dict['subgraphs'][0]['inputs'])
print(load_dict['subgraphs'][0]['outputs'])

# ## check operators

op_set_bincode = set()
op_set_binncode = set()
op_set_nbincode = set()
for op_dict in tqdm(load_dict['subgraphs'][0]['operators']):
    if 'builtin_options_type' in op_dict and 'opcode_index' in op_dict:
        op_set_bincode.add(
            (op_dict['opcode_index'], op_dict['builtin_options_type']))
    elif 'opcode_index' in op_dict and 'builtin_options_type' not in op_dict:
        opadd = load_dict['operator_codes'][
            op_dict['opcode_index']]['builtin_code']
        if opadd == 'CUSTOM':
            opadd = load_dict['operator_codes'][
                op_dict['opcode_index']]['custom_code']
        op_set_nbincode.add(opadd)
    elif 'opcode_index' not in op_dict and 'builtin_options_type' in op_dict:
        op_set_binncode.add(op_dict['builtin_options_type'])
    else:
        print("Don't know some ops")

print(op_set_bincode, op_set_binncode, op_set_nbincode)

# check the tensors
tensor_name_set = set()
for t in tqdm(load_dict['subgraphs'][0]['tensors']):
    tensor_name_set.add(t['name'])
if tensor_name_set.__len__() == load_dict['subgraphs'][0]['tensors'].__len__():
    print("Everyone have an unique name")
else:
    print("Someone have an repeated name")

in_len_set = set()
out_len_set = set()
builtin_options_set = set()
for op in tqdm(load_dict['subgraphs'][0]['operators']):
    # find op func
    op_key = None
    if 'builtin_options_type' in op:
        op_key = op['builtin_options_type']
    elif 'opcode_index' in op:
        op_key = load_dict['operator_codes'][
            op['opcode_index']]['builtin_code']
        if op_key == 'CUSTOM':
            op_key = load_dict['operator_codes'][
                op['opcode_index']]['custom_code']
    else:
        print("Don't know some ops")

    # find input[]
    in_len = op['inputs'].__len__()
    out_len = op['outputs'].__len__()
    in_len_set.add((op_key, in_len))
    out_len_set.add(out_len)
    if 'builtin_options' in op:
        builtin_options_set.add((op_key, tuple(op['builtin_options'])))
        if 'depth_multiplier' in op['builtin_options']:
            print(op['builtin_options']['depth_multiplier'], end=" ")
    else:
        builtin_options_set.add((op_key))

print("The inputs&outputs length:\n", in_len_set, out_len_set)
print("\nThe builtin_options of operators:\n", builtin_options_set)

# ## check tensors

# In[10]:
tensors_key = set()
for t in tqdm(load_dict['subgraphs'][0]['tensors']):
    for k in t.keys():
        tensors_key.add(k)
        if k == 'type':
            print(t)
print(tensors_key)

# # TensorClass


class Tensors():
    def __init__(self, buffers, tensors):
        """
        arg:
        buffers:load_dict['buffers']
        tensors:load_dict['subgraphs'][0]['tensors']
        """
        self.OriBuffers = buffers
        self.OriTensors = tensors
        self.CalTensors = [None] * tensors.__len__()
        pass

    def get_tensor(self, index):
        if self.CalTensors[index] is None:
            dtype = 'f'
            buffers_idx = self.OriTensors[index]['buffer']
            if 'type' in self.OriTensors[index]:
                if self.OriTensors[index]['type'] == 'INT32':
                    dtype = 'i'
            setTen = getOneBuffers(self.OriBuffers[buffers_idx]['data'], dtype)
            self.set_tensor(index,
                            setTen.reshape(self.OriTensors[index]['shape']))
        return self.CalTensors[index]

    def get_shape(self, index):
        return tuple(self.OriTensors[index]['shape'])

    def set_tensor(self, index, tensor, name='Const'):
        #         if tensor.shape!=tuple(self.OriTensors[index]['shape']):
        #             print("input tensor shape not match")
        #             return False
        # if type(tensor)==np.ndarray:
        #     tensor=tf.constant(tensor,name=name)
        self.CalTensors[index] = tensor
        return True


# testTensor
# tT=Tensors(load_dict['buffers'],load_dict['subgraphs'][0]['tensors'])
# print(tT.get_tensor(1).shape,tT.get_shape(1))

# # Inference the Net

AllTensor = Tensors(load_dict['buffers'], load_dict['subgraphs'][0]['tensors'])
graph_ops = load_dict['subgraphs'][0]['operators']

# img = tf.constant(np.load(testnp)[np.newaxis])
with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name='inputs')
    inoutname = [
        inputs.name,
    ]
    outname = []
    AllTensor.set_tensor(0, inputs)
    for op in tqdm(graph_ops):
        # find op func
        op_key = None
        if 'builtin_options_type' in op:
            op_key = op['builtin_options_type']
        elif 'opcode_index' in op:
            op_key = load_dict['operator_codes'][
                op['opcode_index']]['builtin_code']
            if op_key == 'CUSTOM':
                op_key = load_dict['operator_codes'][
                    op['opcode_index']]['custom_code']
        else:
            print("Don't know some ops")
        if op_key not in tf_op_dict:
            print("Something error?", op_key, op)
        op_func = tf_op_dict[op_key]['op_func']

        options = None
        if 'builtin_options' in op:
            options = op['builtin_options']
        input_idx = op['inputs']
        output_idx = op['outputs']
        inputs_list = []
        for inidx in input_idx:
            inputs_list.append(AllTensor.get_tensor(inidx))

        out_tensor = op_func(inputs_list, options)
        if not AllTensor.set_tensor(output_idx[0], out_tensor):
            print("maybe opfunc ", op_func, " error")
            raise
        if output_idx[0] in load_dict['subgraphs'][0]['outputs']:
            inoutname.append(out_tensor.name)
            outname.append(out_tensor.name.split(":")[0])
    constant_graph = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, outname)
    with tf.gfile.FastGFile(pb_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    print("Input Name:", inoutname[0])
    print("Output Name:", inoutname[1:])
