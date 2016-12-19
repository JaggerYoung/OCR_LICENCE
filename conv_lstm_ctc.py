import os
import sys
sys.path.insert(0,"../../python")
sys.path.insert(0,"../../warpctc")
import mxnet as mx
import numpy as np

from collections import namedtuple
import time
import math

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                    "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def resNet(data):
    conv_0 = mx.sym.Convolution(data=data, kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=16)
    #(16, 16 ,16)
    act_1_0 = mx.symbol.Activation(data=conv_0, act_type='relu')
    conv_1_1 = mx.symbol.Convolution(data=act_1_0, kernel=(3,3), pad=(1,1), num_filter=16)
    act_1_1 = mx.symbol.Activation(data=conv_1_1, act_type="relu")
    conv_1_2 = mx.symbol.Convolution(data=act_1_1, kernel=(3,3), pad=(1,1), num_filter=16)
    esum_1 = mx.symbol.ElementWiseSum(conv_0, conv_1_2)

    #(16, 24, 24)
    act_2_0 = mx.symbol.Activation(data=esum_1, act_type='relu')
    conv_2_1 = mx.symbol.Convolution(data=act_2_0, kernel=(3,3), pad=(1,1), stride=(2,2), num_filter=24)
    act_2_1 = mx.symbol.Activation(data=conv_2_1, act_type='relu')
    conv_2_2 = mx.symbol.Convolution(data=act_2_1, kernel=(3,3), pad=(1,1), num_filter=24)
    conv_2_3 = mx.symbol.Convolution(data=conv_2_2, kernel=(3,3), pad=(1,1), stride=(2,2), num_filter=24)
    esum_2 = mx.symbol.ElementWiseSum(conv_2_3, conv_2_2)

    #(24, 24, 24)
    act_3_0 = mx.symbol.Activation(data=esum_2, act_type='relu')
    conv_3_1 = mx.symbol.Convolution(data=act_3_0, kernel=(3,3), pad=(1,1), num_filter=24)
    act_3_1 = mx.symbol.Activation(data=conv_3_1, act_type="relu")
    conv_3_2 = mx.symbol.Convolution(data=act_3_1, kernel=(3,3), pad=(1,1), num_filter=24)
    esum_3 = mx.symbol.ElementWiseSum(esum_2, conv_3_2)
    
    #(24, 32, 32)
    act_4_0 = mx.symbol.Activation(data=esum_3, act_type='relu')
    conv_4_1 = mx.symbol.Convolution(data=act_4_0, kernel=(3,3), pad=(1,1), stride=(2,2), num_filter=32)
    act_4_1 = mx.symbol.Activation(data=conv_4_1, act_type='relu')
    conv_4_2 = mx.symbol.Convolution(data=act_4_1, kernel=(3,3), pad=(1,1), num_filter=32)
    conv_4_3 = mx.symbol.Convolution(data=conv_4_2, kernel=(3,3), pad=(1,1), stride=(2,2), num_filter=32)
    esum_4 = mx.symbol.ElementWiseSum(conv_4_3, conv_4_2)

    #(32, 32, 32)
    act_5_0 = mx.symbol.Activation(data=esum_4, act_type='relu')
    conv_5_1 = mx.symbol.Convolution(data=act_5_0, kernel=(3,3), pad=(1,1), num_filter=32)
    act_5_1 = mx.symbol.Activation(data=conv_5_1, act_type="relu")
    conv_5_2 = mx.symbol.Convolution(data=act_5_1, kernel=(3,3), pad=(1,1), num_filter=32)
    esum_5 = mx.symbol.ElementWiseSum(esum_4, conv_5_2)

    #(32, 40, 40)
    act_6_0 = mx.symbol.Activation(data=esum_5, act_type='relu')
    conv_6_1 = mx.symbol.Convolution(data=act_6_0, kernel=(3,3), pad=(1,1), stride=(2,2), num_filter=40)
    act_6_1 = mx.symbol.Activation(data=conv_6_1, act_type='relu')
    conv_6_2 = mx.symbol.Convolution(data=act_6_1, kernel=(3,3), pad=(1,1), num_filter=40)
    conv_6_3 = mx.symbol.Convolution(data=conv_6_2, kernel=(3,3), pad=(1,1), stride=(2,2), num_filter=40)
    esum_6 = mx.symbol.ElementWiseSum(conv_6_3, conv_6_2)

    #(40, 40, 40)
    act_7_0 = mx.symbol.Activation(data=esum_6, act_type='relu')
    conv_7_1 = mx.symbol.Convolution(data=act_7_0, kernel=(3,3), pad=(1,1), num_filter=40)
    act_7_1 = mx.symbol.Activation(data=conv_7_1, act_type="relu")
    conv_7_2 = mx.symbol.Convolution(data=act_7_1, kernel=(3,3), pad=(1,1), num_filter=40)
    esum_7 = mx.symbol.ElementWiseSum(esum_6, conv_7_2)
    
    #40x6x16
    act_8 = mx.sym.Activation(data=esum_7, act_type='relu')
    return act_8

def lstm(num_hidden, indata, prev_data, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmod")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)

def lstmUnroll(data, seq_len, layeridx, num_hidden):
    param_cells = LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight"%i),
                            i2h_bias=mx.sym.Variable("l%d_i2h_bias"%i),
                            h2h_weight=mx.sym.Variable("l%d_h2h_weight"%i),
                            h2h_bias=mx.sym.Variable("l%d_h2h_bias"%i))
    last_states = LSTMState(c=mx.sym.Variable("l%d_init_c"%i),
                            h=mx.sym.Variable("l%d_init_h"%i))

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = data[seqidx]
        next_state = lstm(num_hidden, indata=hidden, prev_state=last_states,
                          param=param_cells, seqidx=seqidx, layeridx=layeridx)
        hidden = next_state.h
        last_states = next_state
        hidden_all.append(hidden)
    return hidden_all

def get_symbol(num = 66, data_len = 16, label_len = 7):
    indata = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')

    data = resNet(indata)
    wordvec = mx.symbol.SliceChannel(data=data, num_outputs=data_len, axis=3)
    flatten = []
    for i in range(data_len):
        flatten.append(mx.symbol.Flatten(data=wordvec[i]))

    lstm_0 = lstmUnroll(flatten, 16, 0, 128)
    lstm_1 = lstmUnroll(lstm0, 16, 1, 128)

    concat = mx.sym.Concat(*lstm_1, dim=0)

    pred = mx.sym.FullyConnected(data=concat, num_hidden=num_c, name="lstm_fc")

    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm = mx.sym.WarpCTC(data=pred, label=label, label_lenght=label_len, input_len=data_len)
    return sm

