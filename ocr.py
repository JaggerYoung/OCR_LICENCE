import os,sys
sys.path.insert(0,"../../python")
import numpy as np
import mxnet as mx
import cv2, random
from conv_lstm_ctc import get_symbol

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.data_names = data_names
        self.label = label
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

label_map = {' ': 0,
             'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'J': 9,
             'K': 10, 'L': 11, 'M': 12, 'N': 13, 'P': 14, 'Q': 15, 'R': 16,
             'S': 17, 'T': 18, 'U': 19, 'V': 20, 'W': 21, 'X': 22, 'Y': 23, 'Z': 24,
             '0': 25, '1': 26, '2': 27, '3': 28, '4': 29, '5': 30, '6': 31, '7': 32, '8': 33, '9': 34,
             '京': 35, '津': 36, '沪': 37, '渝': 38, '黑': 39, '吉': 40,
             '辽': 41, '冀': 42, '晋': 43, '鲁': 44, '豫': 45, '陕': 46,
             '甘': 47, '青': 48, '苏': 49, '浙': 50, '皖': 51, '鄂': 52,
             '湘': 53, '闽': 54, '赣': 55, '川': 56, '黔': 57, '滇': 58,
             '粤': 59, '琼': 60, '蒙': 61, '宁': 62, '新': 63, '桂': 64, '藏': 65}

def get_label(licence):
    label = []
    for i in licence:
        if not label_map.has_key(i):
            return None
        label.append(label_map[i])
    return tuple(label)

def readList(fname, label_len):
    ###
    """
    read the file of data and label
    """
    gt = []
    return gt
    
class OCRIter(mx.io.DataIter):
    def __init__(self, fname, batch_size, data_shape, label_len, init_states):
        super(OCRIter, self).__init__()

        self.fileList = readList(fname, label_len)
        self.count = len(self.flist)/batch_size

        self.batch_size = batch_size
        self.data_shape = data_shape
        self.label_len = label_len

        self.init_states = init_states
        self.init_state_names = [x[0] for x in init_states]
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size,) + data_shape)] + init_states
        self.provide_label = [('label', (batch_size, label_len))]

    def __iter__(self):
        #random.shuffle(self.flist)
        for k in range(self.count):
            data = []
            label = []
            for i in range(self.batch_size):
                idx = k * self.batch_size + i
                img = cv2.imread(self.file_list[idx][0], cv2.IMREAD_GRAYSCALE)
                img = img[self.fileList[idx][1][0][0]: self.fileList[idx][1][0][3],
                          self.fileList[idx][1][0][0]: self.fileList[idx][1][0][2]]
                img = cv2.resize(img, (self.data_shape[2], self.data_shape[1]))
                img = img.reshape((1,) + img.shape)
                img = np.multiply(img, 1/255.0)
                data.append(img)
                label_tmp = self.fileList[idx][2][0]
                label.append(lb)
            
            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + self.init_state_names
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def ctc_label(p):
    ret = []
    p1 = [0] + p 
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret

def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret

def Accuracy(label, pred):
    BATCH_SIZE = 100
    SEQ_LEN = 16
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LEN):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                     match = False
                     break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total

if __name__ = '__main__':
    label_len = 7
    num_c = len(label_map.key())
    batch_size = 100
    data_shape = (1, 48, 128)
    train_file = 'the file of training data and label'
    test_file = 'the file of testing data and label'

    load_epoch = 70
    model_prefix = 'the prefix of resnet params'
    num_epoch = 1000
    learning_rate = 0.001
    momentum = 0.9
    wd = 0.002

    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 2

    contexts = [mx.context.gpu(0)]

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    print init_states

    data_train = OCRIter(train_file, batch_size, data_shape, label_len, init_states)
    data_val = OCRIter(data_file, batch_size, data_shape, label_len, init_states)

    symbol = sym.get_symbol(num_c = num_c, data_len = 16, label_len = label_len)

    model_args = {}
    if laod_epoch is not None:
        tmp = mx.model.FeedForward.load(model_prefix, load_epoch)
        model_args = {'arg_params': tmp.arg_params,
                      'aux_params': tmp.aux_params,
                      'begin_epoch': load_epoch}

    model = mx.model.FeedForward(ctx=contexts,
                                 num_epoch=num_epoch,
                                 
                                 optimizer = 'adam',
                                 learning_rate = 0.0001
                                 wd = 0.002,
                                 initializer=mx.init.Xavier(factor_type="in", mangitude=2.34),
                                 **model_args)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    print 'begin fit'
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Accuracy),
              batch_end_callback = mx.callback.Speedometer(batch_size, 10),
              epoch_end_callback = checkpoint,)
