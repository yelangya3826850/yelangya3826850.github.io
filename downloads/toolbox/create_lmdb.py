import scipy.io
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2
import lmdb
import caffe


% matplotlib inline

### parameters
path = '2016-07-28_Images/caffe_data'
phase = 'train'
suffix = 'jpg'


### function
def create_lmdb(path = './', phase = 'train', suffix = 'jpg'):
    file_list = glob.glob(os.path.join(path, phase, '*', '*.'+suffix))
    N = len(file_list)

    # Let's pretend this is interesting data
    Shape = cv2.imread(file_list[0]).transpose(2,0,1).shape
    if len(Shape) == 3: # RGB images
        X = np.zeros((N, Shape[0], Shape[1], Shape[2]), dtype=np.uint8)
        y = np.zeros(N, dtype=np.int64)
    elif len(Shape) == 2:
        X = np.zeros((N, 1, Shape[1], Shape[2]), dtype=np.uint8)
        y = np.zeros(N, dtype=np.int64)
    
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    map_size = X.nbytes * 10
    
    
    for i in xrange(len(file_list)):
        img = cv2.imread(file_list[i])
        img = img.transpose(2,0,1)
        # print img.shape
        X[i,:,:,:] = img
    labels = np.loadtxt(os.path.join(path, phase + '.txt'), str, delimiter = ' ')
    y = labels[:,1].astype('int64')

    env = lmdb.open(os.path.join(path, phase+'_lmdb'), map_size=map_size)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tostring()  # or .tostring() if numpy < 1.9
            datum.label = int(y[i])
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


### implementation
create_lmdb(path, 'train', suffix)
create_lmdb(path, 'val', suffix)
create_lmdb(path, 'test', suffix)


