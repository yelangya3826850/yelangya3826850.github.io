import sys, os
import mxnet as mx
import numpy as np
import random
import string
import glob
import cv2


# path = '../../../../mxnet/example/mywork'
quality = 100
path = './'
def create_rec(path, quality = 100):
	npzfiles = glob.glob(os.path.join(path, '*.npz'))

	X_train = np.array([])
	y_train = np.array([])
	X_val   = np.array([])
	y_val   = np.array([])
	X_test  = np.array([])
	y_test  = np.array([])

	for i in range(len(npzfiles)):
		if npzfiles[i].split('/')[-1] == 'train.npz':
		   	X_train = np.load(npzfiles[i])['X_train']
		   	y_train = np.load(npzfiles[i])['y_train']
		elif npzfiles[i].split('/')[-1] == 'val.npz':
		   	X_val = np.load(npzfiles[i])['X_val']
		   	y_val = np.load(npzfiles[i])['y_val']
		elif npzfiles[i].split('/')[-1] == 'test.npz':
		   	X_test = np.load(npzfiles[i])['X_test']
		   	y_test = np.load(npzfiles[i])['y_test']


	encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]


	if X_train.shape[0] != 0:
		print 'Start to create train.rec ...'

		prefix = 'train'
		record = mx.recordio.MXRecordIO(prefix+'.rec', 'w')

		indice = range(len(X_train))
		labels = np.tile(y_train, (4,1))
		for i in range(len(X_train)):
		    items = X_train[i,:,:,:]
		    content = ''
		    for j in range(len(items)):
			_img = items[j,:,:]
			ret, buf = cv2.imencode('.jpg', _img, encode_params)
			assert ret, 'failed encoding image'
			        
			content = content + buf.tostring()
		    # _label = labels[:,i] 
		    _label = y_train[i]
		    header = (0, _label, indice[i], 0)
		    s = mx.recordio.pack(header, content)
		    record.write(s)


	if X_val.shape[0] != 0:
		print 'Start to create val.rec ...'

		prefix = 'val'
		record = mx.recordio.MXRecordIO(prefix+'.rec', 'w')

		indice = range(len(X_val))
		labels = np.tile(y_val, (4,1))
		for i in range(len(X_val)):
		    items = X_val[i,:,:,:]
		    content = ''
		    for j in range(len(items)):
			_img = items[j,:,:]
			ret, buf = cv2.imencode('.jpg', _img, encode_params)
			assert ret, 'failed encoding image'
			        
			content = content + buf.tostring()
			        
		    # _label = labels[:,i]
		    _label = y_val[i]
		    header = (0, _label, indice[i], 0)
		    s = mx.recordio.pack(header, content)
		    record.write(s)



	if X_test.shape[0] != 0:
		print 'Start to create test.rec ...'

		prefix = 'test'
		record = mx.recordio.MXRecordIO(prefix+'.rec', 'w')

		indice = range(len(X_test))
		labels = np.tile(y_test, (4,1))
		for i in range(len(X_test)):
		    items = X_test[i,:,:,:]
		    content = ''
		    for j in range(len(items)):
			_img = items[j,:,:]
			ret, buf = cv2.imencode('.jpg', _img, encode_params)
			assert ret, 'failed encoding image'
			        
			content = content + buf.tostring()
			        
		    # _label = labels[:,i]
		    _label = y_test[i]
		    header = (0, _label, indice[i], 0)
		    s = mx.recordio.pack(header, content)
		    record.write(s)


	if (X_train.shape[0] == 0) and (X_val.shape[0] == 0) and (X_test.shape[0] == 0):
		assert not (X_train.shape[0] == 0) and (X_val.shape[0] == 0) and (X_test.shape[0] == 0), 'Can not find npz files, please check your data.'
	

	return



create_rec(path)