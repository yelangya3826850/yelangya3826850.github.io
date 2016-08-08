## this file should be run after you run the file:        get_cifar100.py
## and you will find train.rec, test.rec in data folder

import numpy as np
import os, sys
import random

filepath = 'data/cifar100/cifar-100-python/fine'

## check
trainpath = os.path.join(filepath, 'train.txt')
testpath  = os.path.join(filepath, 'test.txt')
assert os.path.exists(trainpath), 'Expected "%s" to exist' % trainpath
assert os.path.exists(testpath), 'Expected "%s" to exist' % testpath


### .lst format:
###### number  label   path
lstname = 'train.lst'


train = np.loadtxt(trainpath, str, delimiter = ' ')
Number = np.array(range(len(train)))
train_truples = zip(Number, train[:,1], train[:,0])
random.shuffle(train_truples)

with open(os.path.join(filepath, lstname), 'w') as lstfile:
    for i in xrange(len(train)):
        lstfile.write('{}\t{}\t{}\n'.format(train_truples[i][0], train_truples[i][1], train_truples[i][2],\
                                            file=lstfile))

### .lst format:
###### number  label   path
lstname = 'test.lst'


test = np.loadtxt(testpath, str, delimiter = ' ')
Number = np.array(range(len(test)))
test_truples = zip(Number, test[:,1], test[:,0])
random.shuffle(test_truples)

with open(os.path.join(filepath, lstname), 'w') as lstfile:
    for i in xrange(len(test)):
        lstfile.write('{}\t{}\t{}\n'.format(test_truples[i][0], test_truples[i][1], test_truples[i][2],\
                                            file=lstfile))

os.system("../../bin/im2rec data/cifar100/cifar-100-python/fine/train.lst ./ data/cifar100/train.rec quality=100 resize=-1")
os.system("! ../../bin/im2rec data/cifar100/cifar-100-python/fine/test.lst ./ data/cifar100/test.rec quality=100 resize=-1")