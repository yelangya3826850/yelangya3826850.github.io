import numpy as np
import matplotlib.pyplot as plt
import Image
import collections
import glob
import os, sys
import random



### parameters setting
path = '2016-07-28_Images/'
phase = 'jpg'
num_train = 1900
num_val = 200
num_test = 400


### loading label file
img_filelist = glob.glob(os.path.join(path, '*', '*.'+phase))
classnames = np.loadtxt(os.path.join(path, 'labels.txt'), str)


lab_dict = dict()
keys = range(len(classnames))
for i in xrange(len(classnames)):
    lab_dict.update({classnames[i]:keys[i]})

# {'THP1': 0, 'OAC': 1, 'MCF7': 2, 'PBMC': 3, 'OST': 4}

labels = []
for items in img_filelist:
    labels.append(lab_dict[items.split('/')[-2]])


data_lab_pairs = zip(img_filelist, labels)
random.shuffle(data_lab_pairs)
train_data_pair = data_lab_pairs[:num_train]
val_data_pair   = data_lab_pairs[num_train : num_train+num_val]
test_data_pair  = data_lab_pairs[num_train+num_val : num_train+num_val+num_test]





##    make storage dir    ##
############################
if os.path.exists(os.path.join(path, 'train')):
    print('The train dir is already existed.')
else:
    os.system("mkdir -p " + os.path.join(path, 'train'))


if os.path.exists(os.path.join(path, 'val')):
    print('The train dir is already existed.')
else:
    os.system("mkdir -p " + os.path.join(path, 'val'))


if os.path.exists(os.path.join(path, 'test')):
    print('The train dir is already existed.')
else:
    os.system("mkdir -p " + os.path.join(path, 'test'))


for _dir in lab_dict:
    if os.path.exists(os.path.join(path, 'train', _dir)):
        print('The {} dir is already existed in {}.'.format(_dir, os.path.join(path, 'train')))
    else:
        os.system("mkdir -p " + os.path.join(path, 'train', _dir))


for _dir in lab_dict:
    if os.path.exists(os.path.join(path, 'val', _dir)):
        print('The {} dir is already existed in {}.'.format(_dir, os.path.join(path, 'val')))
    else:
        os.system("mkdir -p " + os.path.join(path, 'val', _dir))


for _dir in lab_dict:
    if os.path.exists(os.path.join(path, 'test', _dir)):
        print('The {} dir is already existed in {}.'.format(_dir, os.path.join(path, 'test')))
    else:
        os.system("mkdir -p " + os.path.join(path, 'test', _dir))

##############################################################################################





## copy file to storage dir

for _file, _ in train_data_pair:
    os.system('cp ' + _file + ' ' + os.path.join(path, 'train', _file.split('/')[-2]))
for _file, _ in val_data_pair:
    os.system('cp ' + _file + ' ' + os.path.join(path, 'val', _file.split('/')[-2]))
for _file, _ in test_data_pair:
    os.system('cp ' + _file + ' ' + os.path.join(path, 'test', _file.split('/')[-2]))




with open(os.path.join(path, 'train.txt'), 'w') as trainlist:
    for im_file, lab in train_data_pair:
        savepath = os.path.join('/'.join(im_file.split('/')[-2:]))
        trainlist.write('{} {}\n'.format(savepath, lab))


with open(os.path.join(path, 'val.txt'), 'w') as vallist:
    for im_file, lab in val_data_pair:
        savepath = os.path.join('/'.join(im_file.split('/')[-2:]))
        vallist.write('{} {}\n'.format(savepath, lab))


with open(os.path.join(path, 'test.txt'), 'w') as testlist:
    for im_file, lab in test_data_pair:
        savepath = os.path.join('/'.join(im_file.split('/')[-2:]))
        testlist.write('{} {}\n'.format(savepath, lab))


### Notice: when generating the txt file you can only use 'space' as delimiter 
###         of image file path and its label











