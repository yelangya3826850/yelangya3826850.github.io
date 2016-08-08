import scipy.io
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from PIL import Image
import glob



def Normal(_data):
    ''' This function return the normalization of image data
    
        input :  image:               _data
        
        output:  normalized_image:    data
    '''
    data = _data.copy()
    data = (data-np.min(data)) / (np.max(data) - np.min(data))
    return data



def img2uint8(img):
    ''' This function convert image data from double to uint8
        
        input:   image:        img
        
        output:  image_uin8:   img_uint8
    '''
    try:
        _datatype = data.dtype
    except AttributeError:
        print 'input should be of ndarray type.'
    
    
    if data.dtype == 'uint8':
        print 'the input is already of uint8 type.'
        return img
    
    img = Normal(img)
    img_uint8 = (img * 256).astype('uint8')
    return img_uint8



### global parameters
path = '2016-07-28'
prefix = 'phase'
img_type = 'png'


### make storage dir
store_dir = '/'.join(path.split('/')[:-2])
store_dir = os.path.join(store_dir, path + '_Images')

if os.path.exists(store_dir):
    assert os.path.exists(store_dir), ('The storage directory is already exist.')
else:
    os.system('mkdir -p ' + store_dir)



#####################################
### extract images from mat files ###
#####################################
all_file = glob.glob(os.path.join(path, '*'))

classes = {};
for folder in all_file:
    if os.path.isdir(folder):
        _cla = folder.split('/')[-1]
        if _cla not in classes:
            classes[_cla] = len(classes)

### 1. output classes labels
### write a txt file store classes
with open(os.path.join(store_dir, 'labels.txt'), 'w') as txtfile:
    for items in classes.keys():
        txtfile.write('{}\n'.format(items))


### 2. extract and store
### extract images
for _class in classes.keys():
    
    if os.path.exists(os.path.join(store_dir, _class)):
        assert not os.path.exists(os.path.join(store_dir, _class)), \
        ('The {} directory is already exist in {}.'.format(_class, store_dir))
    else:
        os.system('mkdir -p ' + os.path.join(store_dir, _class))
    

    mat_file = glob.glob(os.path.join(path, _class, '*.mat'))
    Numb = len(mat_file)
    for mat_items in mat_file:
        mat = scipy.io.loadmat(mat_items)
        im = Image.fromarray(img2uint8(mat['phase_cell']))

        savename = os.path.join(store_dir, _class, prefix + '_' + mat_items.split('/')[-1].split('.')[0])
        im.save(savename + '.' + img_type)
        print('{}  saved'.format(savename + '.' + img_type))
    print('\n\nAll the class: {} images saved.'.format(_class))
    print('The total number of class: {} is {}'.format(_class, Numb))

