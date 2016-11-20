"""
This code processes images and saves to binary files in the same format as in 
the CiFar10 dataset.

"""
import numpy as np
from scipy import misc
from scipy import ndimage
import random as rd
import os
import shutil

################ Global Constants ################
blur_label = 0
human_label = 1
scene_label = 2

blur_dir = ['blur/Digital_Blur_EvalSet','blur/Digital_Blur_TrainSet',\
    'blur/Natural_Blur_EvalSet','blur/Natural_Blur_TrainSet']
human_dir = ['human']
scene_dir = ['scenic/Crispy_TrainSet','scenic/Digital_Crispy_EvalSet','scenic/Natural_Crispy_EvalSet']

img_ht = 32 # image height in pixel
img_wid = 32 # image width in pixel
n_way_split = 6 # 5 training batches + 1 test batch
blur_sigma = 15 # standard deviation for gaussian filter (no filter if blur_sigma = 0)

bin_dir = 'photo-klass-batches-bin-sigma%d' % blur_sigma # directory that saves binary file

################## Functions #######################

def imageProcess(imageFile, pixelX, pixelY, isblur):
    """
    Convert an image to RGB value with a specific size. 
    Apply guassian filter if it is blurred image (i.e. isblur = True)
    Return a list of 1D image values
    """
    arr = misc.imread(imageFile)
    if isblur and blur_sigma>0:
        arr = gauss_filter(arr,blur_sigma)
    downsizedImage = misc.imresize(arr,(pixelX,pixelY))
    assert(downsizedImage.shape[2] == 3)
    for i in range(3):
        if i == 0:
            image1D = np.reshape(downsizedImage[:,:,i],(1,-1)).ravel().tolist()
        else:
            image1D += np.reshape(downsizedImage[:,:,i],(1,-1)).ravel().tolist()
    return image1D


def unif_div(pos_int, n):
    """
    uniformly divide a positive integer to n groups. Return positive integers x1, x2, ..., xn
    such that x1+x2+ .. xn = pos_int and x1, x2, ..., xn are (approximately) equal
    """
    x_arr = []
    s = 0.0
    for i in xrange(n):
        x = round((pos_int-s)/(n-i))
        x_arr.append(int(x))
        s += x
    assert (min(x_arr)>0), 'invalid output for unif_div' # sanity check, ideally it should never happen
    return x_arr

def gauss_filter(RGB_img, sigma):
    assert(RGB_img.shape[2] == 3)
    # filter image in each color channel
    new_RGB = np.zeros(RGB_img.shape,dtype='uint8')
    for i in range(3):
        new_RGB[:,:,i] = ndimage.gaussian_filter(RGB_img[:,:,i], sigma)
    return new_RGB
        
################# Main Program ###################
        
dirs = [blur_dir, human_dir, scene_dir]
labels = [blur_label, human_label, scene_label]
train_batch = []

rd.seed(14)

for i,data_dir in enumerate(dirs):
    lab = labels[i]
    data = [] # store [label + image]
    for ddir in data_dir:
        # get all the image files in the directory
        img_list = os.listdir(ddir)
        for k,img in enumerate(img_list):
            if not img.startswith('.'): # exclude hidden files                
                img_path = ddir+'/'+img
                if i == 0:
                    # blurred image
                    img_data = imageProcess(img_path,img_ht,img_wid,isblur = True)
                else:
                    # crispy image
                    img_data = imageProcess(img_path,img_ht,img_wid,isblur = False)
                data.append([lab]+img_data)
                print 'Finish processing image %s (%d/%d)' % (img_path,k+1,len(img_list))
    rd.shuffle(data) # random shuffle images
    n_img = len(data) # number of images
    split = unif_div(n_img,n_way_split)
    c_split = np.cumsum(split)
    for j,csp in enumerate(c_split):
        if j == 0:
            sp_data = data[:csp]
        else:
            sp_data = data[c_split[j-1]:csp]
        if j == n_way_split-1:
            # save to test_batch
            if i == 0:
                test_batch = sp_data
            else:
                test_batch += sp_data
        else:
            # save to train_batch
            if i == 0:
                train_batch.append(sp_data)
            else:
                train_batch[j] += sp_data
    print 'Finish extract training and test data for label = %d (%d/%d)' % (lab,i+1,len(labels))

# write training and test batches to binary files
if os.path.exists(bin_dir):
    shutil.rmtree(bin_dir)
os.makedirs(bin_dir)

assert(len(train_batch)==n_way_split-1)
for i,trb in enumerate(train_batch):
    bin_file = '%s/data_batch_%d.bin' % (bin_dir,i+1)
    rd.shuffle(trb)
    with open(bin_file,'wb') as f:
        for data in trb:
            f.write(bytearray(data))
    
bin_file = '%s/test_batch.bin' % bin_dir
rd.shuffle(test_batch)
with open(bin_file,'wb') as f:
    for data in test_batch:
        f.write(bytearray(data))
        
# write meta data file
meta_file = '%s/batches.meta.txt' % bin_dir
with open(meta_file,'w') as f:
    f.write('blur\n')
    f.write('human\n')
    f.write('scenic\n')
    
 
#[image1D,arr] = imageProcess('t1.JPG',32,32)
#with open('test.bin','wb') as f:
#    for i in range(2):
#        f.write(bytearray(image1D))

