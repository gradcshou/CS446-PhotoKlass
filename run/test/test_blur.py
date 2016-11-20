# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:42:14 2016

@author: Chenchao
"""

from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

########## functions ##########
def gauss_filter(RGB_img, sigma):
    assert(RGB_img.shape[2] == 3)
    # filter image in each color channel
    new_RGB = np.zeros(RGB_img.shape,dtype='uint8')
    for i in range(3):
        new_RGB[:,:,i] = ndimage.gaussian_filter(RGB_img[:,:,i], sigma)
    return new_RGB
    
    
######### main program #########

imageFile = '13-08-07_2016.jpg'
#img = misc.face()
img = misc.imread(imageFile)

blurred = gauss_filter(img, sigma=5)
blurred_2 = gauss_filter(img, sigma=10)
blurred_3 = gauss_filter(img, sigma=15)


plt.figure(0)
plt.imshow(img)

plt.figure(1)
plt.imshow(blurred)

plt.figure(2)
plt.imshow(blurred_2)

plt.figure(3)
plt.imshow(blurred_3)

plt.show()
