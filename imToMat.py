# python code to convert an mxn image to an m'xn' image
# usage : python imToMat.py [infile] [m'] [n'] [output_file]
# infile is a file with all the image file names. We can get this by "ls>infile" in the dir
 

import numpy as np
import skimage
from skimage import io
from skimage.transform import resize
import sys
from scipy import misc

# Process an image
def imageProcess(imageFile, pixelX, pixelY):
	arr = misc.imread(imageFile)
	downsizedImage = misc.imresize(arr,(pixelX,pixelY))
	print(downsizedImage.shape)
	print(pixelX*pixelY*3)
	image1D = np.reshape(downsizedImage,(1,pixelX*pixelY*3))
	return image1D

# Main Function
if __name__=="__main__":
	m = int(sys.argv[2])
	n = int(sys.argv[3])
	y = np.zeros((1,m*n*3))
	print (y.shape)
	files = open(sys.argv[1],'r')
	for fileName in files:
		fileName = fileName.rstrip('\n')
		x = imageProcess(fileName,m,n)
		y = np.concatenate((y,x),axis=0)
	print (y.shape)
	np.savetxt(sys.argv[4], y, delimiter=',')  
