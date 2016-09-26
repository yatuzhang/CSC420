import numpy as np
import scipy as sp
from scipy.misc import imread
import matplotlib.pyplot as plt

def imreadGrayScale(image):
	img = imread(image, 'L')
	return img

def oddifyFilter(kernel):
	#if row divisible by 2, add a row
	row = kernel.shape[0]
	col = kernel.shape[1]
	if(row%2==0):
		newRow = np.zeros((1,col))
		kernel = np.concatenate((kernel,newRow), axis=0)
		row = row + 1
	#if col divisible by 2
	if(col%2==0):
		newCol = np.zeros((row,1))
		kernel = np.concatenate((kernel,newCol), axis=1)
	
	return kernel

def padding(image, kernel):

	#rows & columns to padd before and after
	pad_vsize = kernel.shape[0]/2;
	pad_hsize = kernel.shape[1]/2;

	#npad is a tuple of (before and after);
	npad = ((pad_vsize, pad_vsize),(pad_hsize, pad_hsize))

	image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)

def conv2d(image, filter):
	#returns a matrix that is the same size as the input image.  Need to do padding
	kernel = oddifyFilter(kernel)
	image = padding(image, kernel)

	

def main():
	im = imreadGrayScale("waldoNoise.png")
	kernel = imreadGrayScale("templateNoise.png")


main()