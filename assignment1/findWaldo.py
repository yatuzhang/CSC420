import numpy as np
import scipy as sp
from scipy.misc import imread
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.misc import toimage


def imreadGrayScale(image):
	img = imread(image, 'L')
	return img

def oddifyFilter(kernel):
	#if row divisible by 2, add a row
	(row, col) = kernel.shape
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

	return image

def conv2d(image, kernel):
	#returns a matrix that is the same size as the input image.  Need to do padding
	#First save the size of the image before padding (this is for use in the convolution step)
	row_init, col_init = image.shape

	result = np.zeros((row_init, col_init))
	#make filter odd sized and padd the image
	kernel = oddifyFilter(kernel)
	image = padding(image, kernel)

	(row_kernel, col_kernel) = kernel.shape
	(row_im, col_im) = image.shape


	for i in range(row_init):
		for j in range(col_init):
			for u in range(0,row_kernel):
				for v in range(0,col_kernel):
					#NOTE: here the kernel index should start at 0.
					result[i,j] += kernel[row_kernel-u,row_kernel-v] * image[i+row_kernel/2+u,j+col_kernel/2+v]

	return result


def conv2d_vec(image, kernel):
	#returns a matrix that is the same size as the input image.  Need to do padding
	#First save the size of the image before padding (this is for use in the convolution step)
	row_init, col_init = image.shape

	result = np.zeros((row_init, col_init))
	#make filter odd sized and padd the image
	kernel = oddifyFilter(kernel)
	image = padding(image, kernel)

	(row_kernel, col_kernel) = kernel.shape
	(row_im, col_im) = image.shape

	#Flatten the Kernel
	kernel = kernel.flatten()
	k_r = row_kernel/2
	k_c = col_kernel/2

	for i in range(row_init):
		for j in range(col_init):
			#NOTE: here the kernel index should start at 0.
			temp = image[i:row_kernel+i, :][:, j:col_kernel+j]
			print (i,j)
			result[i][j] = np.dot(temp.flatten(), np.fliplr(kernel))

	return result


def conv2d_matmul(image, kernel):
	#returns a matrix that is the same size as the input image.  Need to do padding
	#First save the size of the image before padding (this is for use in the convolution step)
	row_init, col_init = image.shape

	#make filter odd sized and padd the image
	kernel = oddifyFilter(kernel)
	image = padding(image, kernel)

	(row_kernel, col_kernel) = kernel.shape
	(row_im, col_im) = image.shape

	#Flatten the Kernel
	kernel = kernel.flatten()
	k_r = row_kernel/2
	k_c = col_kernel/2

	column_vector = np.empty((1,row_kernel*col_kernel))

	#This part is an equivalent for im2col, needs to be optimized!!!!!
	for i in range(row_init):
		for j in range(col_init):
			#NOTE: here the kernel index should start at 0.
			temp = image[i:row_kernel+i, :][:, j:col_kernel+j].reshape(1,-1)
			print column_vector.shape
			column_vector = np.concatenate((column_vector, temp), axis=0)

	result = np.matmul(kernel, column_vector).reshape(row_init,col_init);

	return result

def gaussian_1D(sigma):

	N = 3*sigma

 	gaussian_filter = np.zeros(N,1)
 	for i in range(N):
 			#Not too sure about here, can gaussian filter be even???
 		gaussian_filter[i] = math.exp(-(i-N/2)**2/(sigma**2))/(math.sqrt(2*math.pi)*sigma)

def gaussian_2D(sigmax, sigmay):

	#HERE NEED TO MAKE USE OF THE 3 SIGMA RULE
 	(M, N) = (3*sigmay, 3*sigmax)
 	#initialize the gaussian filter to 0
 	gaussian_filter = np.zeros((M,N))
 	for i in range(M):
 		for j in range(N):
 			term1 = ((i-M/2)**2)/(2.0*sigmax**2)+((j-N/2)**2)/(2.0*sigmay**2)
 			term2 = 2.0*math.pi*sigmax*sigmay
 			gaussian_filter[i,j] = math.exp(-term1)/term2

 	return gaussian_filter



def main():

	#PART 1: CONVOLUTION
	im = imreadGrayScale("waldoNoise.png")
	kernel = imreadGrayScale("templateNoise.png")
	cat = imreadGrayScale("cat.jpg")
	toimage(cat).save('cat_before.jpg')


	#PART 2: GAUSSIAN FILTER
	gaussian_filter = gaussian_2D(15, 2)
	(U, S, V) = np.linalg.svd(gaussian_filter)
	output = signal.convolve2d(cat,gaussian_filter, boundary='fill', mode='same')
	toimage(output).show()

	fig = plt.figure()
	plt.imshow(gaussian_filter, cmap='seismic')
	plt.show()

main()