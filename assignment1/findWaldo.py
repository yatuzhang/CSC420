import numpy as np
import scipy as sp
from scipy.misc import imread
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.misc import toimage
from skimage.feature import canny
import networkx as nx
import pylab

def buildGraph(image):
	#This builds a graph from a given
	(M, N) = image.shape 
	im_flattened = image.flatten()
	G = nx.DiGraph()
	G.add_nodes_from(range(0,M*N))


	for i in range(M):
		for j in range(N):
			#handles corner cases
			if i==0 and j==0:
				G.add_edge(i*N+j, (i+1)*N+j,weight=im_flattened[(i+1)*N+j])
				G.add_edge(i*N+j, (i+1)*N+j+1,weight=im_flattened[(i+1)*N+j+1])
			elif i==0 and j==N-1:
				G.add_edge(i*N+j, (i+1)*N+j,weight=im_flattened[(i+1)*N+j])
				G.add_edge(i*N+j, (i+1)*N+j-1,weight=im_flattened[(i+1)*N+j-1])
			elif i==0 and j!=0 and j!=N-1:
				G.add_edge(i*N+j, (i+1)*N+j, weight=im_flattened[(i+1)*N+j])
				G.add_edge(i*N+j, (i+1)*N+j+1, weight=im_flattened[(i+1)*N+j+1])
				G.add_edge(i*N+j, (i+1)*N+j-1, weight=im_flattened[(i+1)*N+j-1])
			elif i==M-1:
				pass
			elif j==0 and i!=0 and i!=M-1:
				G.add_edge(i*N+j, (i+1)*N+j,weight=im_flattened[(i+1)*N+j])
				G.add_edge(i*N+j, (i+1)*N+j+1,weight=im_flattened[(i+1)*N+j+1])
			elif j==N-1 and i!=0 and i!=M-1:
				G.add_edge(i*N+j, (i+1)*N+j, weight=im_flattened[(i+1)*N+j])
				G.add_edge(i*N+j, (i+1)*N+j-1, weight=im_flattened[(i+1)*N+j-1])
			else:
				G.add_edge(i*N+j, (i+1)*N+j, weight=im_flattened[(i+1)*N+j])
				G.add_edge(i*N+j, (i+1)*N+j+1, weight=im_flattened[(i+1)*N+j+1])
				G.add_edge(i*N+j, (i+1)*N+j-1, weight=im_flattened[(i+1)*N+j-1])
	return G

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
	if(N%2==0):
 		N=N+1

 	gaussian_filter = np.zeros(N,1)
 	for i in range(N):
 			#Not too sure about here, can gaussian filter be even???
 		gaussian_filter[i] = math.exp(-(i-N/2)**2/(sigma**2))/(math.sqrt(2*math.pi)*sigma)

def gaussian_2D(sigmax, sigmay):

	#HERE NEED TO MAKE USE OF THE 3 SIGMA RULE
 	(M, N) = (3*sigmay, 3*sigmax)
 	if(M%2==0):
 		M=M+1
 	if(N%2==0):
 		N=N+1

 	print(M,N)
 	#initialize the gaussian filter to 0
 	gaussian_filter = np.zeros((M,N))
 	for i in range(M):
 		for j in range(N):
 			term1 = (((i-M/2)**2)+((j-N/2)**2))/((2.0*sigmax**2)+(2.0*sigmay**2))
 			term2 = 2.0*math.pi*sigmax*sigmay

 			gaussian_filter[i,j] = math.exp(-term1)/term2
 	

 	return gaussian_filter


def gradient_2D(sigma, axis):

	N = 3*sigma
	if(N%2==0):
 		N=N+1
 	#initialize the gaussian filter to 0
 	#axis=0 is horizontal edge detector
	if axis==0:
	 	gradient_filter = np.zeros((N,N))
	 	for i in range(N):
	 		for j in range(N):
	 			term1 = ((i-N/2)**2+(j-N/2)**2)/(2.0*sigma**2)
	 			term2 = 2.0*math.pi*(sigma**4)
	 			gradient_filter[i,j] = -(i-N/2)*math.exp(-term1)/term2

 	#axis=1 is vertical edge detector
	elif axis==1:
	 	gradient_filter = np.zeros((N,N))
	 	for i in range(N):
	 		for j in range(N):
	 			term1 = ((i-N/2)**2+(j-N/2)**2)/(2.0*sigma**2)
	 			term2 = 2.0*math.pi*(sigma**4)
	 			gradient_filter[i,j] = -(j-N/2)*math.exp(-term1)/term2

 	return gradient_filter

def normxcorr2D(image, kernel):
	output = signal.fftconvolve(image,np.fliplr(np.flipud(kernel)), mode='same')/np.linalg.norm(image)/np.linalg.norm(kernel)
	return output

def findWaldo(image, kernel):
	corr = normxcorr2D(image, kernel)
	return corr

def main():

	#PART 1: CONVOLUTION
	im = imreadGrayScale("waldoNoise.png")
	kernel = imreadGrayScale("templateNoise.png")
	cat = imreadGrayScale("cat.jpg")
	tennisCourt = imreadGrayScale("tennisCourt.jpg")
	seamCarving = imreadGrayScale("seam_carving.jpg")
	zermatt = imreadGrayScale("zermatt.JPG")
	paris = imreadGrayScale("paris.JPG")
	#toimage(cat).save('cat_before.jpg')

	#PART 2: GAUSSIAN FILTER
	#gaussian_filter = gaussian_2D(15, 2)
	#fig = plt.figure()
	#plt.imshow(gaussian_filter, cmap='rainbow')
	#plt.show()
	#output = signal.convolve2d(cat,gaussian_filter, boundary='fill', mode='same')
	#toimage(output).save('cat_after.jpg')

	#PART 3: GRADIENT
	#gradient_filterX = gradient_2D(2, axis=0)
	#gradient_filterY = gradient_2D(2, axis=1)
	#print gradient_X
	#toimage(kernel).show()
	#g_Y = signal.convolve2d(kernel,gradient_filterY, boundary='fill', mode='same')
	#g_X = signal.convolve2d(kernel,gradient_filterX, boundary='fill', mode='same')
	#mag_kernel = np.sqrt(g_X**2+g_Y**2)
	#toimage(mag_kernel).show()

	#g_im_Y = signal.convolve2d(im,gradient_filterY, boundary='fill', mode='same')
	#g_im_X = signal.convolve2d(im,gradient_filterX, boundary='fill', mode='same')
	#mag_im = np.sqrt(g_im_X**2+g_im_Y**2)
	#toimage(mag_im).show()

	#corr = normxcorr2D(mag_im, mag_kernel)
	#plt.imshow(corr, cmap='rainbow')
	#plt.show()

	#indices = np.unravel_index(np.argmax(corr), corr.shape)
	#print indices


	#PART 4: Canny Edge Detection
	#gaussian_filter = gaussian_2D(3, 3)
	#tennisCourt = signal.convolve2d(tennisCourt,gaussian_filter, boundary='symm', mode='same')
	#edges = canny(tennisCourt, 1, 10, 20)
	#toimage(edges).save("sigma1low10high20.jpg")
	#sigma, low_threshold, high_threshold
	#leaf_edges = canny(tennisCourt, 20, 20, 30)
	#toimage(leaf_edges).show()
	#plt.imshow(edges, cmap='rainbow')
	#plt.show()


	#PART 5: Dijkstra's Seam Carving

	#Compute the Gradient
	gradient_filterX = gradient_2D(2, axis=0)
	gradient_filterY = gradient_2D(2, axis=1)
	toimage(zermatt).show()
	g_Y = signal.convolve2d(paris,gradient_filterY, boundary='fill', mode='same')
	g_X = signal.convolve2d(paris,gradient_filterX, boundary='fill', mode='same')
	mag = np.sqrt(g_X**2+g_Y**2)
	print mag.shape
	G = buildGraph(mag)
	(M,N) = mag.shape
	implot = plt.imshow(paris, cmap='Greys_r')
	(row, col) = np.unravel_index(nx.dijkstra_path(G, 800, (M-1)*N+1000), (M,N))
	plt.scatter(col, row, c='red', s=2)
	(row, col) = np.unravel_index(nx.dijkstra_path(G, 0, (M-1)*N), (M,N))
	plt.scatter(col, row, c='red', s=2)
	(row, col) = np.unravel_index(nx.dijkstra_path(G, 600, (M-1)*N+250), (M,N))
	plt.scatter(col, row, c='red', s=2)
	plt.axis('off')
	plt.show()


main()