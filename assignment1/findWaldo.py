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
import cv2
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

def buildGraph(image):
	#This builds a graph from a given
	(M, N) = image.shape 
	im_flattened = image.flatten()

	#Build graph by patches and vectorization

	#First, initialize a dense array of MxN x MxN  (filled with np.inf, meaning no edge)
	sink = M*N
	source = M*N+1

	G_dense = np.full((M*N+2, M*N+2), np.inf)
	#Build a 2D array of indices
	indices = arange(M*N).reshape(M,N)
	# Get windows and flatten, meaning no edge
	patch_left_top = indices[0:M-1, 0:N-1].reshape(-1,1)
	patch_right_bottom = indices[1:M, 1:N].reshape(-1,1)
	patch_top_right = indices[0:M-1, 1:N].reshape(-1,1)
	path_bottom_left = indices[1:M, 0:N-1].reshape(-1,1)
	patch_top = indices[0:M-1, N]
	patch_bottom = indices[1:M, N]

	#Edit the weights
	G_dense[patch_left_top, patch_right_bottom] = image_flatteneed[path_right_bottom]
	G_dense[patch_top_right, patch_bottom_left] = image_flatteneed[path_bottom_left]
	G_dense[patch_top, patch_bottom] = image_flatteneed[path_bottom]
	#Edit Source and Sink
	G_dense[sink,:]  = 
	G_dense[source, :]
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

def buildGraph_Segmentation(image):
	#This builds a graph from a given
	(M, N) = image.shape 
	im_flattened = image.flatten()
	G = nx.DiGraph()
	G.add_nodes_from(range(0,M*N))


	for i in range(1, M-1):
		for j in range(1, N-1):
			#Don't really care about corners here
			#Down
			G.add_edge(i*N+j, (i+1)*N+j,weight=im_flattened[(i+1)*N+j])
			#Up
			G.add_edge(i*N+j, (i-1)*N+j,weight=im_flattened[(i-1)*N+j])
			#Left	
			G.add_edge(i*N+j, i*N+j+1,weight=im_flattened[i*N+j+1])
			#Right
			G.add_edge(i*N+j, i*N+j-1,weight=im_flattened[i*N+j-1])
			#Down right
			G.add_edge(i*N+j, (i+1)*N+j+1,weight=im_flattened[(i+1)*N+j+1])
			#Up right
			G.add_edge(i*N+j, (i-1)*N+j+1,weight=im_flattened[(i-1)*N+j+1])
			#Down left
			G.add_edge(i*N+j, (i+1)*N+j-1,weight=im_flattened[(i+1)*N+j-1])
			#Up left
			G.add_edge(i*N+j, (i-1)*N+j-1,weight=im_flattened[(i-1)*N+j-1])
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

	N = 6*sigma
	if(N%2==0):
 		N=N+1

 	print N
 	gaussian_filter = np.zeros((N,1))
 	for i in range(N):
 			#Not too sure about here, can gaussian filter be even???
 		gaussian_filter[i] = math.exp(-(i-N/2)**2/(sigma**2))/(math.sqrt(2*math.pi)*sigma)

 	return gaussian_filter


def gaussian_2D(sigmax, sigmay):

	#HERE NEED TO MAKE USE OF THE 3 SIGMA RULE
 	(M, N) = (6*sigmay, 6*sigmax)
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

	N = 6*sigma
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
	k_r, k_c = kernel.shape

	gradient_filterX = gradient_2D(1, axis=0)
	gradient_filterY = gradient_2D(1, axis=1)
	toimage(kernel).show()
	g_Y = signal.convolve2d(kernel,gradient_filterY, boundary='fill', mode='same')
	g_X = signal.convolve2d(kernel,gradient_filterX, boundary='fill', mode='same')
	mag_kernel = np.sqrt(g_X**2+g_Y**2)
	toimage(mag_kernel).show()

	g_im_Y = signal.convolve2d(image,gradient_filterY, boundary='fill', mode='same')
	g_im_X = signal.convolve2d(image,gradient_filterX, boundary='fill', mode='same')
	mag_im = np.sqrt(g_im_X**2+g_im_Y**2)
	#toimage(mag_im).show()

	corr = normxcorr2D(mag_im, mag_kernel)
	(max_x, max_y) = np.unravel_index(np.argmax(corr), corr.shape)

	#plt.imshow(corr, cmap='rainbow')
	#plt.show()

	box_coor1 = (max_x-k_r/2, max_y-k_c/2)
	box_coor2 = (max_x+k_r/2, max_y+k_c/2)

	fig,ax = plt.subplots(1)
	ax.imshow(image, cmap="Greys_r")
	rect = patches.Rectangle((max_y-k_c/2,max_x-k_r/2),k_c,k_r,linewidth=1,edgecolor='r',facecolor='none')
	ax.add_patch(rect)
	plt.show()

def SeamCarving(image, gradient, num_seams):

	M, N = image.shape
	im_flattened = image.flatten()
	gradsums = []
	# Repeat num_seams times.
	for i in range(num_seams):
		print i
		last_row = (M-1)*N 
		G = buildGraph(gradient)
		for j in range(N):
			for k in range(max(0,j+1-M), min(N, j+M)):
				if(M>N or k<M):
					gradsums.append((i, last_row+k, nx.dijkstra_path_length(G, j, last_row+k)))

		gradsums.sort(key=lambda x: x[2])
		#Get index and weight
		x,y,value = gradsums[0]
		gradsums = []
		shortest_path = nx.dijkstra_path(G, x, y)
		#Delete the paths
		#reduce column by one
		N = N - 1
		im_flattened = np.delete(im_flattened, shortest_path)
		gradient = np.delete(gradient, shortest_path).reshape(M, N)
		print gradient.shape
	return im_flattened.reshape(M,N)

def main():

	#PART 1: CONVOLUTION
	im = imreadGrayScale("waldoNoise.png")
	kernel = imreadGrayScale("templateNoise.png")
	cat = imreadGrayScale("cat.jpg")
	tennisCourt = imreadGrayScale("tennisCourt.jpg")
	seamCarving = imreadGrayScale("seam_carving.jpg")
	zermatt = imreadGrayScale("zermatt.JPG")
	paris = imreadGrayScale("paris.JPG")
	cattle = imreadGrayScale("cattle.png")
	toimage(paris).save('paris_before.jpg')

	#PART 2: GAUSSIAN FILTER
	#gaussian_filter = gaussian_2D(15, 2)
	#fig = plt.figure()
	#plt.imshow(gaussian_filter, cmap='rainbow')
	#plt.show()
	#output = signal.convolve2d(cat,gaussian_filter, boundary='fill', mode='same')
	#toimage(output).save('cat_after.jpg')

	#PART 3: GRADIENT

	#findWaldo(im, kernel)
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
	#gradient_filterX = gradient_2D(2, axis=0)
	#gradient_filterY = gradient_2D(2, axis=1)
	#toimage(paris).show()
	#g_Y = signal.convolve2d(paris,gradient_filterY, boundary='fill', mode='same')
	#g_X = signal.convolve2d(paris,gradient_filterX, boundary='fill', mode='same')
	#mag = np.sqrt(g_X**2+g_Y**2)

	im_new = SeamCarving(paris, mag, 10)
	toimage(im_new).save("paris.jpg")
	#print mag.shape
	#G = buildGraph(mag)
	#(M,N) = mag.shape
	#implot = plt.imshow(paris, cmap='Greys_r')
	#(row, col) = np.unravel_index(nx.dijkstra_path(G, 800, (M-1)*N+1000), (M,N))
	#plt.scatter(col, row, c='red', s=2)
	#(row, col) = np.unravel_index(nx.dijkstra_path(G, 0, (M-1)*N), (M,N))
	#plt.scatter(col, row, c='red', s=2)
	#(row, col) = np.unravel_index(nx.dijkstra_path(G, 600, (M-1)*N+250), (M,N))
	#plt.scatter(col, row, c='red', s=2)
	#plt.axis('off')
	#plt.show()

	#gradient_filterX = gradient_2D(2, axis=0)
	#gradient_filterY = gradient_2D(2, axis=1)
	#toimage(cattle).show()
	#g_Y = signal.convolve2d(cattle,gradient_filterY, boundary='fill', mode='same')
	#g_X = signal.convolve2d(cattle,gradient_filterX, boundary='fill', mode='same')
	#mag = np.sqrt(g_X**2+g_Y**2)

	#G = buildGraph_Segmentation(mag)
	#(M,N) = mag.shape
	#plt.imshow(cattle, cmap='Greys_r')
	#num_points = 8
	#x = plt.ginput(num_points)
	#plt.show()
	#implot = plt.imshow(cattle, cmap='Greys_r')
	#for i in xrange(1,num_points,1):
  		#x1, y1 = x[i-1]
  		#x2, y2 = x[i]
  		#point1 = int(y1*N+x1)
  		#point2 = int(y2*N+x2)
		#(row, col) = np.unravel_index(nx.dijkstra_path(G, point1, point2), (M,N))
		#plt.scatter(col, row, c='red', s=3)
  	#x1, y1 = x[-1]
  	#x2, y2 = x[0]
  	#point1 = int(y1*N+x1)
  	#point2 = int(y2*N+x2)
	#(row, col) = np.unravel_index(nx.dijkstra_path(G, point1, point2), (M,N))
	#plt.scatter(col, row, c='red', s=3)
	#plt.show()
main()