import numpy as np
import scipy as sp
from scipy.misc import imread
import matplotlib
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.misc import toimage
from skimage.feature import canny
from skimage import morphology
import skimage
import networkx as nx
import pylab
import cv2
from matplotlib.patches import Rectangle
import matplotlib.patches as patches


def nonMaxSuppression(scores, radius, threshold):
	'''
	Non Maximum Suppression:
		-use a disk of certain radius to convolve with the order_filter to return a local maximas
	'''
	#Generates a disk-shaped morphological element
	disk = morphology.disk(radius)
	localmax = sp.ndimage.rank_filter(scores, -1, \
				mode='constant', cval=0, footprint=disk)
	scores = np.logical_and(localmax==scores,localmax>threshold)

	return scores

def HarrisDetector(image, base_sigma, sigma, threshold, radius):
	'''
	Harris Detector Implementation (Taking gradient first, then apply Gaussian):
		-Ix, Iy, IxIx, IyIy IxIy are computed for each pixel
		-M is formed by [Ixx, Ixy; Ixy, Iyy] and the R value is taken as the Harmonic Mean
	'''


	M, N = image.shape

    #base-level smoothing to supress noise
	image_smooth = sp.ndimage.filters.gaussian_filter(image, \
				   base_sigma, mode='constant', cval=0, truncate=3)

	#Gradient in the X direction
	Ix = np.gradient(image_smooth, axis=0)
	#Gradient in the Y direction
	Iy = np.gradient(image_smooth, axis=1)
	#Gaussian
	#Harmonic Mean
	R = np.zeros((M,N))
	IxIx = sp.ndimage.filters.gaussian_filter(Ix*Ix, \
			sigma, mode='constant', cval=0, truncate=3)
	IyIy = sp.ndimage.filters.gaussian_filter(Iy*Iy, \
			sigma, mode='constant', cval=0, truncate=3)
	IxIy = sp.ndimage.filters.gaussian_filter(Ix*Iy, \
			sigma, mode='constant', cval=0, truncate=3)

	for i in range(M):
		for j in range(N):
			#Harris & Stephens
			R[i,j] = np.linalg.det([[IxIx[i,j], IxIy[i,j]], \
					 [IxIy[i,j], IyIy[i,j]]])/(np.trace([[IxIx[i,j], \
					 IxIy[i,j]], [IxIy[i,j], IyIy[i,j]]]))
	#toimage(R).show()
	suppressed = nonMaxSuppression(R, radius, threshold)
	#plt.imshow(R, cmap='rainbow')

	#PLOT CIRCLES
	fig = plt.figure()
	plt.imshow(image, cmap='Greys_r')
	ax = fig.add_subplot(1, 1, 1)
	for i in range(M):
		for j in range(N):
			if suppressed[i,j] != 0:
				circ=plt.Circle((j,i), radius=radius, color='r', fill=False)
				ax.add_patch(circ)
	plt.show()


def BlobDetector(image):
	'''
	Blob Detector Implementation
		-Use the LoG over a set of scales and perform Non Maxima Suppression
	'''

	M, N = image.shape

	#Absolute threshold of the maxima
	threshold = 60
	#Used for base-level smoothing
	sigma_base = 2
    #base-level smoothing to supress noise
	image_smooth = sp.ndimage.filters.gaussian_filter(image,     \
												 sigma_base,     \
												 mode='constant',\
												 cval=0, truncate=3)


	k = 1.1
	sigma_laplace = 1.5
	sigma = np.multiply(np.power(k, np.arange(20))+1, sigma_laplace)
	LoG = np.zeros((M,N,sigma.size))
	for i in range(sigma.size):
		# A factor of sigma^2 is the normalization needed for true scale invariance.
		LoG[:,:,i] = np.square(sigma[i]) * \
					sp.ndimage.filters.gaussian_laplace(image_smooth, sigma[i], \
						mode='constant', cval=0, truncate=3)
	extrema = sp.ndimage.rank_filter(np.abs(LoG), -1, mode='nearest',\
				footprint=np.ones((10,10,10)))
	#Flatten into 2D
	suppressed = np.amax(np.logical_and(extrema==np.abs(LoG),extrema>threshold), axis=-1)
	sigma_indices = np.argmax(extrema==np.abs(LoG), axis=-1)
	#Get a list of the corresponding sigmas
	#toimage(flattened).save("1_c_2.png")
	fig = plt.figure()
	plt.imshow(image, cmap='Greys_r')
	#PLOT CIRCLES
	ax = fig.add_subplot(1, 1, 1)
	for i in range(M):
		for j in range(N):
			if suppressed[i,j] != 0:
				circ=plt.Circle((j,i), \
					radius=sigma[sigma_indices[i,j]], color='r', fill=False)
				ax.add_patch(circ)
	plt.show()
	

def imreadGrayScale(image):
	'''
	Load image in Grayscale
	'''
	img = imread(image, 'L')
	return img

def euclidean_distance(a, b):
	'''
	Find Euclidean Distance between every vector in a and every vector in b
	a - MxD
	b - NxD
	'''
	aa = np.sum(np.square(a), axis = 1).reshape(-1,1)
	bb = np.sum(np.square(b), axis = 1).reshape(-1,1)
	ab = np.dot(a,b.T)

	distances = np.sqrt(aa+bb.T-2*ab)
	return distances

def find_affine(template, kp_template, image, kp_image, k, score):
	'''
	Compute the affition transformation
	'''
	P = np.zeros((k*2, 6))
	Pp = np.zeros((k*2,1))

	print P
	print Pp
	score = np.array(score)
	indices = score.argsort()[:k]
	print indices
	for i in range(k):
		x, y = kp_template[indices[i]].pt
		print x, y
		xp, yp = kp_image[indices[i]].pt
		print xp, yp
		P[i*2,:] = np.array([x, y, 0, 0, 1, 0])
		P[i*2+1,:] = np.array([0, 0, x, y, 0, 1])
		Pp[i*2] = xp
		Pp[i*2+1] = yp 

	a = np.dot(np.dot(np.linalg.inv(np.dot(P.T, P)), P.T), Pp)
	print a
	return a

def Visualize_Affine(template, kp_template, image,kp_image, mapping):
	'''
	Visualize the affine transformation
	'''
	M,N = template.shape
	fig = plt.figure()
	#NEED TO DO COPY, ASSIGNING TO SCALAR WILL NOT WORK
	#need to swap c, d, e
	c,d,e = mapping.item(2), mapping.item(3), mapping.item(4)
	mapping[2] = e
	mapping[3] = c
	mapping[4] = d
	mapping = mapping.reshape(2,3)
	
	ax2 = fig.add_subplot(122)
	ax1 = fig.add_subplot(121)
	ax1.imshow(template, cmap='Greys_r')
	ax2.imshow(image, cmap='Greys_r')

	template = cv2.drawKeypoints(template, kp_template, None)
	image = cv2.drawKeypoints(image, kp_image, None)
	ax1.imshow(template)
	ax2.imshow(image)

	x = np.array([[0,0], [0, M-1],[N-1,0], [N-1,M-1]])
	#Append a column of ones
	x = np.append(x, np.ones((4,1)), axis=1)
	xp = np.dot(mapping, x.T)
	ax2.scatter(xp[0,:], xp[1,:])
	print xp[:,1]
	line = matplotlib.lines.Line2D((xp[0,0], xp[0,1]), (xp[1,0], xp[1,1]), linewidth=2)
	ax2.add_line(line)
	line = matplotlib.lines.Line2D((xp[0,1], xp[0,3]), (xp[1,1], xp[1,3]), linewidth=2)
	ax2.add_line(line)
	line = matplotlib.lines.Line2D((xp[0,3], xp[0,2]), (xp[1,3], xp[1,2]), linewidth=2)
	ax2.add_line(line)
	line = matplotlib.lines.Line2D((xp[0,0], xp[0,2]), (xp[1,0], xp[1,2]), linewidth=2)
	ax2.add_line(line)
	plt.show()

def Visualize_Matching(template, kp_template, image, kp_image):
	'''
	Visualize the keypoints
	'''
	fig = plt.figure()
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	template = cv2.drawKeypoints(template, kp_template, None)
	image = cv2.drawKeypoints(image, kp_image, None)
	ax1.imshow(template)
	ax2.imshow(image)
	for kp1, kp2 in zip(kp_template, kp_image):
		coord1 = kp1.pt
		coord2 = kp2.pt
		con = ConnectionPatch(xyA=coord2, xyB=coord1, coordsA="data", coordsB="data",
                      axesA=ax2, axesB=ax1, arrowstyle="-", color="b")
		ax2.add_patch(con)
	plt.show()

def SIFT_Matching_Color_2(template, image):
	sift = cv2.xfeatures2d.SIFT_create()
	#Covert to Grayscale
	template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
	image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	print image.shape
	#Use SIFT to ket keypoints and feature descriptors
	kp_image, des_image = sift.detectAndCompute(image_gray,None)
	kp_template, des_template = sift.detectAndCompute(template_gray,None)

	#Padd Features with RGB Values and extend into a longer feature vector
	des_image = np.append(des_image, \
			np.zeros((des_image.shape[0],3)), axis=1)
	des_template = np.append(des_template, \
			np.zeros((des_template.shape[0],3)), axis=1)

	for i, kp in enumerate(kp_image):
		x, y  = kp_image[i].pt
		colors = image[y,x,:]
		des_image[i, -3:] = colors

	for i, kp in enumerate(kp_template):
		x, y  = kp_template[i].pt
		colors = template[y,x,:]
		des_template[i, -3:] = colors


	distances = euclidean_distance(des_template[:, :-3], des_image[:, :-3])
	color_distances = euclidean_distance(des_template[:,-3:], des_image[:,-3:])
	distances = color_distances*0+distances
	print distances.shape
	matching_indices = np.argmin(distances, axis=1)
	#Calculate the reliability score of each correspondance (only keep the ones above a threshold)
	threshold = 0.8
	score = np.divide(np.sort(distances, axis=1)[:,0],np.sort(distances, axis=1)[:,1])
	#Capture the keypoints after applying the threshold
	kp_image = [kp_image[i] for i in matching_indices]
	kp_template = [kp_template[i] for i in range(len(kp_template)) if score[i]<threshold]
	kp_image = [kp_image[i] for i in range(len(kp_image)) if score[i]<threshold]
	score = [score[i] for i in range(len(score)) if score[i]<threshold]
	#This part visualizes the correspondances

	#Visualize_Matching(template, kp_template, image, kp_image)
	#mapping = find_affine(template, kp_template, image, kp_image, 6, score)
	#Visualize_Affine(template,kp_template, image,kp_image, mapping)
	Visualize_Matching(template,kp_template,image,kp_image)

def SIFT_Matching_Color_1(template, image):

	#relative weight assigned for the color features
	weight = 2
	sift = cv2.xfeatures2d.SIFT_create()
	#Covert to Grayscale
	template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
	image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	print image.shape
	#Use SIFT to ket keypoints and feature descriptors
	kp_image, des_image = sift.detectAndCompute(image_gray,None)
	kp_template, des_template = sift.detectAndCompute(template_gray,None)

	#Padd Features with RGB Values and extend into a longer feature vector
	des_image = np.append(des_image, np.zeros((des_image.shape[0],3)), axis=1)
	des_template = np.append(des_template, np.zeros((des_template.shape[0],3)), axis=1)

	for i, kp in enumerate(kp_image):
		x, y  = kp_image[i].pt
		colors = image[y,x,:]
		des_image[i, -3:] = colors*weight

	for i, kp in enumerate(kp_template):
		x, y  = kp_template[i].pt
		colors = template[y,x,:]
		des_template[i, -3:] = colors*weight


	distances = euclidean_distance(des_template, des_image)


	matching_indices = np.argmin(distances, axis=1)
	#Calculate the reliability score of each correspondance (only keep the ones above a threshold)
	threshold = 0.8
	score = np.divide(np.sort(distances, axis=1)[:,0],np.sort(distances, axis=1)[:,1])
	#Capture the keypoints after applying the threshold
	kp_image = [kp_image[i] for i in matching_indices]
	kp_template = [kp_template[i] for i in range(len(kp_template)) if score[i]<threshold]
	kp_image = [kp_image[i] for i in range(len(kp_image)) if score[i]<threshold]
	score = [score[i] for i in range(len(score)) if score[i]<threshold]
	#This part visualizes the correspondances

	Visualize_Matching(template, kp_template, image, kp_image)
	#mapping = find_affine(template, kp_template, image, kp_image, 6, score)
	#Visualize_Affine(template,kp_template, image,kp_image, mapping)
	#Visualize_Matching(template,kp_template,image,kp_image)

def SIFT_Matching(template, image):
	sift = cv2.xfeatures2d.SIFT_create()
	#Covert to Grayscale
	template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	print image.shape
	#Use SIFT to ket keypoints and feature descriptors
	kp_image, des_image = sift.detectAndCompute(image,None)
	kp_template, des_template = sift.detectAndCompute(template,None)
	distances = euclidean_distance(des_template, des_image)


	matching_indices = np.argmin(distances, axis=1)
	#Calculate the reliability score of each correspondance (only keep the ones above a threshold)
	threshold = 0.5
	score = np.divide(np.sort(distances, axis=1)[:,0],np.sort(distances, axis=1)[:,1])
	#Capture the keypoints after applying the threshold
	kp_image = [kp_image[i] for i in matching_indices]
	kp_template = [kp_template[i] for i in range(len(kp_template)) if score[i]<threshold]
	kp_image = [kp_image[i] for i in range(len(kp_image)) if score[i]<threshold]
	score = [score[i] for i in range(len(score)) if score[i]<threshold]
	#This part visualizes the correspondances
	#Visualize_Matching(template, kp_template, image, kp_image)
	mapping = find_affine(template, kp_template, image, kp_image, 30, score)
	Visualize_Affine(template,kp_template, image,kp_image, mapping)

def main():
	building = imreadGrayScale("building.jpg")
	synthetic = imreadGrayScale("synthetic.png")
	book = imread('book.jpg')
	findBook = imread('findBook.jpg')
	colorSearch = imread('colourSearch.png')
	colorTemplate = imread('colourTemplate.png')

	#HarrisDetector(synthetic, 4, 6, 10, 16)
	BlobDetector(building)
	#SIFT_Matching(book, findBook)
	#SIFT_Matching_Color_1(colorTemplate, colorSearch)
main()