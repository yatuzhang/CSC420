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


def find_homography(kp_template, kp_image):
	'''
	Given 4 points, Compute the homography
	'''
	A = np.zeros((4*2, 9))

	#We only want 4 points
	for i in range(4):
		x, y = kp_template[i].pt
		xp, yp = kp_image[i].pt
		print xp, yp
		print x, y
		A[i*2,:] = np.array([x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp])
		A[i*2+1,:] = np.array([0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp])

	#Find eigenvalues of ATA
	eigvalues, eigvectors = np.linalg.eig(np.dot(A.T, A))
	min_idx = eigvalues.argmin()   
	h = eigvectors[min_idx]
	h = h.reshape(3,3)
	print h

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

def SIFT_Matching(template, image):

	sift = cv2.xfeatures2d.SIFT_create()
	#Covert to Grayscale
	template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	#Use SIFT to ket keypoints and feature descriptors
	kp_image, des_image = sift.detectAndCompute(image,None)
	kp_template, des_template = sift.detectAndCompute(template,None)
	distances = euclidean_distance(des_template, des_image)

	#Calculate the reliability score of each correspondance (only keep the ones above a threshold)
	threshold = 0.8

 	bf = cv2.BFMatcher()
 	#Return the top 2 matches
 	matches = bf.knnMatch(des_image,des_template, k=2)

 	#keep only matches with a good reliability ratio
 	good = [m for m,n in matches if m.distance < threshold*n.distance]

	src_pts = [ kp_image[m.queryIdx] for m in good ]
	dst_pts = [ kp_template[m.trainIdx] for m in good ]
	src_des = np.float32([ des_image[m.queryIdx] for m in good ])
	dst_des = np.float32([ des_template[m.trainIdx] for m in good ])

	find_homography(src_pts, dst_pts)

#def RANSAC(src_pts, dst_pts):


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

#def drawRec(plot, points):


def computeShoeSize(image):

	fig = plt.figure()
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	#Define the Size of the Dollar Bill
	width= 152.4
	height = 69.85
	#pixels per mm
	pixels = 2
	#Points Order: Top Left, Top Right, Bottom Left, Bottom Right
	dstPts = np.array([[1, 1],[1, width*pixels],[height*pixels, width*pixels],[height*pixels, 1]] )
	ax1.imshow(image)
	srcPts = np.asarray(plt.ginput(4))
	ax1.scatter(*zip(*srcPts))
    # Compute Homography
	h, status = cv2.findHomography(srcPts, dstPts)
	# Warp the image
 	im_out = cv2.warpPerspective(image, h, (int(2*height), int(2*width)))
    # Display the warped 5 dollar bill
	ax2.imshow(np.transpose(im_out, axes=(1, 0, 2)))
	ax1.imshow(image)

	# Input points along the length of the shoe
	srcPts = np.asarray(plt.ginput(2))
	srcPts = np.append(srcPts, [[1],[1]], axis=1)
	plt.show()
	ax1.scatter(*zip(*srcPts))
	line = matplotlib.lines.Line2D(srcPts[0], srcPts[1], linewidth=2)

	#Warp the two points
	warpPts = np.dot(h, srcPts.T)
	#Calculate the number of pixels between the two points (distance)
	length = np.linalg.norm(warpPts[1,0:2]-warpPts[0,0:2])
	#Divide by pixels per mm to get the length
	print "length"+str(length/2)

def main():
	shoe = imread("shoe.jpg")
	tracks = imread("tracks.jpg")

	SIFT_Matching(shoe, tracks)
	#computeShoeSize(shoe)

main()