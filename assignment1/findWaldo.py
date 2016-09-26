import numpy as np
import scipy as sp
from scipy.misc import imread
import matplotlib.pyplot as plt

def imreadGrayScale(image):
	img = imread(image, 'L')
	return img

#def padding(image):


#def conv2d(image, filter):
	#returns a matrix that is the same size as the input image.  Need to do padding


def main():
	im = imreadGrayScale("waldoNoise.png")
	kernel = imreadGrayScale("templateNoise.png")
	print im.shape
	print kernel.shape
main()