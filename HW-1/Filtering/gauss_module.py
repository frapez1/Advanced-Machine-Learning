###########
# This Python script contains all the functions to filter and evaluate the 
# derivative of an image
###########
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2


###########
# Theese two are the two sparable filter
# Input: image 2d and kernel 1d
# Output: filtered image w.r.t. the reference axis
###########
def x_filter(image, kernel):
    shape = np.array(image.shape)
    new_image = np.zeros(shape)
    for row in range(shape[0]):
        new_image[row] = np.convolve(image[row],kernel, mode = 'same')
    return new_image

def y_filter(image, kernel):
    shape = np.array(image.shape)
    new_image = np.zeros(shape)
    for col in range(shape[1]):
        new_image[:,col] = np.convolve(image[:,col],kernel, mode = 'same')
    return new_image


###########
# Gaussian function taking as argument the standard deviation sigma
# The filter should be defined for all integer values x in the range [-3sigma,3sigma]
# The function should return the Gaussian values Gx computed at the indexes x
###########
def gauss(sigma):
    x = np.arange(int(-3*sigma), int(3*sigma)+0.01, 1)
    Gx = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-x**2 / (2 * sigma**2))
    return Gx, x

###########
# Implement a 2D Gaussian filter, leveraging the previous gauss.
# Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
# Leverage the separability of Gaussian filtering
# Input: image, sigma (standard deviation)
# Output: smoothed image
###########
def gaussianfilter(img, sigma):
    
    [Gx,x] = gauss(sigma)
    smooth_img = x_filter(img, Gx)
    smooth_img = y_filter(smooth_img, Gx)
    return smooth_img


###########
# Gaussian derivative function taking as argument the standard deviation sigma
# The filter should be defined for all integer values x in the range [-3sigma,3sigma]
# The function should return the Gaussian derivative values Dx computed at the indexes x
###########
def gaussdx(sigma):
    x = np.arange(int(-3*sigma), int(3*sigma)+0.01, 1)
    Dx = -1/(np.sqrt(2*np.pi)*sigma**3)*x*np.exp(-x**2 / (2 * sigma**2))
    return Dx, x



def gaussderiv(img, sigma):
    
    [Dx,x] = gaussdx(sigma)
    imgDx = x_filter(img, Dx)
    imgDy = y_filter(img, Dx)
    
    return imgDx, imgDy



    
    
    
    
    
    
    
    
    
    
    
