## import packages
import numpy as np
from PIL import Image
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt

import gauss_module



def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



## function gauss (Question 1.a)

sigma = 4.0
[Gx, x] = gauss_module.gauss(sigma)

plt.figure(1)
plt.plot(x, Gx, '.-')
plt.show()



## function gaussianfilter (Question 1.b)

img = rgb2gray(np.array(Image.open('graf.png')))
smooth_img = gauss_module.gaussianfilter(img, sigma)

plt.figure(2)
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
plt.sca(ax1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.sca(ax2)
plt.imshow(smooth_img, cmap='gray', vmin=0, vmax=255)
plt.show()



## function gaussdx (Question 1.c)

sigma = 4.0
[Gx, x] = gauss_module.gauss(sigma)
[Dx, x] = gauss_module.gaussdx(sigma)

plt.figure(5)
plt.plot(x, Gx, 'b.-')
plt.plot(x, Dx, 'r.-')
plt.legend( ('gauss', 'gaussdx'))
plt.show()



## function gaussdx (Question 1.d)

img_imp = np.zeros([27,27])
img_imp[13, 13] = 1.0


plt.figure(6), plt.imshow(img_imp, cmap='gray')

sigma = 7.0
[Gx, x] = gauss_module.gauss(sigma)
[Dx, x] = gauss_module.gaussdx(sigma)

Gx = Gx.reshape(1, Gx.size)
Dx = Dx.reshape(1, Dx.size)

plt.figure(7)
plt.subplot(2,3,1)
plt.axis('off')
plt.imshow(conv2(conv2(img_imp, Gx, 'same'), Gx.T, 'same') , cmap='gray')
plt.title('Gx.T[Gx(Img)]')
plt.subplot(2,3,2)
plt.axis('off')
plt.imshow(conv2(conv2(img_imp, Gx, 'same'), Dx.T, 'same') , cmap='gray')
plt.title('Dx.T[Gx(Img)]')
plt.subplot(2,3,3)
plt.axis('off')
plt.imshow(conv2(conv2(img_imp, Dx.T, 'same'), Gx, 'same') , cmap='gray')
plt.title('Gx[Dx.T(Img)]')
plt.subplot(2,3,4)
plt.axis('off')
plt.imshow(conv2(conv2(img_imp, Dx, 'same'), Dx.T, 'same') , cmap='gray')
plt.title('Dx.T[Dx(Img)]')
plt.subplot(2,3,5)
plt.axis('off')
plt.imshow(conv2(conv2(img_imp, Dx, 'same'), Gx.T, 'same') , cmap='gray')
plt.title('Dx[Gx.T(Img)]')
plt.subplot(2,3,6)
plt.axis('off')
plt.imshow(conv2(conv2(img_imp, Gx.T, 'same'), Dx, 'same') , cmap='gray')
plt.title('Gx.T[Dx(Img)]')
plt.show()

###########
# Comments Question 1.d
# 
# zero) Since convolution is Associative and Commutative we have that
#     firs X then Y is equal to first Y then X (with X row vector and Y col vector)
#  
# 1) ﬁrst Gx, then Gx.T
#     As we know the gaussian filter is separable so apply Gx and then Gx.T is 
#     the same of apply the 2D Gaussian kernel at the image, and the same of apply
#     firs Gx.T then Gx.
#     Of course the operation with the 2d kernel is much slower.
#     We need:
#         -   kernel.shape[0]_k**2 * img.shape[0]**2       product operation for 2D kernel
#         -   2 * (kernel.shape[0]_k * img.shape[0]**2)    product operation for 2 1D kernel
# 
# 2-3) ﬁrst Gx, then Dx.T == ﬁrst Dx.T, then Gx   (for the point 0)
#     Here they are smoothing along one direction (the x axis) and trying to identify 
#     the edges along another (the y axis)
# 
# 4) ﬁrst Dx, then Dx.T
#     This is like apply the 2D kernel of derivatives.
#     With this filter we are unable to detect edges parallel to the axes (x and y), 
#     we can only identify edges along mixed directions (f(x,y)) or corners
#     (of course a single pixel is a kind of "corner")    
   
# 5-6) ﬁrst Dx, then Gx.T == ﬁrst Gx.T, then Dx   (for the point 0)
#     This is like 2-3 but with the inverted axis (are their transposed)
###########


## function gaussderiv (Question 1.e)

img_c = np.array(Image.open('graf.png')).astype('double')
img = rgb2gray(img_c)
[imgDx, imgDy] = gauss_module.gaussderiv(img, 7.0)

plt.figure(8)
ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2)
ax3 = plt.subplot(1,3,3)
plt.sca(ax1)
plt.axis('off')
plt.imshow(imgDx, cmap='gray')
plt.sca(ax2)
plt.axis('off')
plt.imshow(imgDy, cmap='gray')
plt.sca(ax3)
plt.axis('off')
imgmag_c = np.sqrt(imgDx**2 + imgDy**2)
plt.imshow(imgmag_c, cmap='gray')
plt.show()

img_b = np.array(Image.open('gantrycrane.png')).astype('double')
img = rgb2gray(img_b)
[imgDx, imgDy] = gauss_module.gaussderiv(img, 7.0)

plt.figure(9)
ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2)
ax3 = plt.subplot(1,3,3)
plt.sca(ax1)
plt.axis('off')
plt.imshow(imgDx, cmap='gray')
plt.sca(ax2)
plt.axis('off')
plt.imshow(imgDy, cmap='gray')
plt.sca(ax3)
plt.axis('off')
imgmag_b = np.sqrt(imgDx**2 + imgDy**2)
plt.imshow(imgmag_b, cmap='gray')
plt.show()

###########
# Comments Question 1.e
# 
# 
# Let's start by commenting on the two images on which the directional 
# derivativeskernels have been applied.
# To better understand, let's focus on the image 'gantrycrane.png', since it has 
# many vertical and horizontal lines.
# These lines are highlighted in two different ways by the two filters: 
# the contours of the vertical pole in the center of the photo are seen only 
# through the filter with the derivative along the x axis and vice versa.
# 
# The important part is to note that the image filtered in this way is strongly 
# conditioned by the presence of noise. 
# To limit this dependence, it is useful to filter the image previously with 
# a Gaussian filter; let's see the difference below.
###########

# first image
plt.figure(10, figsize=(15,8))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
# no filtered
img_c = np.array(Image.open('graf.png')).astype('double')
img = rgb2gray(img_c)
[imgDx, imgDy] = gauss_module.gaussderiv(img, 7.0)
plt.sca(ax1)
imgmag_c = np.sqrt(imgDx**2 + imgDy**2)
plt.imshow(imgmag_c, cmap='gray')

# filtered with a gaussian with sigma = 4
sigma = 4
img = gauss_module.gaussianfilter(rgb2gray(img_c), sigma)
[imgDx, imgDy] = gauss_module.gaussderiv(img, 7.0)
imgmag_b = np.sqrt(imgDx**2 + imgDy**2)
plt.sca(ax2)
plt.imshow(imgmag_b, cmap='gray')
plt.show()


# second image
# no filtered
plt.figure(11, figsize=(15,8))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
# no filtered
img_c = np.array(Image.open('gantrycrane.png')).astype('double')
img = rgb2gray(img_c)
[imgDx, imgDy] = gauss_module.gaussderiv(img, 7.0)
plt.sca(ax1)
imgmag_c = np.sqrt(imgDx**2 + imgDy**2)
plt.imshow(imgmag_c, cmap='gray')

# filtered with a gaussian with sigma = 2
sigma = 2
img = gauss_module.gaussianfilter(rgb2gray(img_c), sigma)
[imgDx, imgDy] = gauss_module.gaussderiv(img, 7.0)
imgmag_b = np.sqrt(imgDx**2 + imgDy**2)
plt.sca(ax2)
plt.imshow(imgmag_b, cmap='gray')
plt.show()

###########
# Comments Question 1.e part.2
# 
# As we can see, the images on the left have a lot of noise (in the form of very
# thin net lines) but as soon as we apply a Gaussian filter this noise is 
# smoothed and these lines disappear.
# In general, we always do this: first we filter to reduce noise and
# then we filter to detect edges.
###########
