import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import heapq

path='/content/drive/My Drive/Advance/Identification/'
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


##########
# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image
##########
def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(model_images), len(query_images)))
    
    
    best_match = []

    
    for i in range(len(query_images)):
        for j in range(len(model_images)):
            D[j,i] = dist_module.get_dist_by_name(model_hists[j],query_hists[i],dist_type)
        minimum = list(D[:,i]).index(min(list(D[:,i]))) 
        best_match.append(model_images[minimum])


    return best_match, D
#where D is a matrix which contains distances between all pairs of model and query images



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []
    
    for i in range(len(image_list)):
        img = (np.array(Image.open(image_list[i]))).astype('double')
        if(hist_type=='dxdy'):
            img = rgb2gray(img)
        hist = histogram_module.get_hist_by_name(img, num_bins, hist_type)
        image_hist.append(hist)


    return image_hist


##########
# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image
##########
def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
   
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    
    best_match,D = find_best_match(model_images, query_images, dist_type, hist_type,num_bins)
    
    for i in range(len(query_images)):
        img= np.array(Image.open(query_images[i]))
        #plt.subplot(1,3,1)
        plt.subplot(3,6,i*6+1)
        plt.imshow(img)
        for j in range(1,num_nearest+1):
            minpos = list(D[:,i]).index(heapq.nsmallest(j,(list(D[:,i])))[-1])
            img_match= np.array(Image.open(model_images[minpos]))
            plt.subplot(3,6,i*6+1+j)
            plt.imshow(img_match)

        plt.show()

