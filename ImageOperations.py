import cv2,numpy as np
from scipy.signal import convolve2d
import math
from skimage.util import random_noise
def main_translate(image,t1,t2):
    T_matrix = np.array([[1, 0, t1],
                     [0, 1, t2],
                     [0, 0, 1]])
    return translate(image,T_matrix)

def translate(image, translation_matrix):
    # This function applies translation, with zero padding, with possibly cutting off part of the picture as result
    new_image = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            new_position = [x, y, 1]
            new_position = np.matmul(translation_matrix, new_position)
            if new_position[0] < image.shape[0] and new_position[1] < image.shape[1]:
                new_image[new_position[0]][new_position[1]] = image[x][y]
    return new_image
def cartoonify(image,edge_factor,means_factor,outlines_factor):
    s=giveShapes(image,edge_factor)
    im=k_means(image,means_factor)
    addOutlines(im,s/255,outlines_factor)
    return im
def single_powerLawTransform(image, r):
    newImage = image.copy()
    # Carrying out the s=n^r operation on all values
    newImage = pow(newImage / 255, r)
    # Multiplying the values by 255 again, for display reasons
    return (newImage * 255).astype(np.uint8)
def all_power_law(image,n):
    b, g, r = cv2.split(image)
    bN = single_powerLawTransform(b,n)
    gN = single_powerLawTransform(g,n)
    rN = single_powerLawTransform(r,n)
    return cv2.merge([bN, gN, rN])
def main_power_law(image,r):
    if len(image.shape)==3:
        return all_power_law(image,r)
    else:
        return single_powerLawTransform(image,r)

def pointwiseInverse(image):
    # Since the values range from [0-255], we do the operation s=255-n, where s in the new value and n the old value
    return 255 - image

def allInverse(image):
    b,g,r=cv2.split(image)
    bN = pointwiseInverse(b)
    gN = pointwiseInverse(g)
    rN = pointwiseInverse(r)
    return cv2.merge([bN, gN, rN])
def main_inverse(image):
    if len(image.shape)==3:
        return allInverse(image)
    else:
        return pointwiseInverse(image)

def uniform_quan(image,q):
    if len(image.shape) == 3:
       return allcolors_quantization(image,q)
    else:
       return uniform_quantization(image,q)
def uniform_quantization(GrayImage, q):
    if q == 2:
        QImage = np.zeros(GrayImage.shape)
        for x in range(GrayImage.shape[0]):
            for y in range(GrayImage.shape[1]):
                QImage[x][y] = math.floor(float(GrayImage[x][y]) / (256.0 / q)) * (256 / q)
        return QImage.astype(np.uint8)
    else:
        bins = np.linspace(GrayImage.min(), GrayImage.max(), q)
        QImage = np.digitize(GrayImage, bins)
        QImage = (np.vectorize(bins.tolist().__getitem__)(QImage-1).astype(int))
        return QImage.astype(np.uint8)
def noise(im,seed):
    b,g,r=cv2.split(im)
    bN=random_noise(b, mode='gaussian', seed=seed)
    gN=random_noise(g, mode='gaussian', seed=seed)
    rN=random_noise(r, mode='gaussian', seed=seed)
    #print(np.shape(b))
    bN = (255 * bN).astype(np.uint8)
    gN=(255*gN).astype(np.uint8)
    rN=(255*rN).astype(np.uint8)
    return cv2.merge([bN,gN,rN])
def allcolors_quantization(Image,q): # splits colors and applies uniform quantization separately
    b,g,r=cv2.split(Image)
    b2=uniform_quantization(b,q)
    g2=uniform_quantization(g,q)
    r2=uniform_quantization(r,q)
    return cv2.merge([b2,g2,r2])
def linear_sampling(image,factor):
    if len(image.shape) == 3:
        sampled_image = image[::factor, ::factor, :]
        [n1, n2, n3] = image.shape
    else:
        sampled_image = image[::factor, ::factor]
        [n1, n2] = image.shape
    return cv2.resize(sampled_image, [n2,n1], interpolation=cv2.INTER_LINEAR)
def nearest_sampling(image,factor):
    if len(image.shape) == 3:
        sampled_image = image[::factor, ::factor, :]
        [n1, n2, n3] = image.shape
    else:
        sampled_image = image[::factor, ::factor]
        [n1, n2] = image.shape
    return cv2.resize(sampled_image, [n2, n1], interpolation=cv2.INTER_NEAREST)

def giveShapes(image, factor=5):
    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)*255
    SOBEL_X = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    SOBEL_Y = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    # By trial and error, for sharp outcome I multiply the normalization factor by 5
    normalization = 1 / sumUp(SOBEL_X) * factor
    # Manual Sobel Edge Detection
    sx = convolve2d(image, SOBEL_X * normalization, mode="same", boundary="symm")
    sy = convolve2d(image, SOBEL_Y * normalization, mode="same", boundary="symm")
    s = np.hypot(sx, sy).astype(np.uint8)
    # Divided by 255 to fit into [0-1]
    s = s #/255
    return s

def sumUp(kernel):
    return np.sum(np.absolute(kernel))

    # T his is a k-means built-in method, which was also used in one of the labs

def addOutlines(image,shape,number):
    if len(image.shape)==3:
        for x in range(shape.shape[0]):
            for y in range(shape.shape[1]):
                if (shape[x][y] > number):
                    image[x][y] = [0, 0, 0]
    else:
        for x in range(shape.shape[0]):
            for y in range(shape.shape[1]):
                if (shape[x][y] > number):
                    image[x][y] = 0
def k_means(image, k):
    pixel_values = image.reshape(-1, 3)
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image
    # This function takes a grayscale images of edges, with edges being white
    # it applies the edges as black onto the image
def resizing(image, scale):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    newSize = (width, height)
    return cv2.resize(image, newSize, interpolation=cv2.INTER_NEAREST)
def resizing(image, width,height):
    newSize = (width, height)
    return cv2.resize(image, newSize, interpolation=cv2.INTER_NEAREST)

