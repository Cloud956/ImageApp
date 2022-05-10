import cv2,numpy as np
from scipy.signal import convolve2d
import math
def uniform_quan(image,q):
    if len(image.shape) == 3:
       return allcolors_quantization(image,q)
    if len(image.shape)==2:
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
    if len(image.shape)==2:
        sampled_image = image[::factor, ::factor]
        [n1, n2] = image.shape
    return cv2.resize(sampled_image, [n2,n1], interpolation=cv2.INTER_LINEAR)
def nearest_sampling(image,factor):
    if len(image.shape) == 3:
        sampled_image = image[::factor, ::factor, :]
        [n1, n2, n3] = image.shape
    if len(image.shape) == 2:
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

def addOutlines(shape, image):
    for x in range(shape.shape[0]):
        for y in range(shape.shape[1]):
            if (shape[x][y] > 0.4):
                image[x][y] = [0, 0, 0]

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

