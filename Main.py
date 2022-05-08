from tkinter import *
from PIL import ImageTk,Image
import cv2
import numpy as np
from scipy.signal import convolve2d

def load_image(name):
    image = cv2.imread(name)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image
def resizing(image, scale):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    newSize = (width, height)
    return cv2.resize(image, newSize, interpolation=cv2.INTER_NEAREST)
def resizing(image, width,height):
    newSize = (width, height)
    return cv2.resize(image, newSize, interpolation=cv2.INTER_NEAREST)

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
def giveShapes(image, factor=5):
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
    s = s / 255
    return s

def sumUp(kernel):
    return np.sum(np.absolute(kernel))

    # T his is a k-means built-in method, which was also used in one of the labs

def addOutlines(shape, image):
    for x in range(shape.shape[0]):
        for y in range(shape.shape[1]):
            if (shape[x][y] > 0.4):
                image[x][y] = [0, 0, 0]



def main():

    root = Tk()
    im=load_image("bibi.jpg")
    im = resizing(im,1080,720)
    global myimg
    myimg = ImageTk.PhotoImage(Image.fromarray(im))
    global image_box
    image_box = Label(image=myimg)
    image_box.grid(row=0,column=3,columnspan=3)
    yes =1
    LeftBox=Label(text="LEFT")
    LeftBox.grid(row=0,column=0,columnspan=3)
    RightBox=Label(text="RIGHT")
    RightBox.grid(row=0,column=6,columnspan=3)
    def update_image(image):
        global image_box
        global myimg
        image_box.grid_forget()
        myimg = ImageTk.PhotoImage(Image.fromarray(image))
        image_box = Label(image=myimg)
        image_box.grid(row=0, column=3, columnspan=3)


    def k_means_exec(image):
        im = k_means(image, 10)
        update_image(im)


    button =Button(root,text="K_means!",command=lambda :k_means_exec(im))
    button.grid(row=1,column=4)
    root.mainloop()


if __name__ == "__main__":
    main()