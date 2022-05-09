from tkinter import *
from PIL import ImageTk,Image
import cv2
import numpy as np
from scipy.signal import convolve2d
from tkinter import filedialog
def makeBasic():
    global root
    root = Tk()
    root.resizable(width=NO, height=NO)
    global main_array
    global MainImage
    global image_box
    global white_image
    global LeftBox
    image_box = Label()
    white = np.ones([720, 1080, 3]).astype(np.uint8) * 255
    white_image = ImageTk.PhotoImage(Image.fromarray(white))
    LeftBox = Canvas(root)
    LeftBox.grid(row=1, column=0, columnspan=3)
    LeftTop=Label(text="Select the transformation")
    LeftTop.grid(row=0,column=0,columnspan=3)
    button= Button(root,text="Display the original(loaded) image",command=lambda:back_to_main())
    button.grid(row=2,column=4)
    buttonFilePick = Button(root, text="Load a new Image", command=lambda: load_file())
    buttonFilePick.grid(row=2, column=3)
    global List
    List=give_list()
    List.grid(row=0,columns=4)
def back_to_main():
    global main_array
    update_image(main_array)
def give_list():
    global root
    mylist=["K_means","To BGR","To HSV","To Gray","To HLS"]
    global var
    var=StringVar(root)
    var.set(mylist[0])
    var.trace("w",update_left)
    w=OptionMenu(root,var,*mylist)
    update_left()
    return w
def update_left(*args):
    global var
    global LeftBox
    LeftBox.grid_forget()
    LeftBox=Canvas(width=200,height=700)
    LeftBox.grid(row=1,column=0,columnspan=3)
    Labl = Label(LeftBox,text=f"Current option: {var.get()}")
    Labl.pack()
    text = Text(LeftBox,width=30)


    text.pack()
    match var.get():
        #case "K_means":
            #LeftBox
        case "To BGR":
            text.insert(INSERT, "Transforms the RGB image to its BGR representative.")
            text.config(state=DISABLED)
            button= Button(LeftBox,text="Transform to BGR",command=lambda:to_transform(cv2.COLOR_RGB2BGR))
            button.pack()
        case "To HSV":
            text.insert(INSERT,"Transforms the RGB image to its HSV representative.")
            text.config(state=DISABLED)
            button=Button(LeftBox,text="Transform to HSV",command=lambda:to_transform(cv2.COLOR_RGB2HSV))
            button.pack()
        case "To Gray":
            text.insert(INSERT, "Transforms the RGB image to its grayscale representative.")
            text.config(state=DISABLED)
            button = Button(LeftBox, text="Transform to grayscale", command=lambda: to_transform(cv2.COLOR_RGB2GRAY))
            button.pack()
        case "To HLS":
            text.insert(INSERT, "Transforms the RGB image to its HLS representative.")
            text.config(state=DISABLED)
            button = Button(LeftBox, text="Transform to HLS", command=lambda: to_transform(cv2.COLOR_RGB2HLS))
            button.pack()

def to_transform(option):
    global main_array
    im = cv2.cvtColor(main_array,option)
    update_image(im)
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

def setWhite():
    global image_box
    global white_image
    image_box.grid_forget()
    image_box=Label(image=white_image)
    image_box.grid(row=1,column=3,columnspan=7)
def update_image(image):
    global image_box
    global MainImage
    image_box.grid_forget()
    MainImage = ImageTk.PhotoImage(Image.fromarray(image))
    image_box = Label(image=MainImage)
    image_box.grid(row=1, column=3, columnspan=7)
    if image.all()==None:
         setWhite()


def k_means_exec(image):
    im = k_means(image, 20)
    update_image(im)

def update_main_image(filename):
    global main_array
    main_array=cv2.imread(filename)
    main_array=cv2.cvtColor(main_array,cv2.COLOR_BGR2RGB)
    main_array = resizing(main_array, 1080, 720)
    update_image(main_array)

def load_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select an image")
    update_main_image(filename)
def main():
    global root
    makeBasic()
    setWhite()
    root.mainloop()


if __name__ == "__main__":
    main()