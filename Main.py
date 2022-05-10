from tkinter import *
from PIL import ImageTk,Image
from ImageOperations import *
from tkinter import filedialog
def makeBasic():
    global root,main_array,MainImage,image_box,white_image,LeftBox,white,current_array,List
    root = Tk()
    root.config(bg="#A0A882")
    root.resizable(width=NO, height=NO)
    image_box = Label()
    white = np.ones([720, 1080, 3]).astype(np.uint8) * 255
    main_array = white
    white_image = ImageTk.PhotoImage(Image.fromarray(white))
    LeftBox = Canvas(root)
    LeftBox.grid(row=1, column=0, columnspan=3)
    StartText=Text(LeftBox,width=30)
    StartText.insert(INSERT,"To start, load an image")
    StartText.config(state=DISABLED)
    StartText.pack()
    LeftTop=Label(text="Select the transformation")
    LeftTop.config(bg="#CFCF2F")
    LeftTop.grid(row=0,column=0,columnspan=3)
    button= Button(root,text="Display the original(loaded) image",command=lambda:back_to_main())
    button.config(bg="#CFCF2F")
    button.grid(row=2,column=4)
    buttonFilePick = Button(root, text="Load a new Image", command=lambda: load_file())
    buttonFilePick.config(bg="#CFCF2F")
    buttonFilePick.grid(row=2, column=3)
    button=Button(root,text="BIBI",command=lambda:update_main_image("bibi.jpg"))
    button.config(bg="#CFCF2F")
    button.grid(row=2, column=5)
    List=give_list()
    List.config(bg="#CFCF2F")
    List.grid(row=0,columns=4)
    current_array=main_array
def back_to_main():
    global main_array,white
    if main_array is not white:
        update_image(main_array)
def give_list():
    global root
    mylist=["To BGR","To HSV","To Gray","To HLS","K_means","Sobel Edge Detection","Linear sampling","Nearest Neighbour sampling","Uniform quantization GRAY"]
    global var
    var=StringVar(root)
    var.set(mylist[0])
    var.trace("w",update_left)
    w=OptionMenu(root,var,*mylist)
    update_left()
    return w
def update_left(*args):
    global var,LeftBox,main_array,white
    if main_array is not white:
        LeftBox.grid_forget()
        LeftBox=Canvas(width=200,height=700)
        LeftBox.grid(row=1,column=0,columnspan=3)
        Labl = Label(LeftBox,text=f"Current option: {var.get()}")
        Labl.pack()
        text = Text(LeftBox,width=30)
        text.pack()
        match var.get():
            case "To BGR":
                text.insert(INSERT, "Transforms the RGB image to its BGR representative.")
                text.config(state=DISABLED)
                button= Button(LeftBox,text="Transform to BGR",command=lambda:to_transform(cv2.COLOR_RGB2BGR))
                button.config(bg="#CFCF2F")
                button.pack()
            case "To HSV":
                text.insert(INSERT,"Transforms the RGB image to its HSV representative.")
                text.config(state=DISABLED)
                button=Button(LeftBox,text="Transform to HSV",command=lambda:to_transform(cv2.COLOR_RGB2HSV))
                button.config(bg="#CFCF2F")
                button.pack()
            case "To Gray":
                text.insert(INSERT, "Transforms the RGB image to its grayscale representative.")
                text.config(state=DISABLED)
                button = Button(LeftBox, text="Transform to grayscale", command=lambda: to_transform(cv2.COLOR_RGB2GRAY))
                button.config(bg="#CFCF2F")
                button.pack()
            case "To HLS":
                text.insert(INSERT, "Transforms the RGB image to its HLS representative.")
                text.config(state=DISABLED)
                button = Button(LeftBox, text="Transform to HLS", command=lambda: to_transform(cv2.COLOR_RGB2HLS))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Sobel Edge Detection":
                text.insert(INSERT,
                            "Displays the shapes in the image, aqcuired using the manual Sobel Edge Detection. "
                            "You can insert and integer below, which is used in the code to maake the edges stronger/weaker. Recommended number is 5!")
                text.config(state=DISABLED)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT,"5")
                entry1.pack()
                button = Button(LeftBox, text="Detect the edges on the main image!", command=lambda: shapes_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Detect the edges on the current image!", command=lambda: shapes_exec(int(entry1.get()),1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "K_means":
                text.insert(INSERT,
                            "Limits the number of colors on the image, you can set the number of colors allowed below and see the transformation!")
                text.config(state=DISABLED)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT, "5")
                entry1.pack()
                button = Button(LeftBox, text="K_means transform the main image!", command=lambda: k_means_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="K_means transform the current image!",command=lambda: k_means_exec(int(entry1.get()),1))
                button.config(bg="#CFCF2F")
                button.pack()#"Linear sampling","Nearest Neighbour sampling"
            case "Linear sampling":
                    text.insert(INSERT,
                    "Does the sampling, with resizing using the linear rezising method! Input the sampling factor, which will determine the size of the sampled image below!")
                    text.config(state=DISABLED)
                    entry1 = Entry(LeftBox)
                    entry1.pack()
                    entry1.insert(INSERT, "5")
                    button = Button(LeftBox, text="Sample the main image",command=lambda: linear_sampling_exec(int(entry1.get())))
                    button.config(bg="#CFCF2F")
                    button.pack()
                    button = Button(LeftBox, text="Sample the current image",command=lambda: linear_sampling_exec(int(entry1.get()), 1))
                    button.config(bg="#CFCF2F")
                    button.pack()
            case "Nearest Neighbour sampling":
                text.insert(INSERT,"Does the sampling, with resizing using the nearest neighbour rezising method! Input the sampling factor, which will determine the size of the sampled image below!")
                text.config(state=DISABLED)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT,"5")
                entry1.pack()
                button = Button(LeftBox, text="Sample the main image",command=lambda: nearest_sampling_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Sample the current image",command=lambda: nearest_sampling_exec(int(entry1.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Uniform quantization GRAY":
                text.insert(INSERT,   "On grayscale images, reduces the number of colors on the image to the X number you put below. On other images, does the same operation on each of the 3 layers of the image, producing a different, reduced in color image. The number of colors on these images is equal to or lower than X^3")
                text.config(state=DISABLED)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT, "5")
                entry1.pack()
                button = Button(LeftBox, text="Uniformly quantize the main image!", command=lambda: uniform_quan_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Uniformly quantize the current image!",command=lambda: uniform_quan_exec(int(entry1.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()

def to_transform(option):
    global main_array
    im = cv2.cvtColor(main_array,option)
    update_image(im)
def load_image(name):
    print(1)
    image = cv2.imread(name)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def setWhite():
    global image_box,white_image,white
    image_box.grid_forget()
    image_box=Label(image=white_image)
    image_box.grid(row=1,column=3,columnspan=3)
    updateCurrent(white)
def updateCurrent(image):
    global current_array
    current_array=image
def update_image(image):
    global image_box,MainImage,current_array
    image_box.grid_forget()
    MainImage = ImageTk.PhotoImage(Image.fromarray(image))
    image_box = Label(image=MainImage)
    image_box.grid(row=1, column=3, columnspan=3)
    if image.all()==None:
         setWhite()
    updateCurrent(image)

def uniform_quan_exec(numbers,bool=0):
    global main_array, current_array
    if bool == 0:
        im = uniform_quan(main_array, numbers)
    if bool == 1:
        im = uniform_quan(current_array, numbers)
    update_image(im)
def k_means_exec(numbers,bool=0):
    global main_array,current_array
    if bool==0:
        im = k_means(main_array, numbers)
    if bool==1:
        im=k_means(current_array,numbers)
    update_image(im)
def nearest_sampling_exec(numbers,bool=0):
    global main_array, current_array
    if bool == 0:
        im = nearest_sampling(main_array, numbers)
    if bool == 1:
        im = nearest_sampling(current_array, numbers)
    update_image(im)
def linear_sampling_exec(numbers,bool=0):
    global main_array, current_array
    if bool == 0:
        im = linear_sampling(main_array, numbers)
    if bool == 1:
        im = linear_sampling(current_array, numbers)
    update_image(im)
def shapes_exec(factor,bool=0):
    global main_array,current_array
    if bool==0:
        im=giveShapes(main_array,factor)
    if bool==1:
        im=giveShapes(current_array,factor)
    update_image(im)

def update_main_image(filename):
    global main_array
    main_array=cv2.imread(filename)
    main_array=cv2.cvtColor(main_array,cv2.COLOR_BGR2RGB)
    main_array = resizing(main_array, 1080, 720)
    update_image(main_array)
    update_left()

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