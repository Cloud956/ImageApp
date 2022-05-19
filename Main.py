from tkinter import *
from PIL import ImageTk, Image
from ImageOperations import *
from tkinter import filedialog


def makeBasic():
    global root, main_array, MainImage, image_box, white_image, LeftBox, white, current_array, List
    root = Tk()
    root.title("Image processing app")
    root.config(bg="#A0A882")
    root.resizable(width=NO, height=NO)
    image_box = Label()
    white = np.ones([720, 1080, 3]).astype(np.uint8) * 255
    main_array = white
    white_image = ImageTk.PhotoImage(Image.fromarray(white))
    LeftBox = Canvas(root)
    LeftBox.grid(row=1, column=0, columnspan=3)
    RightBox = Canvas(root)
    RightBox.grid(row=1, column=6, columnspan=3)
    StartText = Text(LeftBox, width=30)
    StartText.insert(INSERT, "To start, load an image")
    StartText.config(state=DISABLED)
    StartText.pack()
    StartText = Text(RightBox, width=30)
    StartText.insert(INSERT, "To start, load an image")
    StartText.config(state=DISABLED)
    StartText.pack()
    LeftTop = Label(text="Select the transformation-->")
    LeftTop.config(bg="#CFCF2F")
    LeftTop.grid(row=0, column=0, columnspan=3)
    button = Button(root, text="Display the original(loaded) image", command=lambda: back_to_main())
    button.config(bg="#CFCF2F")
    button.grid(row=2, column=4)
    buttonFilePick = Button(root, text="Load a new Image", command=lambda: load_file())
    buttonFilePick.config(bg="#CFCF2F")
    buttonFilePick.grid(row=2, column=3)
    button = Button(root, text="BIBI", command=lambda: update_main_image("bibi.jpg"))
    button.config(bg="#CFCF2F")
    button.grid(row=2, column=5)
    button = Button(RightBox, text="Save the current image!", command=lambda: saving_image())
    button.config(bg="#CFCF2F")
    button.pack()
    List = give_list()
    List.config(bg="#CFCF2F")
    List.grid(row=0, columns=4)
    current_array = main_array


def back_to_main():
    global main_array, white
    if main_array is not white:
        update_image(main_array)

def saving_image():
    a=1
    #TODO implement this
def give_list():
    global root
    mylist = ["To BGR", "To HSV", "To Gray", "To HLS", "K_means", "Sobel Edge Detection", "Linear sampling",
              "Nearest Neighbour sampling", "Uniform quantization GRAY", "Gaussian Noise", "Pointwise inverse",
              "Power Law Transformation",
              "Cartoonify", "Translation - vertical and horizontal", "Salt and Pepper Noise", "Median filter",
              "Horizontal Periodic Noise","Vertical Periodic Noise","FFT Power Spectrum","FFT Magnitude Spectrum","De-noise in FT"]
    global var
    var = StringVar(root)
    var.set(mylist[0])
    var.trace("w", update_left)
    w = OptionMenu(root, var, *mylist)
    update_left()
    return w

def update_right(*args):
    TODO=2
    #TODO implement this
def update_left(*args):
    global var, LeftBox, main_array, white
    if main_array is not white:
        LeftBox.grid_forget()
        LeftBox = Canvas(width=200, height=700)
        LeftBox.grid(row=1, column=0, columnspan=3)
        Labl = Label(LeftBox, text=f"Current option: {var.get()}")
        Labl.pack()
        text = Text(LeftBox, width=30)
        text.pack()
        match var.get():
            case "To BGR":
                text.insert(INSERT, "Transforms the RGB image to its BGR representative.")
                text.config(state=DISABLED, wrap=WORD)
                button = Button(LeftBox, text="Transform to BGR", command=lambda: to_transform(cv2.COLOR_RGB2BGR))
                button.config(bg="#CFCF2F")
                button.pack()
            case "To HSV":
                text.insert(INSERT, "Transforms the RGB image to its HSV representative.")
                text.config(state=DISABLED, wrap=WORD)
                button = Button(LeftBox, text="Transform to HSV", command=lambda: to_transform(cv2.COLOR_RGB2HSV))
                button.config(bg="#CFCF2F")
                button.pack()
            case "To Gray":
                text.insert(INSERT, "Transforms the RGB image to its grayscale representative.")
                text.config(state=DISABLED, wrap=WORD)
                button = Button(LeftBox, text="Transform to grayscale",
                                command=lambda: to_transform(cv2.COLOR_RGB2GRAY))
                button.config(bg="#CFCF2F")
                button.pack()
            case "To HLS":
                text.insert(INSERT, "Transforms the RGB image to its HLS representative.")
                text.config(state=DISABLED, wrap=WORD)
                button = Button(LeftBox, text="Transform to HLS", command=lambda: to_transform(cv2.COLOR_RGB2HLS))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Sobel Edge Detection":
                text.insert(INSERT,
                            "Displays the shapes in the image, aqcuired using the manual Sobel Edge Detection. "
                            "You can insert and integer below, which is used in the code to maake the edges stronger/weaker. Recommended number is 5!")
                text.config(state=DISABLED, wrap=WORD)
                label1 = Label(LeftBox, text="Edge strength variable")
                label1.pack()
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT, "5")
                entry1.pack()
                button = Button(LeftBox, text="Detect the edges on the main image!",
                                command=lambda: shapes_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Detect the edges on the current image!",
                                command=lambda: shapes_exec(int(entry1.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "K_means":
                text.insert(INSERT,
                            "Limits the number of colors on the image, you can set the number of colors allowed below and see the transformation!")
                text.config(state=DISABLED, wrap=WORD)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT, "5")
                label1 = Label(LeftBox, text="Number of colors")
                label1.pack()
                entry1.pack()
                button = Button(LeftBox, text="K_means transform the main image!",
                                command=lambda: k_means_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="K_means transform the current image!",
                                command=lambda: k_means_exec(int(entry1.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()  # "Linear sampling","Nearest Neighbour sampling"
            case "Linear sampling":
                text.insert(INSERT,
                            "Does the sampling, with resizing using the linear rezising method! Input the sampling factor, which will determine the size of the sampled image below!")
                text.config(state=DISABLED, wrap=WORD)
                entry1 = Entry(LeftBox)
                label1 = Label(LeftBox, text="Sampling factor")
                label1.pack()
                entry1.pack()
                entry1.insert(INSERT, "5")
                button = Button(LeftBox, text="Sample the main image",
                                command=lambda: linear_sampling_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Sample the current image",
                                command=lambda: linear_sampling_exec(int(entry1.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Nearest Neighbour sampling":
                text.insert(INSERT,
                            "Does the sampling, with resizing using the nearest neighbour rezising method! Input the sampling factor, which will determine the size of the sampled image below!")
                text.config(state=DISABLED, wrap=WORD)
                entry1 = Entry(LeftBox)
                label1 = Label(LeftBox, text="Sampling factor")
                label1.pack()
                entry1.insert(INSERT, "5")
                entry1.pack()
                button = Button(LeftBox, text="Sample the main image",
                                command=lambda: nearest_sampling_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Sample the current image",
                                command=lambda: nearest_sampling_exec(int(entry1.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Uniform quantization GRAY":
                text.insert(INSERT,
                            "On grayscale images, reduces the number of colors on the image to the X number you put below. On other images, does the same operation on each of the 3 layers of the image, producing a different, reduced in color image. The number of colors on these images is equal to or lower than X^3.")
                text.config(state=DISABLED, wrap=WORD)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT, "5")
                label1 = Label(LeftBox, text="Number of colors")
                label1.pack()
                entry1.pack()
                button = Button(LeftBox, text="Uniformly quantize the main image!",
                                command=lambda: uniform_quan_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Uniformly quantize the current image!",
                                command=lambda: uniform_quan_exec(int(entry1.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Gaussian Noise":
                text.insert(INSERT,
                            "Applies Gaussian Random Noise to the image. You can put in a seed for the noise below!")
                text.config(state=DISABLED, wrap=WORD)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT, "10001")
                label1 = Label(LeftBox, text="Random noise seed")
                label1.pack()
                entry1.pack()
                button = Button(LeftBox, text="Apply noise to the main image!",
                                command=lambda: gauss_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Apply noise to the current image!",
                                command=lambda: gauss_exec(int(entry1.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Pointwise inverse":
                text.insert(INSERT,
                            "Creates a negative of the image by applying the pointwise inverse operation.")
                text.config(state=DISABLED, wrap=WORD)
                button = Button(LeftBox, text="Invert the main image!",
                                command=lambda: inverse_exec())
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Invert the current image!",
                                command=lambda: inverse_exec(1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Power Law Transformation":
                text.insert(INSERT,
                            "Applies the power law inverse operation on the image. For gray images applies it to the image itself, "
                            "on other images, applies it to all 3 of the layers. Please input a number below."
                            "The operation of (n/255^R)*255 will be performed on the images."
                            "This transformationa is also called gamma adjustment, feel free to experiment with different numbers between 0 and positive infinity."
                            )
                text.config(state=DISABLED, wrap=WORD)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT, "0.5")
                label1 = Label(LeftBox, text="Power variable")
                label1.pack()
                entry1.pack()
                button = Button(LeftBox, text="Apply the transformation to the main image!",
                                command=lambda: power_law_exec(float(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Apply the transformation the current image!",
                                command=lambda: power_law_exec(float(entry1.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Cartoonify":
                text.insert(INSERT,
                            "Cartoonifies the image, using the sobel edge detection and k_means color quantization. "
                            "Input the sobel edge strength factor, number of colors on the image, and the edge outline factor."
                            " The edge outline factor dictates how strong the detected edges need to be, to be shown on the cartoon image."
                            )
                text.config(state=DISABLED, wrap=WORD)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT, "5")
                label1 = Label(LeftBox, text="Sobel Edge Detection factor")
                label1.pack()
                entry1.pack()

                entry2 = Entry(LeftBox)
                entry2.insert(INSERT, "25")
                label2 = Label(LeftBox, text="Number of colours")
                label2.pack()
                entry2.pack()

                entry3 = Entry(LeftBox)
                entry3.insert(INSERT, "0.4")
                label3 = Label(LeftBox, text="Outline variable")
                label3.pack()
                entry3.pack()
                button = Button(LeftBox, text="Cartoonify the main image!",
                                command=lambda: cartoonify_exec(float(entry1.get()), int(entry2.get()),
                                                                float(entry3.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Cartoonify the current image!",
                                command=lambda: cartoonify_exec(float(entry1.get()), int(entry2.get()),
                                                                float(entry3.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Translation - vertical and horizontal":
                text.insert(INSERT,
                            "Applies a vertical and horizontal translation, using a translation matri. "
                            "Input the numbers for horizontal and vertical translating below!"
                            )
                text.config(state=DISABLED, wrap=WORD)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT, "5")
                label1 = Label(LeftBox, text="Vertical translation")
                label1.pack()
                entry1.pack()

                entry2 = Entry(LeftBox)
                entry2.insert(INSERT, "25")
                label2 = Label(LeftBox, text="Horizontal translation")
                label2.pack()
                entry2.pack()

                button = Button(LeftBox, text="Cartoonify the main image!",
                                command=lambda: translation_exec(int(entry1.get()), int(entry2.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Cartoonify the current image!",
                                command=lambda: translation_exec(int(entry1.get()), int(entry2.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Salt and Pepper Noise":
                text.insert(INSERT,
                            "Applies salt and pepper noise to the image. For grayscale applies it directly, for other image formats applies the noise to each layer separately."
                            " Input below number X, where 1/X will be the chance for noise to appear on each pixel."
                            )
                text.config(state=DISABLED, wrap=WORD)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT, "10")
                label1 = Label(LeftBox, text="Chance (1/X)")
                label1.pack()
                entry1.pack()

                button = Button(LeftBox, text="Apply salt and pepper to the main image!",
                                command=lambda: salt_pepper_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Apply salt and pepper to the current image!",
                                command=lambda: salt_pepper_exec(int(entry1.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Median filter":
                text.insert(INSERT,
                            "De-noises the image using a median filter. Good at removing salt and pepper! "
                            "Takes some time and might make the app not respond for a while."
                            )
                text.config(state=DISABLED, wrap=WORD)
                entry1 = Entry(LeftBox)
                entry1.insert(INSERT, "3")
                label1 = Label(LeftBox, text="Size of the kernel X*X")
                label1.pack()
                entry1.pack()

                button = Button(LeftBox, text="Apply salt and pepper to the main image!",
                                command=lambda: median_filter_exec(int(entry1.get())))
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Apply salt and pepper to the current image!",
                                command=lambda: median_filter_exec(int(entry1.get()), 1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Horizontal Periodic Noise":
                text.insert(INSERT,
                            "Applies Horizontal Periodic Noise to the image!"
                            )
                text.config(state=DISABLED, wrap=WORD)

                button = Button(LeftBox, text="Apply periodic noise to the main image!",
                                command=lambda: horizontal_noise_exec())
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Apply periodic noise to the current image!",
                                command=lambda: horizontal_noise_exec(1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "Vertical Periodic Noise":
                text.insert(INSERT,
                            "Applies Vertical Periodic Noise to the image!"
                            )
                text.config(state=DISABLED, wrap=WORD)

                button = Button(LeftBox, text="Apply periodic noise to the main image!",
                                command=lambda: vertical_noise_exec())
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Apply periodic noise to the current image!",
                                command=lambda: vertical_noise_exec(1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "FFT Power Spectrum":
                text.insert(INSERT,
                            "Displays the power spectrium of FFT of the image"
                            )
                text.config(state=DISABLED, wrap=WORD)

                button = Button(LeftBox, text="Display the power spectrum of the main image!",
                                command=lambda: power_exec())
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Display the power spectrum of the current image!",
                                command=lambda: power_exec(1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "FFT Magnitude Spectrum":
                text.insert(INSERT,
                            "Displays the magnitude spectrium of FFT of the image"
                            )
                text.config(state=DISABLED, wrap=WORD)

                button = Button(LeftBox, text="Display the magnitude spectrum of the main image!",
                                command=lambda: magnitude_exec())
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="Display the magnitude spectrum of the current image!",
                                command=lambda: magnitude_exec(1))
                button.config(bg="#CFCF2F")
                button.pack()
            case "De-noise in FT":
                text.insert(INSERT,
                            "De-noises the image, by cutting out a big part of his FFT, in an attempt to remove the parts causing the periodic noise. Outputs a grayscale image, which will probably be quite blurry.")
                text.config(state=DISABLED, wrap=WORD)

                button = Button(LeftBox, text="De-noise the main image!",
                                command=lambda: denoise_fft_exec())
                button.config(bg="#CFCF2F")
                button.pack()
                button = Button(LeftBox, text="De-noise the current image!",
                                command=lambda: denoise_fft_exec(1))
                button.config(bg="#CFCF2F")
                button.pack()


def to_transform(option):
    global main_array
    im = cv2.cvtColor(main_array, option)
    update_image(im)


def load_image(name):
    # print(1)
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def setWhite():
    global image_box, white_image, white
    image_box.grid_forget()
    image_box = Label(image=white_image)
    image_box.grid(row=1, column=3, columnspan=3)
    updateCurrent(white)


def updateCurrent(image):
    global current_array
    current_array = image


def update_image(image):
    global image_box, MainImage, current_array
    image_box.grid_forget()
    MainImage = ImageTk.PhotoImage(Image.fromarray(image))
    image_box = Label(image=MainImage)
    image_box.grid(row=1, column=3, columnspan=3)
    if image.all() == None:
        setWhite()
    updateCurrent(image)


def horizontal_noise_exec(bool=0):
    global main_array, current_array
    if bool == 0:
        im = main_periodic_noise_horizontal(main_array)
    if bool == 1:
        im = main_periodic_noise_horizontal(current_array)
    update_image(im)

def vertical_noise_exec(bool=0):
    global main_array, current_array
    if bool == 0:
        im = main_periodic_noise_vertical(main_array)
    if bool == 1:
        im = main_periodic_noise_vertical(current_array)
    update_image(im)



def salt_pepper_exec(r, bool=0):
    global main_array, current_array
    if bool == 0:
        im = main_salt_pepper(main_array, r)
    if bool == 1:
        im = main_salt_pepper(current_array, r)
    update_image(im)
def denoise_fft_exec(bool=0):
    global main_array, current_array
    if bool == 0:
        im = denoise(main_array)
    if bool == 1:
        im = denoise(current_array)
    update_image(im)
def magnitude_exec(bool=0):
    global main_array, current_array
    if bool == 0:
        im = giveMagnitude(main_array)
    if bool == 1:
        im = giveMagnitude(current_array)
    update_image(im)
def power_exec(bool=0):
    global main_array, current_array
    if bool == 0:
        im = givePower(main_array)
    if bool == 1:
        im = givePower(current_array)
    update_image(im)
def translation_exec(t1=0, t2=0, bool=0):
    global main_array, current_array
    if bool == 0:
        im = main_translate(main_array, t1, t2)
    if bool == 1:
        im = main_translate(current_array, t1, t2)
    update_image(im)


def power_law_exec(numbers, bool=0):
    global main_array, current_array
    if bool == 0:
        im = main_power_law(main_array, numbers)
    if bool == 1:
        im = main_power_law(current_array, numbers)
    update_image(im)


def median_filter_exec(numbers, bool=0):
    global main_array, current_array
    if bool == 0:
        im = main_median_filter(main_array, numbers)
    if bool == 1:
        im = main_median_filter(current_array, numbers)
    update_image(im)


def uniform_quan_exec(numbers, bool=0):
    global main_array, current_array
    if bool == 0:
        im = uniform_quan(main_array, numbers)
    if bool == 1:
        im = uniform_quan(current_array, numbers)
    update_image(im)


def k_means_exec(numbers, bool=0):
    global main_array, current_array
    if bool == 0:
        im = k_means(main_array, numbers)
    if bool == 1:
        im = k_means(current_array, numbers)
    update_image(im)


def cartoonify_exec(edge, k, outlines, bool=0):
    global main_array, current_array
    if bool == 0:
        im = cartoonify(main_array, edge, k, outlines)
    if bool == 1:
        im = cartoonify(current_array, edge, k, outlines)
    update_image(im)


def inverse_exec(bool=0):
    global main_array, current_array
    if bool == 0:
        im = main_inverse(main_array)
    if bool == 1:
        im = main_inverse(current_array)
    update_image(im)


def nearest_sampling_exec(numbers, bool=0):
    global main_array, current_array
    if bool == 0:
        im = nearest_sampling(main_array, numbers)
    if bool == 1:
        im = nearest_sampling(current_array, numbers)
    update_image(im)


def gauss_exec(numbers, bool=0):
    global main_array, current_array
    if bool == 0:
        im = main_noise(main_array, numbers)
    if bool == 1:
        im = main_noise(current_array, numbers)
    update_image(im)


def linear_sampling_exec(numbers, bool=0):
    global main_array, current_array
    if bool == 0:
        im = linear_sampling(main_array, numbers)
    if bool == 1:
        im = linear_sampling(current_array, numbers)
    update_image(im)


def shapes_exec(factor, bool=0):
    global main_array, current_array
    if bool == 0:
        im = giveShapes(main_array, factor)
    if bool == 1:
        im = giveShapes(current_array, factor)
    update_image(im)


def update_main_image(filename):
    global main_array
    main_array = cv2.imread(filename)
    main_array = cv2.cvtColor(main_array, cv2.COLOR_BGR2RGB)
    main_array = resizing(main_array, 1080, 720)
    update_image(main_array)
    update_left()
    update_right();


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
