#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
folderpath = (os.path.dirname(os.path.realpath(__file__))) + ("\\")
print(folderpath)

import tensorflow as tf
import keras
from keras.datasets import mnist

from tkinter import *
import tkinter as tk

import numpy as np
import cv2
from PIL import Image

from sklearn.cluster import DBSCAN
import pandas as pd
# import matplotlib
import matplotlib.pyplot as plt

# In[2]:


(trainX, trainy), (testX, testy) = mnist.load_data()


# In[3]:


# the mnist dataset has files with white numbers against black backgrounds, but 
# we will just invert that
trainX = 255 - trainX
testX = 255 - testX

# make the arrays 3D, even though it just has 1 channel
trainX_3D = np.expand_dims(trainX, axis=3)
testX_3D = np.expand_dims(testX, axis=3)


# In[4]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(5, (4, 4), activation='relu',
                       input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy')
model.fit(trainX_3D, trainy, epochs = 10)


# In[5]:


test_loss = model.evaluate((testX_3D), (testy))


# In[6]:


# MULTI DIGIT RECOGNITION
CANVAS_HEIGHT = 28
def pixels_to_scatterplot(array):
    ## transpose array / rotate clockwise by 90
    array = np.rot90(array, k=1, axes=(1, 0))
    ## convert black pixels to dots
    binary_inkloc_array = (array <= 127.5)
    #print(binary_inkloc_array)
    ink_coords = np.nonzero(binary_inkloc_array)
    return list(zip(ink_coords[0], ink_coords[1]))
def cluster_classification(ink_coords_list):
    
    X = np.array(ink_coords_list)
    clusters = DBSCAN(eps=3, min_samples=2).fit(X)
    
    canvas = pd.DataFrame(X, columns = ["x", "y"])
    canvas['digit'] = clusters.labels_
    
    return canvas
def see_cluster_classification(canvas):
    groups = canvas.groupby("digit")
    for name, group in groups:
        plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name)
    plt.xlim((0, 28))
    plt.ylim((0, 28))
    plt.legend()
def canvas_cluster_corners(cluster_points):
    # descr: finding the corners for the cluster
    # input: np.array of any given height but a fixed width of 2; stores ints of coord pixels;
    # part of the possible_clusters contant
    # output: list of 4 ints, the corners in this order: min_x, max_x, min_y, max_y
    len_cluster_points = len(cluster_points)
    
    # getting corners
    x_sorted = cluster_points[cluster_points[:,0].argsort()]
    y_sorted = cluster_points[cluster_points[:,1].argsort()]
    min_x = x_sorted[0][0]
    max_x = x_sorted[len_cluster_points - 1][0]
    min_y = y_sorted[0][1]
    max_y = y_sorted[len_cluster_points - 1][1]
    
    return [min_x, max_x, min_y, max_y]
def frame_padding(max_length_possible):
    # if you give in the max height possible for the cluster, this gives the paddings for the top and bottom
    # if you give in the max width possible for the cluster, this gives the paddings for the left wnd right
    padding_range = int (CANVAS_HEIGHT - max_length_possible )
    if padding_range % 2 == 0: # if even, use the top_bottom padding divided 2 for bot the top and bottom
        side_1_padding = padding_range // 2
        side_2_padding = padding_range // 2
    else:  # if odd, use the top_bottom padding divided 2 for bot the top and bottom, but the top gets one added
        side_1_padding = padding_range // 2 + 1
        side_2_padding = padding_range // 2
    return [side_1_padding, side_2_padding]
def canvas_cluster_individualize(clustered_points):

    WHITE = 255

    # SETTING CORNERS AND SIDELENGTHS
    min_x, max_x, min_y, max_y = canvas_cluster_corners(clustered_points)
    x_diff = abs(max_x - min_x)
    x_diff = x_diff + 1
    y_diff = abs(max_y - min_y)
    y_diff = y_diff + 1
    # we add the one to the dimensions to convert the cartesian to table_cells

    # PAINTING ON THE RAW CANVAS
    raw_canvas = np.ones((y_diff, x_diff)) * WHITE
    for pixel_loc in clustered_points:
        temp_x, temp_y = pixel_loc
        raw_canvas[max_y - temp_y][(temp_x - min_x)] = 0

    # DEBATING ABOUT LINEAR TRANSFORMATIONS
    max_height_possible = y_diff
    max_width_possible = x_diff

    # GETTING THE PADDING AMOUNT
    top_padding, bottom_padding = frame_padding(max_height_possible)
    left_padding, right_padding = frame_padding(max_width_possible)

    # PADDING THE CANVAS
    bottom_pad = (np.ones((bottom_padding, raw_canvas.shape[1]))*WHITE).astype(float)
    raw_canvas = np.concatenate((raw_canvas, bottom_pad))

    top_pad = (np.ones((top_padding, raw_canvas.shape[1]))*WHITE).astype(float)
    raw_canvas = np.concatenate((top_pad, raw_canvas))

    left_pad = (np.ones((raw_canvas.shape[0], left_padding))*WHITE).astype(float)
    raw_canvas = np.hstack((left_pad, raw_canvas))

    right_pad = (np.ones((raw_canvas.shape[0], right_padding))*WHITE).astype(float)
    raw_canvas = np.hstack((raw_canvas, right_pad))

    # RETURNING THE CANVAS

    return raw_canvas


def multidigit_test_convtemp_image(image_np_array, model, visible = False):
    # descr: tests the image that you just created
    digit_answers = ""

    ink_coords_list = pixels_to_scatterplot(image_np_array)
    canvas = cluster_classification(ink_coords_list)
    
    
    for cluster_i in list(set(list(canvas['digit']))):
        cluster_subtable = canvas.loc[canvas['digit'] == cluster_i]
        # separate the table by digit
        cluster_subtable = cluster_subtable.drop(['digit'], axis=1)
        # drop the digits column
        clustered_points = cluster_subtable.to_numpy()
        # convert to numpy
        digit_image_np_array = canvas_cluster_individualize(clustered_points)
        digit_answer = test_convtemp_image(digit_image_np_array, model)
        digit_answers = digit_answers + str(digit_answer)
    if visible == True:
        see_cluster_classification(canvas)    
    return digit_answers


# To test your own numbers:
# 
# * Set the global constant "MULTI_DIGIT" to be True if you want the script to recognize a multi-digit number, False if it's one digit
# * The script should hopefully still work for single digit numbers even if multi-digit recognition is activated
# * Change the global constant "folderpath" to the directory of this file
# * This probably will never happen, but make sure whichever directory you choose has no file that has "(temp_mnist)" in its name.
# * Run the cell below. 
# * When the tkinter window opens, draw a single digit number by holding left-click on mouse
# * To erase, hold right-click
# * When you finish drawing your number, click the save button first
# * Then click the test button second, and the jupyter cell below will output it's guess at the bottom.
# * You don't need to close out your tkinter window to reuse it. Just erase your drawing, draw a new number, click save, and then click test. The next guess will appear under the previously outputted guess.

# In[8]:


MULTI_DIGIT = True

factor = 10
compartments = 28
main_width = compartments * factor
main_height = compartments * factor
brush_size = 1 * factor
INSPECTION = not True

def retrieve_saved_image():
    for filename in os.listdir(folderpath):
        if filename.split(".")[-1] in ["jpeg", "jpg"] and "(temp_mnist)" in filename:
            # there is only one image file, conv% temp
            image_np_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            assert image_np_array.shape == (28, 28)
            image_np_array = np.expand_dims(image_np_array, axis=2).astype("uint8")
            assert image_np_array.shape == (28, 28, 1)
            return image_np_array
    

def test_convtemp_image(image_np_array, model):
    # descr: tests the image that you just created
    if len(image_np_array.shape) == 2:
        image_np_array = np.expand_dims(image_np_array, axis=2).astype("uint8")
    image_singleton = [image_np_array]
    prediction_singleton = model.predict(np.array(image_singleton))
    pred_label = np.argmax(prediction_singleton[0])
    return pred_label


black_max = "#282828"
white_min = "#d7d7d7"
ink = "#282828";
HEX_CONVERTER = list("0123456789abcdef")
len_HEX_CONVERTER = len(HEX_CONVERTER)

class Canvas_Pixel:
    def __init__(self, hex_, corners):
        self.hex_ = hex_
        self.corners = corners # [x0, y0, x1, y1]
    def __repr__(self):
        return str(self.corners) + ", " +  str(self.hex_)

def set_up_present_labels():
    # labels are a tkinter-related thing. This just sets them up beforehand
    compartment_width = main_width // compartments # aka 100, if main_width = 500 and compartments = 5
    compartment_height = main_height // compartments

    present_labels = []
    for y in range(0, compartments):
        row = []
        for x in range(0, compartments):
            corners = (x * (compartment_width),
                       y * (compartment_height),
                       x * (compartment_width) + (compartment_width) - 1,
                       y * (compartment_height) + (compartment_height) -1)
            canvas_pixel = Canvas_Pixel("#ffffff", corners)
            row.append(canvas_pixel)
        present_labels.append(row)

    return present_labels
present_labels = set_up_present_labels()


# DRAWING RELATED FUNCTIONS
def hexify_channel(channel_hex):
    # converts each c
    assert len(channel_hex) == 2
    channel_hex = channel_hex.lower()
    part1,part2 = list(channel_hex)
    channel_part1 = HEX_CONVERTER.index(part1)
    channel_part2 = HEX_CONVERTER.index(part2)
    value = (channel_part1 * len_HEX_CONVERTER) + channel_part2 
    return value
def hex_to_rgb(hex_str):
    # input: hex_str, the string hex for the color, like #282828, etc
    # output: list<int> with 3 ints ranging from 0 to 255, for the RGB Channels
    hex_str = hex_str[1:] # remove pound symnbol
    assert len(hex_str) == 6
    rgb_values = []
    for i in range(0, 6, 2):
        channel_hex = hex_str[i:i + 2]
        channel_value = hexify_channel(channel_hex)
        rgb_values.append(channel_value)
    return rgb_values
def rgb_to_hex(rgb):
    # input: rgb, list<int> of len 3. Each int ranges from 0 to 255 as the RGB Channels
    # output: string
    hex_str = "#"
    for channel_value in rgb:
        part1 = channel_value // len_HEX_CONVERTER
        part2 = channel_value % len_HEX_CONVERTER
        channel_part1 = HEX_CONVERTER[part1]
        channel_part2 = HEX_CONVERTER[part2]
        hex_str = hex_str + channel_part1 + channel_part2
    return hex_str
def color_opacity(color_hex_str, direction = True):
    # descr: increments/decrements a color in terms of opacity
    # input: color_hex_str, a string that's a given color in hex form
    # input: direction: the direction the opacity is going towards,
    # where towards black = false, towards white = true

    f = 10 # factor of adjustment for increase/decrease. So Black is -10, white is +10
    rgb_form = hex_to_rgb(color_hex_str)
    r, g, b = rgb_form
    
    # brightening/ +10 / towards white
    if direction == True:
        new_rgb_form = [min([255, r + f]),
                        min([255, g + f]),
                        min([255, b + f])]
    # increases the channels, but ceilinged at 255
    # darkening/ -10 / towards black
    else:
        new_rgb_form = [max([0, r - f]),
                        max([0, g - f]),
                        max([0, b - f])]
    new_color_hex_str = rgb_to_hex(new_rgb_form)
    return new_color_hex_str
def color_update(corners, direction):
    # descr: changes the color of the pixel you are coloring
    # for example, if you are painting black over an area that's already black, it will be an even darker black 
    # (to give the illusion of opacity), until it stops at the darkest black possible.
    # or, if you are painting white over a black area, the area is set to the least bright white allowed, and can 
    # only be brighter if you paint white over it again. Vice versa for black.
    # input: corners, tuple of 4 ints that mark the coordinates for the corners of the area of pixels 
    # painted in 1 brushstroke on the canvas,
    # each int ranges from 0 to canvas sidelength, since they are coordinates
    # input: 
    # outputs new color, a hex_string
    curr_canvas_pixel = present_labels[corners[1]//100][corners[0]//100]
    old_color = curr_canvas_pixel.hex_
    old_rgb = hex_to_rgb(old_color)
    # deciding new color: if same direction, color_opacity,
    # if other direction, new edge color
    old_direction = float(old_rgb[0]) > 127.5 # true if above, false if below
    if direction == old_direction:
        new_color = color_opacity(old_color, direction)
    else:
        # diverge
        if direction == True: # if diverging to white
            new_color = white_min
        else: # if diverging to black
            new_color = black_max
    present_labels[corners[1]//10][corners[0]//10] = Canvas_Pixel(new_color, corners)
    return new_color
def floored_corners_paint(eventx, eventy):
    # paints on the canvas
    compartment_width = main_width // compartments # aka 100, if main_width = 500 and compartments = 5
    compartment_height = main_height // compartments
    event_x = max([0, min([main_width - 1, eventx])]) # to prevent going off camera
    event_y = max([0, min([main_height - 1, eventy])])
    assert event_y >= 0 and event_y < main_height
    assert event_x >= 0 and event_x < main_width
    anchor_x0 = ((event_x // brush_size) * compartment_width)
    anchor_y0 = ((event_y // brush_size) * compartment_height)
    anchor_x1 = ((event_x // brush_size) * compartment_width) + compartment_width
    anchor_y1 = ((event_y // brush_size) * compartment_height) + compartment_height
    corners = (anchor_x0, anchor_y0, anchor_x1 - 1, anchor_y1 - 1)
    return corners
def painter(event):
    # descr: paints the canvas (ink is black)
    corners = floored_corners_paint(event.x, event.y)
    # deciding new color:
    new_color = color_update(corners, False)
    canvas.create_rectangle(corners, fill=new_color, width=1, outline=new_color)
def eraser(event):
    # descr: erases the canvas (simply switches the ink to white)
    corners = floored_corners_paint(event.x, event.y)
    new_color = color_update(corners, True)
    canvas.create_rectangle(corners, fill=new_color, width=1, outline=new_color)
def start_paint(event):
    # descr: binds the paint action to holding left click on mouse
    canvas.bind('<B1-Motion>', painter)
def start_erase(event):
    # descr: binds the paint action to holding right click on mouse
    canvas.bind('<B3-Motion>', eraser)
    lastx, lasty = event.x, event.y
# END OF DRAWING RELATED FUNCTIONS
    
def tester():
    # descr: tester function that activates when the test button is clicked
    # the image on the canvas is fed to the model, and the guessed number is printed under this jupyter cell
    global model
    image_np_array = retrieve_saved_image()
    if MULTI_DIGIT == True:
        answer = multidigit_test_convtemp_image(image_np_array, model, visible = INSPECTION)
    else:
        answer = test_convtemp_image(image_np_array, model)
    print("Is this {0}?".format(str(answer)))

root = tk.Tk()
def save():
    # descr: saves the image you just created on your canvas so its file will be eventually tested by the model
    filename = "conv (temp_mnist)" + ".jpg"

    compartment_width = main_width // compartments # aka 100, if main_width = 500 and compartments = 5
    compartment_height = main_height // compartments
    
    image_matrix = []
    for y in range(0, compartments):
        image_row = []
        for x in range(0, compartments):
            canvas_pixel = present_labels[y][x]
            image_row.append(list(hex_to_rgb(canvas_pixel.hex_)))
        image_matrix.append(image_row)
    image_array = np.array(image_matrix)
    assert image_array.shape == (28, 28, 3)
    image = Image.fromarray(image_array.astype("uint8"))
    for oldfilename in os.listdir(folderpath):
        if "(temp_mnist)" in oldfilename:
            os.remove(folderpath + oldfilename)
    image.save(folderpath + filename)

button_tester = Button(text = "Test", command = tester)
button_tester.pack()
canvas = Canvas(root, width = main_width, height = main_height, bg='white')
canvas.bind('<1>', start_paint)
canvas.bind('<3>', start_erase)
canvas.pack(expand=YES, fill=BOTH)
btn_save = Button(text="save", command=save)
btn_save.pack()
root.mainloop()


# In[ ]:




