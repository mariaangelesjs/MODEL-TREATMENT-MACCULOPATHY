#!/usr/bin/python
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import skimage
from skimage.color import rgb2gray
import numpy as np
from numpy.fft import fft2
from numpy import fft
from numpy.fft import fftn
import os
import cv2
import glob
import cv2
import numpy as np
import pygame 
import imageio
import imshowpair
from PIL import Image, ImageDraw, ImageFont,ImageChops
from mpl_toolkits.mplot3d import Axes3D
import glob
import math
import sympy
import xarray
from scipy import signal, interpolate, optimize
from scipy.optimize import fmin
from scipy.signal import correlate2d,convolve
from os.path import isfile
import sys
import matplotlib.colors as mplcl
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#KELLY EQUATION
def S(x, y):
    abs_value = math.log10(y/3*x)
    within_exponent = (((y/x)+3)/45.9)
    return (6.1 + 7.3*((math.fabs(abs_value))**3))*y*x*math.exp(-2*x*within_exponent)
x = np.linspace(0.01, 65, 25)
y = np.linspace(0.01, 65, 1152)
X, Y = np.meshgrid(x, y)
f2 = np.vectorize(S)
Z= f2(X, Y)
print(X.shape)
print(Y.shape)
print(Z.shape)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, linewidth = 0, antialiased = False, cmap = 'viridis', edgecolor = 'none')
ax.set_xlabel('Spatial Frequency (cycles/°)')
ax.set_ylabel('Temporal frequency(cycles/s)')
ax.set_zlabel('Sensitivity (AU)')
ax.set_xlim(X.min(),X.max())
ax.set_ylim(Y.min(),Y.max())
ax.set_zlim3d(1e1,Z.max())
ax.set_xscale('symlog')
ax.set_yscale('symlog')
ax.set_zscale('symlog')
plt.tight_layout()
plt.savefig('Kelly equation.png')
plt.show()

#save pygame background
screen = pygame.display.set_mode((1152, 870))
background = pygame.image.load('pygame background 2.png').convert()
screen.blit(background, (0, 0))
pygame.display.update()
pygame.image.save(screen,'pygame background.png')

img = Image.new('RGBA', (1152,870), (255, 0, 0, 0))
img.save('test.png', 'PNG')
color_white = (255, 255, 255, 255)
color_black = (0, 0, 0, 255)
def save_landolt_ring(color,output):
    background = Image.open('test.png')
    draw = ImageDraw.Draw(background)
    # desired size
    font = ImageFont.truetype('Sloan.otf', size=45)
    message = "C"
    color = color
    W, H = (1152,870)
    w, h = draw.textsize(message)
    draw.text(((W-w)/2,(H-h)/2), message, fill=color, font=font)
    # save the edited image
    return background.save(output)
save_landolt_ring(color_white,'Landolt_ring_white.png')
save_landolt_ring(color_black,'Landolt_ring_black.png')
#do concadenated image
white_1 = Image.open('45_white_frame1_10_deg.png')
white_2 = Image.open('45_white_frame6_10_deg.png')
white_2 = Image.blend(white_1,white_2,alpha = 0.5)
white_3 = Image.open('45_white_frame11_10_deg.png')
white_3 = Image.blend(white_2,white_3,alpha = 0.5)
white_4 = Image.open('45_white_frame16_10_deg.png')
white_4 = Image.blend(white_3,white_4,alpha = 0.5)
white_5 = Image.open('45_white_frame21_10_deg.png')
white_5 = Image.blend(white_4,white_5,alpha = 0.5)
#save the edited image
white_5.save('MATERIALS AND METHODS.png')
#do concadenated image
white_1 = Image.open('ds_white_frame1_10_deg.png')
white_2 = Image.open('ds_white_frame6_10_deg.png')
white_2 = Image.blend(white_1,white_2,alpha = 0.5)
white_3 = Image.open('ds_white_frame11_10_deg.png')
white_3 = Image.blend(white_2,white_3,alpha = 0.5)
white_4 = Image.open('ds_white_frame16_10_deg.png')
white_4 = Image.blend(white_3,white_4,alpha = 0.5)
white_5 = Image.open('ds_white_frame21_10_deg.png')
white_5 = Image.blend(white_4,white_5,alpha = 0.5)

white_5.save('MATERIALS AND METHODS 1.png')

def rotate_landolt_ring(input,output1,output2):
    landolt_ring = Image.open(input)
    image_rot_90 = landolt_ring.rotate(45)
    image_rot_90.save(output1)
    image_rot_180 = landolt_ring.rotate(315,translate= (24,-28))
    image_rot_180.save(output2)
rotate_landolt_ring('Landolt_ring_black.png','black_lr_rot_45.png','black_lr_rot_315.png')
rotate_landolt_ring('Landolt_ring_white.png','white_lr_rot_45.png','white_lr_rot_315.png')
#Difference spectrum
# Open the images

image1_black = Image.open('black_lr_rot_45.png');
image2_black = Image.open('black_lr_rot_315.png')
image1_white = Image.open('white_lr_rot_45.png');
image2_white = Image.open('white_lr_rot_315.png')
# Get the image buffer as ndarray
buffer1_black    = np.asarray(image1_black);
buffer2_black   = np.asarray(image2_black);
buffer1_white    = np.asarray(image1_white);
buffer2_white   = np.asarray(image2_white)
# Subtract image2 from image1
buffer3_black = ImageChops.difference(image1_black,image2_black)
buffer3_white = ImageChops.difference(image1_white,image2_white)
# Construct a new Image from the resultant buffer
differenceImage_white     = (buffer3_white)
differenceImage_black     = (buffer3_black)
differenceImage_white.save('ds_white.png')
differenceImage_black.save('ds_black.png')

 # change everything to white where pixel is not black
def copy(image_to_copy,output_path):
            img = Image.open(image_to_copy)
            img1 = img.copy()
            return img1.save(output_path)

#PLAY GAME AND SAVE IMAGES
def move_landolt_ring(input,coordinate_x,coordinate_y,output):
    screen = pygame.display.set_mode((1152, 870))
    player = pygame.image.load(input)
    background = pygame.image.load('pygame background 2.png').convert()
    screen.blit(background, (0, 0))
    position = player.get_rect()
    screen.blit(player, position)          #draw the player
    pygame.display.update()
    for x in range(1):  
        screen.blit(background, position, position) #erase
        position = position.move(coordinate_x,coordinate_y)    #move player
        screen.blit(player, position)      #draw new player
        pygame.display.update()            #and show it all
        pygame.image.save(screen,output)
#and show it all
#0 deg/s
#a.45 Landolt ring
frame1 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame1_0.png')
frame2 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame2_0.png')
frame3 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame3_0.png')
frame4 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame4_0.png')
frame5 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame5_0.png')
frame6_w = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame6_0.png')
frame6_b = move_landolt_ring('black_lr_rot_45.png',512,0,'45_black_frame6_0.png')
frame7_w = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame7_0.png')
frame7_b = move_landolt_ring('black_lr_rot_45.png',512,0,'45_black_frame7_0.png')
frame8_w = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame8_0.png')
frame8_b = move_landolt_ring('black_lr_rot_45.png',512,0,'45_black_frame8_0.png')
frame9_w = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame9_0.png')
frame9_b = move_landolt_ring('black_lr_rot_45.png',512,0,'45_black_frame9_0.png')
frame10_w = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame10_0.png')
frame10_b = move_landolt_ring('black_lr_rot_45.png',512,0,'45_black_frame10_0.png')
frame11 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame11_0.png')
frame12 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame12_0.png')
frame13 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame13_0.png')
frame14 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame14_0.png')
frame15 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame15_0.png')
frame16_w = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame16_0.png')
frame16_b = move_landolt_ring('black_lr_rot_45.png',512,0,'45_black_frame16_0.png')
frame17_w = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame17_0.png')
frame17_b = move_landolt_ring('black_lr_rot_45.png',512,0,'45_black_frame17_0.png')
frame18_w = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame18_0.png')
frame18_b = move_landolt_ring('black_lr_rot_45.png',512,0,'45_black_frame18_0.png')
frame19_w = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame19_0.png')
frame19_b = move_landolt_ring('black_lr_rot_45.png',512,0,'45_black_frame19_0.png')
frame20_w = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame20_0.png')
frame20_b = move_landolt_ring('black_lr_rot_45.png',512,0,'45_black_frame20_0.png')
frame21 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame21_0.png')
frame22 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame22_0.png')
frame23 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame23_0.png')
frame24 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame24_0.png')
frame25 = move_landolt_ring('white_lr_rot_45.png',512,0,'45_white_frame25_0.png')
#b.315 Landolt ring
frame1 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame1_0.png')
frame2 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame2_0.png')
frame3 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame3_0.png')
frame4 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame4_0.png')
frame5 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame5_0.png')
frame6_w = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame6_0.png')
frame6_b = move_landolt_ring('black_lr_rot_315.png',512,0,'315_black_frame6_0.png')
frame7_w = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame7_0.png')
frame7_b = move_landolt_ring('black_lr_rot_315.png',512,0,'315_black_frame7_0.png')
frame8_w = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame8_0.png')
frame8_b = move_landolt_ring('black_lr_rot_315.png',512,0,'315_black_frame8_0.png')
frame9_w = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame9_0.png')
frame9_b = move_landolt_ring('black_lr_rot_315.png',512,0,'315_black_frame9_0.png')
frame10_w = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame10_0.png')
frame10_b = move_landolt_ring('black_lr_rot_315.png',512,0,'315_black_frame10_0.png')
frame11 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame11_0.png')
frame12 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame12_0.png')
frame13 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame13_0.png')
frame14 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame14_0.png')
frame15 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame15_0.png')
frame16_w = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame16_0.png')
frame16_b = move_landolt_ring('black_lr_rot_315.png',512,0,'315_black_frame16_0.png')
frame17_w = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame17_0.png')
frame17_b = move_landolt_ring('black_lr_rot_315.png',512,0,'315_black_frame17_0.png')
frame18_w = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame18_0.png')
frame18_b = move_landolt_ring('black_lr_rot_315.png',512,0,'315_black_frame18_0.png')
frame19_w = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame19_0.png')
frame19_b = move_landolt_ring('black_lr_rot_315.png',512,0,'315_black_frame19_0.png')
frame20_w = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame20_0.png')
frame20_b = move_landolt_ring('black_lr_rot_315.png',512,0,'315_black_frame20_0.png')
frame21 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame21_0.png')
frame22 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame22_0.png')
frame23 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame23_0.png')
frame24 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame24_0.png')
frame25 = move_landolt_ring('white_lr_rot_315.png',512,0,'315_white_frame25_0.png')
#Difference spectrum
frame1 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame1_0.png')
frame2 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame2_0.png')
frame3 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame3_0.png')
frame4 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame4_0.png')
frame5 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame5_0.png')
frame6_w = move_landolt_ring('ds_white.png',512,0,'ds_white_frame6_0.png')
frame6_b = move_landolt_ring('ds_black.png',512,0,'ds_black_frame6_0.png')
frame7_w = move_landolt_ring('ds_white.png',512,0,'ds_white_frame7_0.png')
frame7_b = move_landolt_ring('ds_black.png',512,0,'ds_black_frame7_0.png')
frame8_w = move_landolt_ring('ds_white.png',512,0,'ds_white_frame8_0.png')
frame8_b = move_landolt_ring('ds_black.png',512,0,'ds_black_frame8_0.png')
frame9_w = move_landolt_ring('ds_white.png',512,0,'ds_white_frame9_0.png')
frame9_b = move_landolt_ring('ds_black.png',512,0,'ds_black_frame9_0.png')
frame10_w = move_landolt_ring('ds_white.png',512,0,'ds_white_frame10_0.png')
frame10_b = move_landolt_ring('ds_black.png',512,0,'ds_black_frame10_0.png')
frame11 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame11_0.png')
frame12 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame12_0.png')
frame13 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame13_0.png')
frame14 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame14_0.png')
frame15 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame15_0.png')
frame16_w = move_landolt_ring('ds_white.png',512,0,'ds_white_frame16_0.png')
frame16_b = move_landolt_ring('ds_black.png',512,0,'ds_black_frame16_0.png')
frame17_w = move_landolt_ring('ds_white.png',512,0,'ds_white_frame17_0.png')
frame17_b = move_landolt_ring('ds_black.png',512,0,'ds_black_frame17_0.png')
frame18_w = move_landolt_ring('ds_white.png',512,0,'ds_white_frame18_0.png')
frame18_b = move_landolt_ring('ds_black.png',512,0,'ds_black_frame18_0.png')
frame19_w = move_landolt_ring('ds_white.png',512,0,'ds_white_frame19_0.png')
frame19_b = move_landolt_ring('ds_black.png',512,0,'ds_black_frame19_0.png')
frame20_w = move_landolt_ring('ds_white.png',512,0,'ds_white_frame20_0.png')
frame20_b = move_landolt_ring('ds_black.png',512,0,'ds_black_frame20_0.png')
frame21 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame21_0.png')
frame22 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame22_0.png')
frame23 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame23_0.png')
frame24 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame24_0.png')
frame25 = move_landolt_ring('ds_white.png',512,0,'ds_white_frame25_0.png')
#1.25 deg/s
#a.45 Landolt ring
frame1 = move_landolt_ring('white_lr_rot_45.png',511.889115,9.947656809,'45_white_frame1_125.png')
frame2 = move_landolt_ring('white_lr_rot_45.png',511.9061464,9.151945751,'45_white_frame2_125.png')
frame3 = move_landolt_ring('white_lr_rot_45.png',511.9217587,8.356209323,'45_white_frame3_125.png')
frame4 = move_landolt_ring('white_lr_rot_45.png',511.9359518,7.560449728,'45_white_frame4_125.png')
frame5 = move_landolt_ring('white_lr_rot_45.png',511.9487258,6.764669175,'45_white_frame5_125.png')
frame6_w = move_landolt_ring('white_lr_rot_45.png',511.9600805,5.968869869,'45_white_frame6_125.png')
frame6_b = move_landolt_ring('black_lr_rot_45.png',511.9600805,5.968869869,'45_black_frame6_125.png')
frame7_w = move_landolt_ring('white_lr_rot_45.png',511.9700159,5.173054015,'45_white_frame7_125.png')
frame7_b = move_landolt_ring('black_lr_rot_45.png',511.9700159,5.173054015,'45_black_frame7_125.png')
frame8_w = move_landolt_ring('white_lr_rot_45.png',511.978532,4.377223821,'45_white_frame8_125.png')
frame8_b = move_landolt_ring('black_lr_rot_45.png',511.978532,4.377223821,'45_black_frame8_125.png')
frame9_w = move_landolt_ring('white_lr_rot_45.png',511.9856289,3.581381491,'45_white_frame9_125.png')
frame9_b = move_landolt_ring('black_lr_rot_45.png',511.9856289,3.581381491,'45_black_frame9_125.png')
frame10_w = move_landolt_ring('white_lr_rot_45.png',511.9913063,2.785529234,'45_white_frame10_125.png')
frame10_b = move_landolt_ring('black_lr_rot_45.png',511.9913063,2.785529234,'45_black_frame10_125.png')
frame11 = move_landolt_ring('white_lr_rot_45.png',511.9955644,1.989669254,'45_white_frame11_125.png')
frame12 = move_landolt_ring('white_lr_rot_45.png',511.9984032,1.193803759,'45_white_frame12_125.png')
frame13 = move_landolt_ring('white_lr_rot_45.png',511.9998226,0.397934954,'45_white_frame13_125.png')
frame14 = move_landolt_ring('white_lr_rot_45.png',511.9984032,-1.193803759,'45_white_frame14_125.png')
frame15 = move_landolt_ring('white_lr_rot_45.png',511.9955644,-1.989669254,'45_white_frame15_125.png')
frame16_w = move_landolt_ring('white_lr_rot_45.png',511.9913063,-2.785529234,'45_white_frame16_125.png')
frame16_b = move_landolt_ring('black_lr_rot_45.png',511.9913063,-2.785529234,'45_black_frame16_125.png')
frame17_w = move_landolt_ring('white_lr_rot_45.png',511.9856289,-3.581381491,'45_white_frame17_125.png')
frame17_b = move_landolt_ring('black_lr_rot_45.png',511.9856289,-3.581381491,'45_black_frame17_125.png')
frame18_w = move_landolt_ring('white_lr_rot_45.png',511.978532,-4.377223821,'45_white_frame18_125.png')
frame18_b = move_landolt_ring('black_lr_rot_45.png',511.978532,-4.377223821,'45_black_frame18_125.png')
frame19_w = move_landolt_ring('white_lr_rot_45.png',511.9700159,-5.173054015,'45_white_frame19_125.png')
frame19_b = move_landolt_ring('black_lr_rot_45.png',511.9700159,-5.173054015,'45_black_frame19_125.png')
frame20_w = move_landolt_ring('white_lr_rot_45.png',511.9600805,-5.968869869,'45_white_frame20_125.png')
frame20_b = move_landolt_ring('black_lr_rot_45.png',511.9600805,-5.968869869,'45_black_frame20_125.png')
frame21 = move_landolt_ring('white_lr_rot_45.png',511.9487258,-6.764669175,'45_white_frame21_125.png')
frame22 = move_landolt_ring('white_lr_rot_45.png',511.9359518,-7.560449728,'45_white_frame22_125.png')
frame23 = move_landolt_ring('white_lr_rot_45.png',511.9217587,-8.356209323,'45_white_frame23_125.png')
frame24 = move_landolt_ring('white_lr_rot_45.png',511.9061464,-9.151945751,'45_white_frame24_125.png')
frame25 = move_landolt_ring('white_lr_rot_45.png',511.889115,-9.947656809,'45_white_frame25_125.png')
#b.315 Landolt ring
frame1 = move_landolt_ring('white_lr_rot_315.png',511.889115,9.947656809,'315_white_frame1_125.png')
frame2 = move_landolt_ring('white_lr_rot_315.png',511.9061464,9.151945751,'315_white_frame2_125.png')
frame3 = move_landolt_ring('white_lr_rot_315.png',511.9217587,8.356209323,'315_white_frame3_125.png')
frame4 = move_landolt_ring('white_lr_rot_315.png',511.9359518,7.560449728,'315_white_frame4_125.png')
frame5 = move_landolt_ring('white_lr_rot_315.png',511.9487258,6.764669175,'315_white_frame5_125.png')
frame6_w = move_landolt_ring('white_lr_rot_315.png',511.9600805,5.968869869,'315_white_frame6_125.png')
frame6_b = move_landolt_ring('black_lr_rot_315.png',511.9600805,5.968869869,'315_black_frame6_125.png')
frame7_w = move_landolt_ring('white_lr_rot_315.png',511.9700159,5.173054015,'315_white_frame7_125.png')
frame7_b = move_landolt_ring('black_lr_rot_315.png',511.9700159,5.173054015,'315_black_frame7_125.png')
frame8_w = move_landolt_ring('white_lr_rot_315.png',511.978532,4.377223821,'315_white_frame8_125.png')
frame8_b = move_landolt_ring('black_lr_rot_315.png',511.978532,4.377223821,'315_black_frame8_125.png')
frame9_w = move_landolt_ring('white_lr_rot_315.png',511.9856289,3.581381491,'315_white_frame9_125.png')
frame9_b = move_landolt_ring('black_lr_rot_315.png',511.9856289,3.581381491,'315_black_frame9_125.png')
frame10_w = move_landolt_ring('white_lr_rot_315.png',511.9913063,2.785529234,'315_white_frame10_125.png')
frame10_b = move_landolt_ring('black_lr_rot_315.png',511.9913063,2.785529234,'315_black_frame10_125.png')
frame11 = move_landolt_ring('white_lr_rot_315.png',511.9955644,1.989669254,'315_white_frame11_125.png')
frame12 = move_landolt_ring('white_lr_rot_315.png',511.9984032,1.193803759,'315_white_frame12_125.png')
frame13 = move_landolt_ring('white_lr_rot_315.png',511.9998226,0.397934954,'315_white_frame13_125.png')
frame14 = move_landolt_ring('white_lr_rot_315.png',511.9984032,-1.193803759,'315_white_frame14_125.png')
frame15 = move_landolt_ring('white_lr_rot_315.png',511.9955644,-1.989669254,'315_white_frame15_125.png')
frame16_w = move_landolt_ring('white_lr_rot_315.png',511.9913063,-2.785529234,'315_white_frame16_125.png')
frame16_b = move_landolt_ring('black_lr_rot_315.png',511.9913063,-2.785529234,'315_black_frame16_125.png')
frame17_w = move_landolt_ring('white_lr_rot_315.png',511.9856289,-3.581381491,'315_white_frame17_125.png')
frame17_b = move_landolt_ring('black_lr_rot_315.png',511.9856289,-3.581381491,'315_black_frame17_125.png')
frame18_w = move_landolt_ring('white_lr_rot_315.png',511.978532,-4.377223821,'315_white_frame18_125.png')
frame18_b = move_landolt_ring('black_lr_rot_315.png',511.978532,-4.377223821,'315_black_frame18_125.png')
frame19_w = move_landolt_ring('white_lr_rot_315.png',511.9700159,-5.173054015,'315_white_frame19_125.png')
frame19_b = move_landolt_ring('black_lr_rot_315.png',511.9700159,-5.173054015,'315_black_frame19_125.png')
frame20_w = move_landolt_ring('white_lr_rot_315.png',511.9600805,-5.968869869,'315_white_frame20_125.png')
frame20_b = move_landolt_ring('black_lr_rot_315.png',511.9600805,-5.968869869,'315_black_frame20_125.png')
frame21 = move_landolt_ring('white_lr_rot_315.png',511.9487258,-6.764669175,'315_white_frame21_125.png')
frame22 = move_landolt_ring('white_lr_rot_315.png',511.9359518,-7.560449728,'315_white_frame22_125.png')
frame23 = move_landolt_ring('white_lr_rot_315.png',511.9217587,-8.356209323,'315_white_frame23_125.png')
frame24 = move_landolt_ring('white_lr_rot_315.png',511.9061464,-9.151945751,'315_white_frame24_125.png')
frame25 = move_landolt_ring('white_lr_rot_315.png',511.889115,-9.947656809,'315_white_frame25_125.png')
#c.Difference spectrum
frame1 = move_landolt_ring('ds_white.png',511.889115,9.947656809,'ds_white_frame1_125.png')
frame2 = move_landolt_ring('ds_white.png',511.9061464,9.151945751,'ds_white_frame2_125.png')
frame3 = move_landolt_ring('ds_white.png',511.9217587,8.356209323,'ds_white_frame3_125.png')
frame4 = move_landolt_ring('ds_white.png',511.9359518,7.560449728,'ds_white_frame4_125.png')
frame5 = move_landolt_ring('ds_white.png',511.9487258,6.764669175,'ds_white_frame5_125.png')
frame6_w = move_landolt_ring('ds_white.png',511.9600805,5.968869869,'ds_white_frame6_125.png')
frame6_b = move_landolt_ring('ds_black.png',511.9600805,5.968869869,'ds_black_frame6_125.png')
frame7_w = move_landolt_ring('ds_white.png',511.9700159,5.173054015,'ds_white_frame7_125.png')
frame7_b = move_landolt_ring('ds_black.png',511.9700159,5.173054015,'ds_black_frame7_125.png')
frame8_w = move_landolt_ring('ds_white.png',511.978532,4.377223821,'ds_white_frame8_125.png')
frame8_b = move_landolt_ring('ds_black.png',511.978532,4.377223821,'ds_black_frame8_125.png')
frame9_w = move_landolt_ring('ds_white.png',511.9856289,3.581381491,'ds_white_frame9_125.png')
frame9_b = move_landolt_ring('ds_black.png',511.9856289,3.581381491,'ds_black_frame9_125.png')
frame10_w = move_landolt_ring('ds_white.png',511.9913063,2.785529234,'ds_white_frame10_125.png')
frame10_b = move_landolt_ring('ds_black.png',511.9913063,2.785529234,'ds_black_frame10_125.png')
frame11 = move_landolt_ring('ds_white.png',511.9955644,1.989669254,'ds_white_frame11_125.png')
frame12 = move_landolt_ring('ds_white.png',511.9984032,1.193803759,'ds_white_frame12_125.png')
frame13 = move_landolt_ring('ds_white.png',511.9998226,0.397934954,'ds_white_frame13_125.png')
frame14 = move_landolt_ring('ds_white.png',511.9984032,-1.193803759,'ds_white_frame14_125.png')
frame15 = move_landolt_ring('ds_white.png',511.9955644,-1.989669254,'ds_white_frame15_125.png')
frame16_w = move_landolt_ring('ds_white.png',511.9913063,-2.785529234,'ds_white_frame16_125.png')
frame16_b = move_landolt_ring('ds_black.png',511.9913063,-2.785529234,'ds_black_frame16_125.png')
frame17_w = move_landolt_ring('ds_white.png',511.9856289,-3.581381491,'ds_white_frame17_125.png')
frame17_b = move_landolt_ring('ds_black.png',511.9856289,-3.581381491,'ds_black_frame17_125.png')
frame18_w = move_landolt_ring('ds_white.png',511.978532,-4.377223821,'ds_white_frame18_125.png')
frame18_b = move_landolt_ring('ds_black.png',511.978532,-4.377223821,'ds_black_frame18_125.png')
frame19_w = move_landolt_ring('ds_white.png',511.9700159,-5.173054015,'ds_white_frame19_125.png')
frame19_b = move_landolt_ring('ds_black.png',511.9700159,-5.173054015,'ds_black_frame19_125.png')
frame20_w = move_landolt_ring('ds_white.png',511.9600805,-5.968869869,'ds_white_frame20_125.png')
frame20_b = move_landolt_ring('ds_black.png',511.9600805,-5.968869869,'ds_black_frame20_125.png')
frame21 = move_landolt_ring('ds_white.png',511.9487258,-6.764669175,'ds_white_frame21_125.png')
frame22 = move_landolt_ring('ds_white.png',511.9359518,-7.560449728,'ds_white_frame22_125.png')
frame23 = move_landolt_ring('ds_white.png',511.9217587,-8.356209323,'ds_white_frame23_125.png')
frame24 = move_landolt_ring('ds_white.png',511.9061464,-9.151945751,'ds_white_frame24_125.png')
frame25 = move_landolt_ring('ds_white.png',511.889115,-9.947656809,'ds_white_frame25_125.png')
#2.5 deg/s
#a.45
frame1 = move_landolt_ring('white_lr_rot_45.png',511.556508,19.89100484,'45_white_frame1_2_5.png')
frame2 = move_landolt_ring('white_lr_rot_45.png',511.6246201,18.30053626,'45_white_frame2_2_5.png')
frame3 = move_landolt_ring('white_lr_rot_45.png',511.6870588,16.70986474,'45_white_frame3_2_5.png')
frame4 = move_landolt_ring('white_lr_rot_45.png',511.7438234,15.11900792,'45_white_frame4_2_5.png')
frame5 = move_landolt_ring('white_lr_rot_45.png',511.7949134,13.52798346,'45_white_frame5_2_5.png')
frame6_w = move_landolt_ring('white_lr_rot_45.png',511.8403281,11.93680898,'45_white_frame6_2_5.png')
frame6_b = move_landolt_ring('black_lr_rot_45.png',511.8403281,11.93680898,'45_black_frame6_2_5.png')
frame7_w = move_landolt_ring('white_lr_rot_45.png',511.8800671,10.34550213,'45_white_frame7_2_5.png')
frame7_b = move_landolt_ring('black_lr_rot_45.png',511.8800671,10.34550213,'45_black_frame7_2_5.png')
frame8_w = move_landolt_ring('white_lr_rot_45.png',511.91413,8.75408057,'45_white_frame8_2_5.png')
frame8_b = move_landolt_ring('black_lr_rot_45.png',511.91413,8.75408057,'45_black_frame8_2_5.png')
frame9_w = move_landolt_ring('white_lr_rot_45.png',511.9425162,7.162561934,'45_white_frame9_2_5.png')
frame9_b = move_landolt_ring('black_lr_rot_45.png',511.9425162,7.162561934,'45_black_frame9_2_5.png')
frame10_w = move_landolt_ring('white_lr_rot_45.png',511.9652256,5.570963872,'45_white_frame10_2_5.png')
frame10_b = move_landolt_ring('black_lr_rot_45.png',511.9652256,5.570963872,'45_black_frame10_2_5.png')
frame11 = move_landolt_ring('white_lr_rot_45.png',511.9822579,3.979304035,'45_white_frame11_2_5.png')
frame12 = move_landolt_ring('white_lr_rot_45.png',511.9936128,2.387600072,'45_white_frame12_2_5.png')
frame13 = move_landolt_ring('white_lr_rot_45.png',511.9992903,0.795869632,'45_white_frame13_2_5.png')
frame14 = move_landolt_ring('white_lr_rot_45.png',511.9936128,-2.387600072,'45_white_frame14_2_5.png')
frame15 = move_landolt_ring('white_lr_rot_45.png',511.9822579,-3.979304035,'45_white_frame15_2_5.png')
frame16_w = move_landolt_ring('white_lr_rot_45.png',511.9652256,-5.570963872,'45_white_frame16_2_5.png')
frame16_b = move_landolt_ring('black_lr_rot_45.png',511.9652256,-5.570963872,'45_black_frame16_2_5.png')
frame17_w = move_landolt_ring('white_lr_rot_45.png',511.9425162,-7.162561934,'45_white_frame17_2_5.png')
frame17_b = move_landolt_ring('black_lr_rot_45.png',511.9425162,-7.162561934,'45_black_frame17_2_5.png')
frame18_w = move_landolt_ring('white_lr_rot_45.png',511.91413,-8.75408057,'45_white_frame18_2_5.png')
frame18_b = move_landolt_ring('black_lr_rot_45.png',511.91413,-8.75408057,'45_black_frame18_2_5.png')
frame19_w = move_landolt_ring('white_lr_rot_45.png',511.8800671,-10.34550213,'45_white_frame19_2_5.png')
frame19_b = move_landolt_ring('black_lr_rot_45.png',511.8800671,-10.34550213,'45_black_frame19_2_5.png')
frame20_w = move_landolt_ring('white_lr_rot_45.png',511.8403281,-11.93680898,'45_white_frame20_2_5.png')
frame20_b = move_landolt_ring('black_lr_rot_45.png',511.8403281,-11.93680898,'45_black_frame20_2_5.png')
frame21 = move_landolt_ring('white_lr_rot_45.png',511.7949134,-13.52798346,'45_white_frame21_2_5.png')
frame22 = move_landolt_ring('white_lr_rot_45.png',511.7438234,-15.11900792,'45_white_frame22_2_5.png')
frame23 = move_landolt_ring('white_lr_rot_45.png',511.6870588,-16.70986474,'45_white_frame23_2_5.png')
frame24 = move_landolt_ring('white_lr_rot_45.png',511.6246201,-18.30053626,'45_white_frame24_2_5.png')
frame25 = move_landolt_ring('white_lr_rot_45.png',511.556508,-19.89100484,'45_white_frame25_2_5.png')
#b.315 Landolt ring
frame1 = move_landolt_ring('white_lr_rot_315.png',511.556508,19.89100484,'315_white_frame1_2_5.png')
frame2 = move_landolt_ring('white_lr_rot_315.png',511.6246201,18.30053626,'315_white_frame2_2_5.png')
frame3 = move_landolt_ring('white_lr_rot_315.png',511.6870588,16.70986474,'315_white_frame3_2_5.png')
frame4 = move_landolt_ring('white_lr_rot_315.png',511.7438234,15.11900792,'315_white_frame4_2_5.png')
frame5 = move_landolt_ring('white_lr_rot_315.png',511.7949134,13.52798346,'315_white_frame5_2_5.png')
frame6_w = move_landolt_ring('white_lr_rot_315.png',511.8403281,11.93680898,'315_white_frame6_2_5.png')
frame6_b = move_landolt_ring('black_lr_rot_315.png',511.8403281,11.93680898,'315_black_frame6_2_5.png')
frame7_w = move_landolt_ring('white_lr_rot_315.png',511.8800671,10.34550213,'315_white_frame7_2_5.png')
frame7_b = move_landolt_ring('black_lr_rot_315.png',511.8800671,10.34550213,'315_black_frame7_2_5.png')
frame8_w = move_landolt_ring('white_lr_rot_315.png',511.91413,8.75408057,'315_white_frame8_2_5.png')
frame8_b = move_landolt_ring('black_lr_rot_315.png',511.91413,8.75408057,'315_black_frame8_2_5.png')
frame9_w = move_landolt_ring('white_lr_rot_315.png',511.9425162,7.162561934,'315_white_frame9_2_5.png')
frame9_b = move_landolt_ring('black_lr_rot_315.png',511.9425162,7.162561934,'315_black_frame9_2_5.png')
frame10_w = move_landolt_ring('white_lr_rot_315.png',511.9652256,5.570963872,'315_white_frame10_2_5.png')
frame10_b = move_landolt_ring('black_lr_rot_315.png',511.9652256,5.570963872,'315_black_frame10_2_5.png')
frame11 = move_landolt_ring('white_lr_rot_315.png',511.9822579,3.979304035,'315_white_frame11_2_5.png')
frame12 = move_landolt_ring('white_lr_rot_315.png',511.9936128,2.387600072,'315_white_frame12_2_5.png')
frame13 = move_landolt_ring('white_lr_rot_315.png',511.9992903,0.795869632,'315_white_frame13_2_5.png')
frame14 = move_landolt_ring('white_lr_rot_315.png',511.9936128,-2.387600072,'315_white_frame14_2_5.png')
frame15 = move_landolt_ring('white_lr_rot_315.png',511.9822579,-3.979304035,'315_white_frame15_2_5.png')
frame16_w = move_landolt_ring('white_lr_rot_315.png',511.9652256,-5.570963872,'315_white_frame16_2_5.png')
frame16_b = move_landolt_ring('black_lr_rot_315.png',511.9652256,-5.570963872,'315_black_frame16_2_5.png')
frame17_w = move_landolt_ring('white_lr_rot_315.png',511.9425162,-7.162561934,'315_white_frame17_2_5.png')
frame17_b = move_landolt_ring('black_lr_rot_315.png',511.9425162,-7.162561934,'315_black_frame17_2_5.png')
frame18_w = move_landolt_ring('white_lr_rot_315.png',511.91413,-8.75408057,'315_white_frame18_2_5.png')
frame18_b = move_landolt_ring('black_lr_rot_315.png',511.91413,-8.75408057,'315_black_frame18_2_5.png')
frame19_w = move_landolt_ring('white_lr_rot_315.png',511.8800671,-10.34550213,'315_white_frame19_2_5.png')
frame19_b = move_landolt_ring('black_lr_rot_315.png',511.8800671,-10.34550213,'315_black_frame19_2_5.png')
frame20_w = move_landolt_ring('white_lr_rot_315.png',511.8403281,-11.93680898,'315_white_frame20_2_5.png')
frame20_b = move_landolt_ring('black_lr_rot_315.png',511.8403281,-11.93680898,'315_black_frame20_2_5.png')
frame21 = move_landolt_ring('white_lr_rot_315.png',511.7949134,-13.52798346,'315_white_frame21_2_5.png')
frame22 = move_landolt_ring('white_lr_rot_315.png',511.7438234,-15.11900792,'315_white_frame22_2_5.png')
frame23 = move_landolt_ring('white_lr_rot_315.png',511.6870588,-16.70986474,'315_white_frame23_2_5.png')
frame24 = move_landolt_ring('white_lr_rot_315.png',511.6246201,-18.30053626,'315_white_frame24_2_5.png')
frame25 = move_landolt_ring('white_lr_rot_315.png',511.556508,-19.89100484,'315_white_frame25_2_5.png')
#c.Difference spectrum
frame1 = move_landolt_ring('ds_white.png',511.556508,19.89100484,'ds_white_frame1_2_5.png')
frame2 = move_landolt_ring('ds_white.png',511.6246201,18.30053626,'ds_white_frame2_2_5.png')
frame3 = move_landolt_ring('ds_white.png',511.6870588,16.70986474,'ds_white_frame3_2_5.png')
frame4 = move_landolt_ring('ds_white.png',511.7438234,15.11900792,'ds_white_frame4_2_5.png')
frame5 = move_landolt_ring('ds_white.png',511.7949134,13.52798346,'ds_white_frame5_2_5.png')
frame6_w = move_landolt_ring('ds_white.png',511.8403281,11.93680898,'ds_white_frame6_2_5.png')
frame6_b = move_landolt_ring('ds_black.png',511.8403281,11.93680898,'ds_black_frame6_2_5.png')
frame7_w = move_landolt_ring('ds_white.png',511.8800671,10.34550213,'ds_white_frame7_2_5.png')
frame7_b = move_landolt_ring('ds_black.png',511.8800671,10.34550213,'ds_black_frame7_2_5.png')
frame8_w = move_landolt_ring('ds_white.png',511.91413,8.75408057,'ds_white_frame8_2_5.png')
frame8_b = move_landolt_ring('ds_black.png',511.91413,8.75408057,'ds_black_frame8_2_5.png')
frame9_w = move_landolt_ring('ds_white.png',511.9425162,7.162561934,'ds_white_frame9_2_5.png')
frame9_b = move_landolt_ring('ds_black.png',511.9425162,7.162561934,'ds_black_frame9_2_5.png')
frame10_w = move_landolt_ring('ds_white.png',511.9652256,5.570963872,'ds_white_frame10_2_5.png')
frame10_b = move_landolt_ring('ds_black.png',511.9652256,5.570963872,'ds_black_frame10_2_5.png')
frame11 = move_landolt_ring('ds_white.png',511.9822579,3.979304035,'ds_white_frame11_2_5.png')
frame12 = move_landolt_ring('ds_white.png',511.9936128,2.387600072,'ds_white_frame12_2_5.png')
frame13 = move_landolt_ring('ds_white.png',511.9992903,0.795869632,'ds_white_frame13_2_5.png')
frame14 = move_landolt_ring('ds_white.png',511.9936128,-2.387600072,'ds_white_frame14_2_5.png')
frame15 = move_landolt_ring('ds_white.png',511.9822579,-3.979304035,'ds_white_frame15_2_5.png')
frame16_w = move_landolt_ring('ds_white.png',511.9652256,-5.570963872,'ds_white_frame16_2_5.png')
frame16_b = move_landolt_ring('ds_black.png',511.9652256,-5.570963872,'ds_black_frame16_2_5.png')
frame17_w = move_landolt_ring('ds_white.png',511.9425162,-7.162561934,'ds_white_frame17_2_5.png')
frame17_b = move_landolt_ring('ds_black.png',511.9425162,-7.162561934,'ds_black_frame17_2_5.png')
frame18_w = move_landolt_ring('ds_white.png',511.91413,-8.75408057,'ds_white_frame18_2_5.png')
frame18_b = move_landolt_ring('ds_black.png',511.91413,-8.75408057,'ds_black_frame18_2_5.png')
frame19_w = move_landolt_ring('ds_white.png',511.8800671,-10.34550213,'ds_white_frame19_2_5.png')
frame19_b = move_landolt_ring('ds_black.png',511.8800671,-10.34550213,'ds_black_frame19_2_5.png')
frame20_w = move_landolt_ring('ds_white.png',511.8403281,-11.93680898,'ds_white_frame20_2_5.png')
frame20_b = move_landolt_ring('ds_black.png',511.8403281,-11.93680898,'ds_black_frame20_2_5.png')
frame21 = move_landolt_ring('ds_white.png',511.7949134,-13.52798346,'ds_white_frame21_2_5.png')
frame22 = move_landolt_ring('ds_white.png',511.7438234,-15.11900792,'ds_white_frame22_2_5.png')
frame23 = move_landolt_ring('ds_white.png',511.6870588,-16.70986474,'ds_white_frame23_2_5.png')
frame24 = move_landolt_ring('ds_white.png',511.6246201,-18.30053626,'ds_white_frame24_2_5.png')
frame25 = move_landolt_ring('ds_white.png',511.556508,-19.89100484,'ds_white_frame25_2_5.png')
#5 deg/s
#a.45
frame1 = move_landolt_ring('white_lr_rot_45.png',510.2268005,39.7475507,'45_white_frame1_5_deg.png')
frame2 = move_landolt_ring('white_lr_rot_45.png',510.4990307,36.57423793,'45_white_frame2_5_deg.png')
frame3 = move_landolt_ring('white_lr_rot_45.png',510.7486176,33.39930289,'45_white_frame3_5_deg.png')
frame4 = move_landolt_ring('white_lr_rot_45.png',510.97555,30.22288641,'45_white_frame4_5_deg.png')
frame5 = move_landolt_ring('white_lr_rot_45.png',511.1798179,27.04512938,'45_white_frame5_5_deg.png')
frame6_w = move_landolt_ring('white_lr_rot_45.png',511.3614121,23.86617275,'45_white_frame6_5_deg.png')
frame6_b = move_landolt_ring('black_lr_rot_45.png',511.3614121,23.86617275,'45_black_frame6_5_deg.png')
frame7_w = move_landolt_ring('white_lr_rot_45.png',511.5203247,20.68615753,'45_white_frame7_5_deg.png')
frame7_b = move_landolt_ring('black_lr_rot_45.png',511.5203247,20.68615753,'45_black_frame7_5_deg.png')
frame8_w = move_landolt_ring('white_lr_rot_45.png',511.6565486,17.50522476,'45_white_frame8_5_deg.png')
frame8_b = move_landolt_ring('black_lr_rot_45.png',511.6565486,17.50522476,'45_black_frame8_5_deg.png')
frame9_w = move_landolt_ring('white_lr_rot_45.png',511.7700778,14.32351554,'45_white_frame9_5_deg.png')
frame9_b = move_landolt_ring('black_lr_rot_45.png',511.7700778,14.32351554,'45_black_frame9_5_deg.png')
frame10_w = move_landolt_ring('white_lr_rot_45.png',511.8609071,11.141171,'45_white_frame10_5_deg.png')
frame10_b = move_landolt_ring('black_lr_rot_45.png',511.8609071,11.141171,'45_black_frame10_5_deg.png')
frame11 = move_landolt_ring('white_lr_rot_45.png',511.9290327,7.958332283,'45_white_frame11_5_deg.png')
frame12 = move_landolt_ring('white_lr_rot_45.png',511.9744514,4.775140572,'45_white_frame12_5_deg.png')
frame13 = move_landolt_ring('white_lr_rot_45.png',511.9971612,1.591737058,'45_white_frame13_5_deg.png')
frame14 = move_landolt_ring('white_lr_rot_45.png',511.9744514,-4.775140572,'45_white_frame14_5_deg.png')
frame15 = move_landolt_ring('white_lr_rot_45.png',511.9290327,-7.958332283,'45_white_frame15_5_deg.png')
frame16_w = move_landolt_ring('white_lr_rot_45.png',511.8609071,-11.141171,'45_white_frame16_5_deg.png')
frame16_b = move_landolt_ring('black_lr_rot_45.png',511.8609071,-11.141171,'45_black_frame16_5_deg.png')
frame17_w = move_landolt_ring('white_lr_rot_45.png',511.7700778,-14.32351554,'45_white_frame17_5_deg.png')
frame17_b = move_landolt_ring('black_lr_rot_45.png',511.7700778,-14.32351554,'45_black_frame17_5_deg.png')
frame18_w = move_landolt_ring('white_lr_rot_45.png',511.6565486,-17.50522476,'45_white_frame18_5_deg.png')
frame18_b = move_landolt_ring('black_lr_rot_45.png',511.6565486,-17.50522476,'45_black_frame18_5_deg.png')
frame19_w = move_landolt_ring('white_lr_rot_45.png',511.5203247,-20.68615753,'45_white_frame19_5_deg.png')
frame19_b = move_landolt_ring('black_lr_rot_45.png',511.5203247,-20.68615753,'45_black_frame19_5_deg.png')
frame20_w = move_landolt_ring('white_lr_rot_45.png',511.3614121,-23.86617275,'45_white_frame20_5_deg.png')
frame20_b = move_landolt_ring('black_lr_rot_45.png',511.3614121,-23.86617275,'45_black_frame20_5_deg.png')
frame21 = move_landolt_ring('white_lr_rot_45.png',511.1798179,-27.04512938,'45_white_frame21_5_deg.png')
frame22 = move_landolt_ring('white_lr_rot_45.png',510.97555,-30.22288641,'45_white_frame22_5_deg.png')
frame23 = move_landolt_ring('white_lr_rot_45.png',510.7486176,-33.39930289,'45_white_frame23_5_deg.png')
frame24 = move_landolt_ring('white_lr_rot_45.png',510.4990307,-36.57423793,'45_white_frame24_5_deg.png')
frame25 = move_landolt_ring('white_lr_rot_45.png',510.2268005,-39.7475507,'45_white_frame25_5_deg.png')
#b.315 Landolt ring
frame1 = move_landolt_ring('white_lr_rot_315.png',510.2268005,39.7475507,'315_white_frame1_5_deg.png')
frame2 = move_landolt_ring('white_lr_rot_315.png',510.4990307,36.57423793,'315_white_frame2_5_deg.png')
frame3 = move_landolt_ring('white_lr_rot_315.png',510.7486176,33.39930289,'315_white_frame3_5_deg.png')
frame4 = move_landolt_ring('white_lr_rot_315.png',510.97555,30.22288641,'315_white_frame4_5_deg.png')
frame5 = move_landolt_ring('white_lr_rot_315.png',511.1798179,27.04512938,'315_white_frame5_5_deg.png')
frame6_w = move_landolt_ring('white_lr_rot_315.png',511.3614121,23.86617275,'315_white_frame6_5_deg.png')
frame6_b = move_landolt_ring('black_lr_rot_315.png',511.3614121,23.86617275,'315_black_frame6_5_deg.png')
frame7_w = move_landolt_ring('white_lr_rot_315.png',511.5203247,20.68615753,'315_white_frame7_5_deg.png')
frame7_b = move_landolt_ring('black_lr_rot_315.png',511.5203247,20.68615753,'315_black_frame7_5_deg.png')
frame8_w = move_landolt_ring('white_lr_rot_315.png',511.6565486,17.50522476,'315_white_frame8_5_deg.png')
frame8_b = move_landolt_ring('black_lr_rot_315.png',511.6565486,17.50522476,'315_black_frame8_5_deg.png')
frame9_w = move_landolt_ring('white_lr_rot_315.png',511.7700778,14.32351554,'315_white_frame9_5_deg.png')
frame9_b = move_landolt_ring('black_lr_rot_315.png',511.7700778,14.32351554,'315_black_frame9_5_deg.png')
frame10_w = move_landolt_ring('white_lr_rot_315.png',511.8609071,11.141171,'315_white_frame10_5_deg.png')
frame10_b = move_landolt_ring('black_lr_rot_315.png',511.8609071,11.141171,'315_black_frame10_5_deg.png')
frame11 = move_landolt_ring('white_lr_rot_315.png',511.9290327,7.958332283,'315_white_frame11_5_deg.png')
frame12 = move_landolt_ring('white_lr_rot_315.png',511.9744514,4.775140572,'315_white_frame12_5_deg.png')
frame13 = move_landolt_ring('white_lr_rot_315.png',511.9971612,1.591737058,'315_white_frame13_5_deg.png')
frame14 = move_landolt_ring('white_lr_rot_315.png',511.9744514,-4.775140572,'315_white_frame14_5_deg.png')
frame15 = move_landolt_ring('white_lr_rot_315.png',511.9290327,-7.958332283,'315_white_frame15_5_deg.png')
frame16_w = move_landolt_ring('white_lr_rot_315.png',511.8609071,-11.141171,'315_white_frame16_5_deg.png')
frame16_b = move_landolt_ring('black_lr_rot_315.png',511.8609071,-11.141171,'315_black_frame16_5_deg.png')
frame17_w = move_landolt_ring('white_lr_rot_315.png',511.7700778,-14.32351554,'315_white_frame17_5_deg.png')
frame17_b = move_landolt_ring('black_lr_rot_315.png',511.7700778,-14.32351554,'315_black_frame17_5_deg.png')
frame18_w = move_landolt_ring('white_lr_rot_315.png',511.6565486,-17.50522476,'315_white_frame18_5_deg.png')
frame18_b = move_landolt_ring('black_lr_rot_315.png',511.6565486,-17.50522476,'315_black_frame18_5_deg.png')
frame19_w = move_landolt_ring('white_lr_rot_315.png',511.5203247,-20.68615753,'315_white_frame19_5_deg.png')
frame19_b = move_landolt_ring('black_lr_rot_315.png',511.5203247,-20.68615753,'315_black_frame19_5_deg.png')
frame20_w = move_landolt_ring('white_lr_rot_315.png',511.3614121,-23.86617275,'315_white_frame20_5_deg.png')
frame20_b = move_landolt_ring('black_lr_rot_315.png',511.3614121,-23.86617275,'315_black_frame20_5_deg.png')
frame21 = move_landolt_ring('white_lr_rot_315.png',511.1798179,-27.04512938,'315_white_frame21_5_deg.png')
frame22 = move_landolt_ring('white_lr_rot_315.png',510.97555,-30.22288641,'315_white_frame22_5_deg.png')
frame23 = move_landolt_ring('white_lr_rot_315.png',510.7486176,-33.39930289,'315_white_frame23_5_deg.png')
frame24 = move_landolt_ring('white_lr_rot_315.png',510.4990307,-36.57423793,'315_white_frame24_5_deg.png')
frame25 = move_landolt_ring('white_lr_rot_315.png',510.2268005,-39.7475507,'315_white_frame25_5_deg.png')
#c.Difference spectrum
frame1 = move_landolt_ring('ds_white.png',510.2268005,39.7475507,'ds_white_frame1_5_deg.png')
frame2 = move_landolt_ring('ds_white.png',510.4990307,36.57423793,'ds_white_frame2_5_deg.png')
frame3 = move_landolt_ring('ds_white.png',510.7486176,33.39930289,'ds_white_frame3_5_deg.png')
frame4 = move_landolt_ring('ds_white.png',510.97555,30.22288641,'ds_white_frame4_5_deg.png')
frame5 = move_landolt_ring('ds_white.png',511.1798179,27.04512938,'ds_white_frame5_5_deg.png')
frame6_w = move_landolt_ring('ds_white.png',511.3614121,23.86617275,'ds_white_frame6_5_deg.png')
frame6_b = move_landolt_ring('ds_black.png',511.3614121,23.86617275,'ds_black_frame6_5_deg.png')
frame7_w = move_landolt_ring('ds_white.png',511.5203247,20.68615753,'ds_white_frame7_5_deg.png')
frame7_b = move_landolt_ring('ds_black.png',511.5203247,20.68615753,'ds_black_frame7_5_deg.png')
frame8_w = move_landolt_ring('ds_white.png',511.6565486,17.50522476,'ds_white_frame8_5_deg.png')
frame8_b = move_landolt_ring('ds_black.png',511.6565486,17.50522476,'ds_black_frame8_5_deg.png')
frame9_w = move_landolt_ring('ds_white.png',511.7700778,14.32351554,'ds_white_frame9_5_deg.png')
frame9_b = move_landolt_ring('ds_black.png',511.7700778,14.32351554,'ds_black_frame9_5_deg.png')
frame10_w = move_landolt_ring('ds_white.png',511.8609071,11.141171,'ds_white_frame10_5_deg.png')
frame10_b = move_landolt_ring('ds_black.png',511.8609071,11.141171,'ds_black_frame10_5_deg.png')
frame11 = move_landolt_ring('ds_white.png',511.9290327,7.958332283,'ds_white_frame11_5_deg.png')
frame12 = move_landolt_ring('ds_white.png',511.9744514,4.775140572,'ds_white_frame12_5_deg.png')
frame13 = move_landolt_ring('ds_white.png',511.9971612,1.591737058,'ds_white_frame13_5_deg.png')
frame14 = move_landolt_ring('ds_white.png',511.9744514,-4.775140572,'ds_white_frame14_5_deg.png')
frame15 = move_landolt_ring('ds_white.png',511.9290327,-7.958332283,'ds_white_frame15_5_deg.png')
frame16_w = move_landolt_ring('ds_white.png',511.8609071,-11.141171,'ds_white_frame16_5_deg.png')
frame16_b = move_landolt_ring('ds_black.png',511.8609071,-11.141171,'ds_black_frame16_5_deg.png')
frame17_w = move_landolt_ring('ds_white.png',511.7700778,-14.32351554,'ds_white_frame17_5_deg.png')
frame17_b = move_landolt_ring('ds_black.png',511.7700778,-14.32351554,'ds_black_frame17_5_deg.png')
frame18_w = move_landolt_ring('ds_white.png',511.6565486,-17.50522476,'ds_white_frame18_5_deg.png')
frame18_b = move_landolt_ring('ds_black.png',511.6565486,-17.50522476,'ds_black_frame18_5_deg.png')
frame19_w = move_landolt_ring('ds_white.png',511.5203247,-20.68615753,'ds_white_frame19_5_deg.png')
frame19_b = move_landolt_ring('ds_black.png',511.5203247,-20.68615753,'ds_black_frame19_5_deg.png')
frame20_w = move_landolt_ring('ds_white.png',511.3614121,-23.86617275,'ds_white_frame20_5_deg.png')
frame20_b = move_landolt_ring('ds_black.png',511.3614121,-23.86617275,'ds_black_frame20_5_deg.png')
frame21 = move_landolt_ring('ds_white.png',511.1798179,-27.04512938,'ds_white_frame21_5_deg.png')
frame22 = move_landolt_ring('ds_white.png',510.97555,-30.22288641,'ds_white_frame22_5_deg.png')
frame23 = move_landolt_ring('ds_white.png',510.7486176,-33.39930289,'ds_white_frame23_5_deg.png')
frame24 = move_landolt_ring('ds_white.png',510.4990307,-36.57423793,'ds_white_frame24_5_deg.png')
frame25 = move_landolt_ring('ds_white.png',510.2268005,-39.7475507,'ds_white_frame25_5_deg.png')
#10 deg/s
#a.45
frame1 = move_landolt_ring('white_lr_rot_45.png',504.919484,79.21978758,'45_white_frame1_10_deg.png')
frame2 = move_landolt_ring('white_lr_rot_45.png',506.0049234,72.93403521,'45_white_frame2_10_deg.png')
frame3 = move_landolt_ring('white_lr_rot_45.png',507.0005875,66.63534289,'45_white_frame3_10_deg.png')
frame4 = move_landolt_ring('white_lr_rot_45.png',507.9062997,60.32482815,'45_white_frame4_10_deg.png')
frame5 = move_landolt_ring('white_lr_rot_45.png',508.7218993,54.00361059,'45_white_frame5_10_deg.png')
frame6_w = move_landolt_ring('white_lr_rot_45.png',509.4472415,47.67281172,'45_white_frame6_10_deg.png')
frame6_b = move_landolt_ring('black_lr_rot_45.png',509.4472415,47.67281172,'45_black_frame6_10_deg.png')
frame7_w = move_landolt_ring('white_lr_rot_45.png',510.0821978,41.33355475,'45_white_frame7_10_deg.png')
frame7_b = move_landolt_ring('black_lr_rot_45.png',510.0821978,41.33355475,'45_black_frame7_10_deg.png')
frame8_w = move_landolt_ring('white_lr_rot_45.png',510.6266553,34.98696439,'45_white_frame8_10_deg.png')
frame8_b = move_landolt_ring('black_lr_rot_45.png',510.6266553,34.98696439,'45_black_frame8_10_deg.png')
frame9_w = move_landolt_ring('white_lr_rot_45.png',511.0805176,28.63416665,'45_white_frame9_10_deg.png')
frame9_b = move_landolt_ring('black_lr_rot_45.png',511.0805176,28.63416665,'45_black_frame9_10_deg.png')
frame10_w = move_landolt_ring('white_lr_rot_45.png',511.4437041,22.27628865,'45_white_frame10_10_deg.png')
frame10_b = move_landolt_ring('black_lr_rot_45.png',511.4437041,22.27628865,'45_black_frame10_10_deg.png')
frame11 = move_landolt_ring('white_lr_rot_45.png',511.7161504,15.91445839,'45_white_frame11_10_deg.png')
frame12 = move_landolt_ring('white_lr_rot_45.png',511.8978081,9.549804589,'45_white_frame12_10_deg.png')
frame13 = move_landolt_ring('white_lr_rot_45.png',511.988645,3.183456466,'45_white_frame13_10_deg.png')
frame14 = move_landolt_ring('white_lr_rot_45.png',511.8978081,-9.549804589,'45_white_frame14_10_deg.png')
frame15 = move_landolt_ring('white_lr_rot_45.png',511.7161504,-15.91445839,'45_white_frame15_10_deg.png')
frame16_w = move_landolt_ring('white_lr_rot_45.png',511.4437041,-22.27628865,'45_white_frame16_10_deg.png')
frame16_b = move_landolt_ring('black_lr_rot_45.png',511.4437041,-22.27628865,'45_black_frame16_10_deg.png')
frame17_w = move_landolt_ring('white_lr_rot_45.png',511.0805176,-28.63416665,'45_white_frame17_10_deg.png')
frame17_b = move_landolt_ring('black_lr_rot_45.png',511.0805176,-28.63416665,'45_black_frame17_10_deg.png')
frame18_w = move_landolt_ring('white_lr_rot_45.png',510.6266553,-34.98696439,'45_white_frame18_10_deg.png')
frame18_b = move_landolt_ring('black_lr_rot_45.png',510.6266553,-34.98696439,'45_black_frame18_10_deg.png')
frame19_w = move_landolt_ring('white_lr_rot_45.png',510.0821978,-41.33355475,'45_white_frame19_10_deg.png')
frame19_b = move_landolt_ring('black_lr_rot_45.png',510.0821978,-41.33355475,'45_black_frame19_10_deg.png')
frame20_w = move_landolt_ring('white_lr_rot_45.png',509.4472415,-47.67281172,'45_white_frame20_10_deg.png')
frame20_b = move_landolt_ring('black_lr_rot_45.png',509.4472415,-47.67281172,'45_black_frame20_10_deg.png')
frame21 = move_landolt_ring('white_lr_rot_45.png',508.7218993,-54.00361059,'45_white_frame21_10_deg.png')
frame22 = move_landolt_ring('white_lr_rot_45.png',507.9062997,-60.32482815,'45_white_frame22_10_deg.png')
frame23 = move_landolt_ring('white_lr_rot_45.png',507.0005875,-66.63534289,'45_white_frame23_10_deg.png')
frame24 = move_landolt_ring('white_lr_rot_45.png',506.0049234,-72.93403521,'45_white_frame24_10_deg.png')
frame25 = move_landolt_ring('white_lr_rot_45.png',504.919484,-79.21978758,'45_white_frame25_10_deg.png')
#b.315 Landolt ring
frame1 = move_landolt_ring('white_lr_rot_315.png',504.919484,79.21978758,'315_white_frame1_10_deg.png')
frame2 = move_landolt_ring('white_lr_rot_315.png',506.0049234,72.93403521,'315_white_frame2_10_deg.png')
frame3 = move_landolt_ring('white_lr_rot_315.png',507.0005875,66.63534289,'315_white_frame3_10_deg.png')
frame4 = move_landolt_ring('white_lr_rot_315.png',507.9062997,60.32482815,'315_white_frame4_10_deg.png')
frame5 = move_landolt_ring('white_lr_rot_315.png',508.7218993,54.00361059,'315_white_frame5_10_deg.png')
frame6_w = move_landolt_ring('white_lr_rot_315.png',509.4472415,47.67281172,'315_white_frame6_10_deg.png')
frame6_b = move_landolt_ring('black_lr_rot_315.png',509.4472415,47.67281172,'315_black_frame6_10_deg.png')
frame7_w = move_landolt_ring('white_lr_rot_315.png',510.0821978,41.33355475,'315_white_frame7_10_deg.png')
frame7_b = move_landolt_ring('black_lr_rot_315.png',510.0821978,41.33355475,'315_black_frame7_10_deg.png')
frame8_w = move_landolt_ring('white_lr_rot_315.png',510.6266553,34.98696439,'315_white_frame8_10_deg.png')
frame8_b = move_landolt_ring('black_lr_rot_315.png',510.6266553,34.98696439,'315_black_frame8_10_deg.png')
frame9_w = move_landolt_ring('white_lr_rot_315.png',511.0805176,28.63416665,'315_white_frame9_10_deg.png')
frame9_b = move_landolt_ring('black_lr_rot_315.png',511.0805176,28.63416665,'315_black_frame9_10_deg.png')
frame10_w = move_landolt_ring('white_lr_rot_315.png',511.4437041,22.27628865,'315_white_frame10_10_deg.png')
frame10_b = move_landolt_ring('black_lr_rot_315.png',511.4437041,22.27628865,'315_black_frame10_10_deg.png')
frame11 = move_landolt_ring('white_lr_rot_315.png',511.7161504,15.91445839,'315_white_frame11_10_deg.png')
frame12 = move_landolt_ring('white_lr_rot_315.png',511.8978081,9.549804589,'315_white_frame12_10_deg.png')
frame13 = move_landolt_ring('white_lr_rot_315.png',511.988645,3.183456466,'315_white_frame13_10_deg.png')
frame14 = move_landolt_ring('white_lr_rot_315.png',511.8978081,-9.549804589,'315_white_frame14_10_deg.png')
frame15 = move_landolt_ring('white_lr_rot_315.png',511.7161504,-15.91445839,'315_white_frame15_10_deg.png')
frame16_w = move_landolt_ring('white_lr_rot_315.png',511.4437041,-22.27628865,'315_white_frame16_10_deg.png')
frame16_b = move_landolt_ring('black_lr_rot_315.png',511.4437041,-22.27628865,'315_black_frame16_10_deg.png')
frame17_w = move_landolt_ring('white_lr_rot_315.png',511.0805176,-28.63416665,'315_white_frame17_10_deg.png')
frame17_b = move_landolt_ring('black_lr_rot_315.png',511.0805176,-28.63416665,'315_black_frame17_10_deg.png')
frame18_w = move_landolt_ring('white_lr_rot_315.png',510.6266553,-34.98696439,'315_white_frame18_10_deg.png')
frame18_b = move_landolt_ring('black_lr_rot_315.png',510.6266553,-34.98696439,'315_black_frame18_10_deg.png')
frame19_w = move_landolt_ring('white_lr_rot_315.png',510.0821978,-41.33355475,'315_white_frame19_10_deg.png')
frame19_b = move_landolt_ring('black_lr_rot_315.png',510.0821978,-41.33355475,'315_black_frame19_10_deg.png')
frame20_w = move_landolt_ring('white_lr_rot_315.png',509.4472415,-47.67281172,'315_white_frame20_10_deg.png')
frame20_b = move_landolt_ring('black_lr_rot_315.png',509.4472415,-47.67281172,'315_black_frame20_10_deg.png')
frame21 = move_landolt_ring('white_lr_rot_315.png',508.7218993,-54.00361059,'315_white_frame21_10_deg.png')
frame22 = move_landolt_ring('white_lr_rot_315.png',507.9062997,-60.32482815,'315_white_frame22_10_deg.png')
frame23 = move_landolt_ring('white_lr_rot_315.png',507.0005875,-66.63534289,'315_white_frame23_10_deg.png')
frame24 = move_landolt_ring('white_lr_rot_315.png',506.0049234,-72.93403521,'315_white_frame24_10_deg.png')
frame25 = move_landolt_ring('white_lr_rot_315.png',504.919484,-79.21978758,'315_white_frame25_10_deg.png')
#c.Difference spectrum
frame1 = move_landolt_ring('ds_white.png',504.919484,79.21978758,'ds_white_frame1_10_deg.png')
frame2 = move_landolt_ring('ds_white.png',506.0049234,72.93403521,'ds_white_frame2_10_deg.png')
frame3 = move_landolt_ring('ds_white.png',507.0005875,66.63534289,'ds_white_frame3_10_deg.png')
frame4 = move_landolt_ring('ds_white.png',507.9062997,60.32482815,'ds_white_frame4_10_deg.png')
frame5 = move_landolt_ring('ds_white.png',508.7218993,54.00361059,'ds_white_frame5_10_deg.png')
frame6_w = move_landolt_ring('ds_white.png',509.4472415,47.67281172,'ds_white_frame6_10_deg.png')
frame6_b = move_landolt_ring('ds_black.png',509.4472415,47.67281172,'ds_black_frame6_10_deg.png')
frame7_w = move_landolt_ring('ds_white.png',510.0821978,41.33355475,'ds_white_frame7_10_deg.png')
frame7_b = move_landolt_ring('ds_black.png',510.0821978,41.33355475,'ds_black_frame7_10_deg.png')
frame8_w = move_landolt_ring('ds_white.png',510.6266553,34.98696439,'ds_white_frame8_10_deg.png')
frame8_b = move_landolt_ring('ds_black.png',510.6266553,34.98696439,'ds_black_frame8_10_deg.png')
frame9_w = move_landolt_ring('ds_white.png',511.0805176,28.63416665,'ds_white_frame9_10_deg.png')
frame9_b = move_landolt_ring('ds_black.png',511.0805176,28.63416665,'ds_black_frame9_10_deg.png')
frame10_w = move_landolt_ring('ds_white.png',511.4437041,22.27628865,'ds_white_frame10_10_deg.png')
frame10_b = move_landolt_ring('ds_black.png',511.4437041,22.27628865,'ds_black_frame10_10_deg.png')
frame11 = move_landolt_ring('ds_white.png',511.7161504,15.91445839,'ds_white_frame11_10_deg.png')
frame12 = move_landolt_ring('ds_white.png',511.8978081,9.549804589,'ds_white_frame12_10_deg.png')
frame13 = move_landolt_ring('ds_white.png',511.988645,3.183456466,'ds_white_frame13_10_deg.png')
frame14 = move_landolt_ring('ds_white.png',511.8978081,-9.549804589,'ds_white_frame14_10_deg.png')
frame15 = move_landolt_ring('ds_white.png',511.7161504,-15.91445839,'ds_white_frame15_10_deg.png')
frame16_w = move_landolt_ring('ds_white.png',511.4437041,-22.27628865,'ds_white_frame16_10_deg.png')
frame16_b = move_landolt_ring('ds_black.png',511.4437041,-22.27628865,'ds_black_frame16_10_deg.png')
frame17_w = move_landolt_ring('ds_white.png',511.0805176,-28.63416665,'ds_white_frame17_10_deg.png')
frame17_b = move_landolt_ring('ds_black.png',511.0805176,-28.63416665,'ds_black_frame17_10_deg.png')
frame18_w = move_landolt_ring('ds_white.png',510.6266553,-34.98696439,'ds_white_frame18_10_deg.png')
frame18_b = move_landolt_ring('ds_black.png',510.6266553,-34.98696439,'ds_black_frame18_10_deg.png')
frame19_w = move_landolt_ring('ds_white.png',510.0821978,-41.33355475,'ds_white_frame19_10_deg.png')
frame19_b = move_landolt_ring('ds_black.png',510.0821978,-41.33355475,'ds_black_frame19_10_deg.png')
frame20_w = move_landolt_ring('ds_white.png',509.4472415,-47.67281172,'ds_white_frame20_10_deg.png')
frame20_b = move_landolt_ring('ds_black.png',509.4472415,-47.67281172,'ds_black_frame20_10_deg.png')
frame21 = move_landolt_ring('ds_white.png',508.7218993,-54.00361059,'ds_white_frame21_10_deg.png')
frame22 = move_landolt_ring('ds_white.png',507.9062997,-60.32482815,'ds_white_frame22_10_deg.png')
frame23 = move_landolt_ring('ds_white.png',507.0005875,-66.63534289,'ds_white_frame23_10_deg.png')
frame24 = move_landolt_ring('ds_white.png',506.0049234,-72.93403521,'ds_white_frame24_10_deg.png')
frame25 = move_landolt_ring('ds_white.png',504.919484,-79.21978758,'ds_white_frame25_10_deg.png')
#20 deg/s
#a.45
frame1 = move_landolt_ring('white_lr_rot_45.png',483.8737709,156.2484932,'45_white_frame1_20_deg.png')
frame2 = move_landolt_ring('white_lr_rot_45.png',488.160088,144.1600816,'45_white_frame2_20_deg.png')
frame3 = move_landolt_ring('white_lr_rot_45.png',492.0999835,131.9693672,'45_white_frame3_20_deg.png')
frame4 = move_landolt_ring('white_lr_rot_45.png',495.6906614,119.685001,'45_white_frame4_20_deg.png')
frame5 = move_landolt_ring('white_lr_rot_45.png',498.9295736,107.3157006,'45_white_frame5_20_deg.png')
frame6_w = move_landolt_ring('white_lr_rot_45.png',501.8144216,94.87024386,'45_white_frame6_20_deg.png')
frame6_b = move_landolt_ring('black_lr_rot_45.png',501.8144216,94.87024386,'45_black_frame6_20_deg.png')
frame7_w = move_landolt_ring('white_lr_rot_45.png',504.3431581,82.35746269,'45_white_frame7_20_deg.png')
frame7_b = move_landolt_ring('black_lr_rot_45.png',504.3431581,82.35746269,'45_black_frame7_20_deg.png')
frame8_w = move_landolt_ring('white_lr_rot_45.png',506.5139888,69.78623675,'45_white_frame8_20_deg.png')
frame8_b = move_landolt_ring('black_lr_rot_45.png',506.5139888,69.78623675,'45_black_frame8_20_deg.png')
frame9_w = move_landolt_ring('white_lr_rot_45.png',508.325373,57.16548717,'45_white_frame9_20_deg.png')
frame9_b = move_landolt_ring('black_lr_rot_45.png',508.325373,57.16548717,'45_black_frame9_20_deg.png')
frame10_w = move_landolt_ring('white_lr_rot_45.png',509.7760253,44.50417024,'45_white_frame10_20_deg.png')
frame10_b = move_landolt_ring('black_lr_rot_45.png',509.7760253,44.50417024,'45_black_frame10_20_deg.png')
frame11 = move_landolt_ring('white_lr_rot_45.png',510.8649163,31.81127102,'45_white_frame11_20_deg.png')
frame12 = move_landolt_ring('white_lr_rot_45.png',511.5912732,19.09579702,'45_white_frame12_20_deg.png')
frame13 = move_landolt_ring('white_lr_rot_45.png',511.9545805,6.366771728,'45_white_frame13_20_deg.png')
frame14 = move_landolt_ring('white_lr_rot_45.png',511.5912732,-19.09579702,'45_white_frame14_20_deg.png')
frame15 = move_landolt_ring('white_lr_rot_45.png',510.8649163,-31.81127102,'45_white_frame15_20_deg.png')
frame16_w = move_landolt_ring('white_lr_rot_45.png',509.7760253,-44.50417024,'45_white_frame16_20_deg.png')
frame16_b = move_landolt_ring('black_lr_rot_45.png',509.7760253,-44.50417024,'45_black_frame16_20_deg.png')
frame17_w = move_landolt_ring('white_lr_rot_45.png',508.325373,-57.16548717,'45_white_frame17_20_deg.png')
frame17_b = move_landolt_ring('black_lr_rot_45.png',508.325373,-57.16548717,'45_black_frame17_20_deg.png')
frame18_w = move_landolt_ring('white_lr_rot_45.png',506.5139888,-69.78623675,'45_white_frame18_20_deg.png')
frame18_b = move_landolt_ring('black_lr_rot_45.png',506.5139888,-69.78623675,'45_black_frame18_20_deg.png')
frame19_w = move_landolt_ring('white_lr_rot_45.png',504.3431581,-82.35746269,'45_white_frame19_20_deg.png')
frame19_b = move_landolt_ring('black_lr_rot_45.png',504.3431581,-82.35746269,'45_black_frame19_20_deg.png')
frame20_w = move_landolt_ring('white_lr_rot_45.png',501.8144216,-94.87024386,'45_white_frame20_20_deg.png')
frame20_b = move_landolt_ring('black_lr_rot_45.png',501.8144216,-94.87024386,'45_black_frame20_20_deg.png')
frame21 = move_landolt_ring('white_lr_rot_45.png',498.9295736,-107.3157006,'45_white_frame21_20_deg.png')
frame22 = move_landolt_ring('white_lr_rot_45.png',495.6906614,-119.685001,'45_white_frame22_20_deg.png')
frame23 = move_landolt_ring('white_lr_rot_45.png',492.0999835,-131.9693672,'45_white_frame23_20_deg.png')
frame24 = move_landolt_ring('white_lr_rot_45.png',488.160088,-144.1600816,'45_white_frame24_20_deg.png')
frame25 = move_landolt_ring('white_lr_rot_45.png',483.8737709,-156.2484932,'45_white_frame25_20_deg.png')
#b.315 Landolt ring
frame1 = move_landolt_ring('white_lr_rot_315.png',483.8737709,156.2484932,'315_white_frame1_20_deg.png')
frame2 = move_landolt_ring('white_lr_rot_315.png',488.160088,144.1600816,'315_white_frame2_20_deg.png')
frame3 = move_landolt_ring('white_lr_rot_315.png',492.0999835,131.9693672,'315_white_frame3_20_deg.png')
frame4 = move_landolt_ring('white_lr_rot_315.png',495.6906614,119.685001,'315_white_frame4_20_deg.png')
frame5 = move_landolt_ring('white_lr_rot_315.png',498.9295736,107.3157006,'315_white_frame5_20_deg.png')
frame6_w = move_landolt_ring('white_lr_rot_315.png',501.8144216,94.87024386,'315_white_frame6_20_deg.png')
frame6_b = move_landolt_ring('black_lr_rot_315.png',501.8144216,94.87024386,'315_black_frame6_20_deg.png')
frame7_w = move_landolt_ring('white_lr_rot_315.png',504.3431581,82.35746269,'315_white_frame7_20_deg.png')
frame7_b = move_landolt_ring('black_lr_rot_315.png',504.3431581,82.35746269,'315_black_frame7_20_deg.png')
frame8_w = move_landolt_ring('white_lr_rot_315.png',506.5139888,69.78623675,'315_white_frame8_20_deg.png')
frame8_b = move_landolt_ring('black_lr_rot_315.png',506.5139888,69.78623675,'315_black_frame8_20_deg.png')
frame9_w = move_landolt_ring('white_lr_rot_315.png',508.325373,57.16548717,'315_white_frame9_20_deg.png')
frame9_b = move_landolt_ring('black_lr_rot_315.png',508.325373,57.16548717,'315_black_frame9_20_deg.png')
frame10_w = move_landolt_ring('white_lr_rot_315.png',509.7760253,44.50417024,'315_white_frame10_20_deg.png')
frame10_b = move_landolt_ring('black_lr_rot_315.png',509.7760253,44.50417024,'315_black_frame10_20_deg.png')
frame11 = move_landolt_ring('white_lr_rot_315.png',510.8649163,31.81127102,'315_white_frame11_20_deg.png')
frame12 = move_landolt_ring('white_lr_rot_315.png',511.5912732,19.09579702,'315_white_frame12_20_deg.png')
frame13 = move_landolt_ring('white_lr_rot_315.png',511.9545805,6.366771728,'315_white_frame13_20_deg.png')
frame14 = move_landolt_ring('white_lr_rot_315.png',511.5912732,-19.09579702,'315_white_frame14_20_deg.png')
frame15 = move_landolt_ring('white_lr_rot_315.png',510.8649163,-31.81127102,'315_white_frame15_20_deg.png')
frame16_w = move_landolt_ring('white_lr_rot_315.png',509.7760253,-44.50417024,'315_white_frame16_20_deg.png')
frame16_b = move_landolt_ring('black_lr_rot_315.png',509.7760253,-44.50417024,'315_black_frame16_20_deg.png')
frame17_w = move_landolt_ring('white_lr_rot_315.png',508.325373,-57.16548717,'315_white_frame17_20_deg.png')
frame17_b = move_landolt_ring('black_lr_rot_315.png',508.325373,-57.16548717,'315_black_frame17_20_deg.png')
frame18_w = move_landolt_ring('white_lr_rot_315.png',506.5139888,-69.78623675,'315_white_frame18_20_deg.png')
frame18_b = move_landolt_ring('black_lr_rot_315.png',506.5139888,-69.78623675,'315_black_frame18_20_deg.png')
frame19_w = move_landolt_ring('white_lr_rot_315.png',504.3431581,-82.35746269,'315_white_frame19_20_deg.png')
frame19_b = move_landolt_ring('black_lr_rot_315.png',504.3431581,-82.35746269,'315_black_frame19_20_deg.png')
frame20_w = move_landolt_ring('white_lr_rot_315.png',501.8144216,-94.87024386,'315_white_frame20_20_deg.png')
frame20_b = move_landolt_ring('black_lr_rot_315.png',501.8144216,-94.87024386,'315_black_frame20_20_deg.png')
frame21 = move_landolt_ring('white_lr_rot_315.png',498.9295736,-107.3157006,'315_white_frame21_20_deg.png')
frame22 = move_landolt_ring('white_lr_rot_315.png',495.6906614,-119.685001,'315_white_frame22_20_deg.png')
frame23 = move_landolt_ring('white_lr_rot_315.png',492.0999835,-131.9693672,'315_white_frame23_20_deg.png')
frame24 = move_landolt_ring('white_lr_rot_315.png',488.160088,-144.1600816,'315_white_frame24_20_deg.png')
frame25 = move_landolt_ring('white_lr_rot_315.png',483.8737709,-156.2484932,'315_white_frame25_20_deg.png')
#c.Difference spectrum
frame1 = move_landolt_ring('ds_white.png',483.8737709,156.2484932,'ds_white_frame1_20_deg.png')
frame2 = move_landolt_ring('ds_white.png',488.160088,144.1600816,'ds_white_frame2_20_deg.png')
frame3 = move_landolt_ring('ds_white.png',492.0999835,131.9693672,'ds_white_frame3_20_deg.png')
frame4 = move_landolt_ring('ds_white.png',495.6906614,119.685001,'ds_white_frame4_20_deg.png')
frame5 = move_landolt_ring('ds_white.png',498.9295736,107.3157006,'ds_white_frame5_20_deg.png')
frame6_w = move_landolt_ring('ds_white.png',501.8144216,94.87024386,'ds_white_frame6_20_deg.png')
frame6_b = move_landolt_ring('ds_black.png',501.8144216,94.87024386,'ds_black_frame6_20_deg.png')
frame7_w = move_landolt_ring('ds_white.png',504.3431581,82.35746269,'ds_white_frame7_20_deg.png')
frame7_b = move_landolt_ring('ds_black.png',504.3431581,82.35746269,'ds_black_frame7_20_deg.png')
frame8_w = move_landolt_ring('ds_white.png',506.5139888,69.78623675,'ds_white_frame8_20_deg.png')
frame8_b = move_landolt_ring('ds_black.png',506.5139888,69.78623675,'ds_black_frame8_20_deg.png')
frame9_w = move_landolt_ring('ds_white.png',508.325373,57.16548717,'ds_white_frame9_20_deg.png')
frame9_b = move_landolt_ring('ds_black.png',508.325373,57.16548717,'ds_black_frame9_20_deg.png')
frame10_w = move_landolt_ring('ds_white.png',509.7760253,44.50417024,'ds_white_frame10_20_deg.png')
frame10_b = move_landolt_ring('ds_black.png',509.7760253,44.50417024,'ds_black_frame10_20_deg.png')
frame11 = move_landolt_ring('ds_white.png',510.8649163,31.81127102,'ds_white_frame11_20_deg.png')
frame12 = move_landolt_ring('ds_white.png',511.5912732,19.09579702,'ds_white_frame12_20_deg.png')
frame13 = move_landolt_ring('ds_white.png',511.9545805,6.366771728,'ds_white_frame13_20_deg.png')
frame14 = move_landolt_ring('ds_white.png',511.5912732,-19.09579702,'ds_white_frame14_20_deg.png')
frame15 = move_landolt_ring('ds_white.png',510.8649163,-31.81127102,'ds_white_frame15_20_deg.png')
frame16_w = move_landolt_ring('ds_white.png',509.7760253,-44.50417024,'ds_white_frame16_20_deg.png')
frame16_b = move_landolt_ring('ds_black.png',509.7760253,-44.50417024,'ds_black_frame16_20_deg.png')
frame17_w = move_landolt_ring('ds_white.png',508.325373,-57.16548717,'ds_white_frame17_20_deg.png')
frame17_b = move_landolt_ring('ds_black.png',508.325373,-57.16548717,'ds_black_frame17_20_deg.png')
frame18_w = move_landolt_ring('ds_white.png',506.5139888,-69.78623675,'ds_white_frame18_20_deg.png')
frame18_b = move_landolt_ring('ds_black.png',506.5139888,-69.78623675,'ds_black_frame18_20_deg.png')
frame19_w = move_landolt_ring('ds_white.png',504.3431581,-82.35746269,'ds_white_frame19_20_deg.png')
frame19_b = move_landolt_ring('ds_black.png',504.3431581,-82.35746269,'ds_black_frame19_20_deg.png')
frame20_w = move_landolt_ring('ds_white.png',501.8144216,-94.87024386,'ds_white_frame20_20_deg.png')
frame20_b = move_landolt_ring('ds_black.png',501.8144216,-94.87024386,'ds_black_frame20_20_deg.png')
frame21 = move_landolt_ring('ds_white.png',498.9295736,-107.3157006,'ds_white_frame21_20_deg.png')
frame22 = move_landolt_ring('ds_white.png',495.6906614,-119.685001,'ds_white_frame22_20_deg.png')
frame23 = move_landolt_ring('ds_white.png',492.0999835,-131.9693672,'ds_white_frame23_20_deg.png')
frame24 = move_landolt_ring('ds_white.png',488.160088,-144.1600816,'ds_white_frame24_20_deg.png')
frame25 = move_landolt_ring('ds_white.png',483.8737709,-156.2484932,'ds_white_frame25_20_deg.png')

pygame.display.quit()
pygame.quit()

up_white_0  = []
for filename in glob.glob('45*white*_0*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_0.append(im)
up_white_0_3d = fft.fftshift(np.log(fftn(np.dstack(up_white_0))))
print(up_white_0_3d.shape) 
filter  = ["45_white_frame{}_0.png".format(i) for i in range(1, 6)]
filter = [f for f in filter if isfile(f)]
up_white_0_1  = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_0_1.append(im)
filter  = ["45_white_frame{}_0.png".format(i) for i in range(11, 16)]
filter = [f for f in filter if isfile(f)]
up_white_0_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_0_2.append(im)
filter  = ["45_white_frame{}_0.png".format(i) for i in range(21, 26)]
filter = [f for f in filter if isfile(f)]
up_white_0_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_0_3.append(im)

up_white_0_1_3d = fft.fftshift(np.log(fftn(np.dstack(up_white_0_1))))
print(up_white_0_1_3d.shape)
filter  = ["45_black_frame{}_0.png".format(i) for i in range(6, 11)]
filter = [f for f in filter if isfile(f)]
up_black_0_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_0_1.append(im)
filter  = ["45_black_frame{}_0.png".format(i) for i in range(16, 21)]
filter = [f for f in filter if isfile(f)]
up_black_0_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_0_2.append(im)
down_white_0 = []
for filename in glob.glob('315*white*_0*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_0.append(im)
filter  = ["315_white_frame{}_0.png".format(i) for i in range(1, 6)]
filter = [f for f in filter if isfile(f)]
down_white_0_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_0_1.append(im)
filter  = ["315_white_frame{}_0.png".format(i) for i in range(11, 16)]
filter = [f for f in filter if isfile(f)]
down_white_0_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_0_2.append(im)
filter  = ["315_white_frame{}_0.png".format(i) for i in range(21, 26)]
filter = [f for f in filter if isfile(f)]
down_white_0_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_0_3.append(im)
filter  = ["315_black_frame{}_0.png".format(i) for i in range(6, 11)]
filter = [f for f in filter if isfile(f)]
down_black_0_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_0_1.append(im)
filter  = ["315_black_frame{}_0.png".format(i) for i in range(16, 21)]
filter = [f for f in filter if isfile(f)]
down_black_0_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_0_2.append(im)
ds_white_0  = []
for filename in glob.glob('ds*white*_0*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_0.append(im)
ds_white_0_3d = fft.fftshift(np.log(fftn(np.dstack(ds_white_0))))
print(ds_white_0_3d.shape) 
filter  = ["ds_white_frame{}_0.png".format(i) for i in range(1, 6)]
filter = [f for f in filter if isfile(f)]
ds_white_0_1  = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_0_1.append(im)
filter  = ["ds_white_frame{}_0.png".format(i) for i in range(11, 16)]
filter = [f for f in filter if isfile(f)]
ds_white_0_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_0_2.append(im)
filter  = ["ds_white_frame{}_0.png".format(i) for i in range(21, 26)]
filter = [f for f in filter if isfile(f)]
ds_white_0_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_0_3.append(im)

ds_white_0_1_3d = fft.fftshift(np.log(fftn(np.dstack(ds_white_0_1))))
print(ds_white_0_1_3d.shape)
filter  = ["ds_black_frame{}_0.png".format(i) for i in range(6, 11)]
filter = [f for f in filter if isfile(f)]
ds_black_0_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_0_1.append(im)
filter  = ["ds_black_frame{}_0.png".format(i) for i in range(16, 21)]
filter = [f for f in filter if isfile(f)]
ds_black_0_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_0_2.append(im)
up_white_125  = []
for filename in glob.glob('45*white*_125*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_125.append(im)
up_white_125_3d = fft.fftshift(np.log(fftn(np.dstack(up_white_125))))
print(up_white_125_3d.shape)
filter  = ["45_white_frame{}_125.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
up_white_125_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_125_1.append(im)
filter  = ["45_white_frame{}_125.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
up_white_125_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_125_2.append(im)
filter  = ["45_white_frame{}_125.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
up_white_125_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_125_3.append(im)
filter  = ["45_black_frame{}_125.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
up_black_125_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_125_1.append(im)
filter  = ["45_black_frame{}_125.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
up_black_125_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_125_2.append(im)
down_white_125 = []
for filename in glob.glob('315*white*_125*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_125.append(im)
filter  = ["315_white_frame{}_125.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
down_white_125_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_125_1.append(im)
filter  = ["315_white_frame{}_125.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
down_white_125_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_125_2.append(im)
filter  = ["315_white_frame{}_125.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
down_white_125_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_125_3.append(im)
filter  = ["315_black_frame{}_125.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
down_black_125_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_125_1.append(im)
filter  = ["315_black_frame{}_125.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
down_black_125_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_125_2.append(im)
ds_white_125  = []
for filename in glob.glob('ds*white*_125*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_125.append(im)
ds_white_125_3d = fft.fftshift(np.log(fftn(np.dstack(ds_white_125))))
print(ds_white_125_3d.shape)
filter  = ["ds_white_frame{}_125.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
ds_white_125_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_125_1.append(im)
filter  = ["ds_white_frame{}_125.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
ds_white_125_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_125_2.append(im)
filter  = ["ds_white_frame{}_125.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
ds_white_125_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_125_3.append(im)
filter  = ["ds_black_frame{}_125.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
ds_black_125_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_125_1.append(im)
filter  = ["ds_black_frame{}_125.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
ds_black_125_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_125_2.append(im)
up_white_25  = []
for filename in glob.glob('45*white*_2_5*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_25.append(im)
up_white_25_3d = fft.fftshift(np.log(fftn(np.dstack(up_white_25))))
print(up_white_25_3d.shape)
filter  = ["45_white_frame{}_2_5.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
up_white_25_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_25_1.append(im)
filter  = ["45_white_frame{}_2_5.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
up_white_25_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_25_2.append(im)
filter  = ["45_white_frame{}_2_5.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
up_white_25_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_25_3.append(im)
filter  = ["45_black_frame{}_2_5.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
up_black_25_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_25_1.append(im)
filter  = ["45_black_frame{}_2_5.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
up_black_25_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_25_2.append(im)

down_white_25 = []
for filename in glob.glob('315*white*_2_5*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_25.append(im)
filter  = ["315_white_frame{}_2_5.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
down_white_25_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_25_1.append(im)
filter  = ["315_white_frame{}_2_5.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
down_white_25_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_25_2.append(im)
filter  = ["315_white_frame{}_2_5.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
down_white_25_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_25_3.append(im)
filter  = ["315_black_frame{}_2_5.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
down_black_25_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_25_1.append(im)
filter  = ["315_black_frame{}_2_5.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
down_black_25_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_25_2.append(im)
ds_white_25  = []
for filename in glob.glob('ds*white*_2_5*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_25.append(im)
ds_white_25_3d = fft.fftshift(np.log(fftn(np.dstack(ds_white_25))))
print(ds_white_25_3d.shape)
filter  = ["ds_white_frame{}_2_5.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
ds_white_25_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_25_1.append(im)
filter  = ["ds_white_frame{}_2_5.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
ds_white_25_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_25_2.append(im)
filter  = ["ds_white_frame{}_2_5.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
ds_white_25_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_25_3.append(im)
filter  = ["ds_black_frame{}_2_5.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
ds_black_25_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_25_1.append(im)
filter  = ["ds_black_frame{}_2_5.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
ds_black_25_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_25_2.append(im)
up_white_5  = []
for filename in glob.glob('45*white*_5_deg*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_5.append(im)
up_white_5_3d = fft.fftshift(np.log(fftn(np.dstack(up_white_5))))
print(up_white_5_3d.shape)
filter  = ["45_white_frame{}_5_deg.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
up_white_5_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_5_1.append(im)
filter  = ["45_white_frame{}_5_deg.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
up_white_5_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_5_2.append(im)
filter  = ["45_white_frame{}_5_deg.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
up_white_5_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_5_3.append(im)
filter  = ["45_black_frame{}_5_deg.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
up_black_5_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_5_1.append(im)
filter  = ["45_black_frame{}_5_deg.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
up_black_5_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_5_2.append(im)

down_white_5 = []
for filename in glob.glob('315*white*_5_deg*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_5.append(im)
filter  = ["315_white_frame{}_5_deg.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
down_white_5_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_5_1.append(im)
filter  = ["315_white_frame{}_5_deg.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
down_white_5_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_5_2.append(im)
filter  = ["315_white_frame{}_5_deg.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
down_white_5_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_5_3.append(im)
filter  = ["315_black_frame{}_5_deg.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
down_black_5_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_5_1.append(im)
filter  = ["315_black_frame{}_5_deg.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
down_black_5_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_5_2.append(im)
ds_white_5  = []
for filename in glob.glob('ds*white*_5_deg*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_5.append(im)
ds_white_5_3d = fft.fftshift(np.log(fftn(np.dstack(ds_white_5))))
print(ds_white_5_3d.shape)
filter  = ["ds_white_frame{}_5_deg.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
ds_white_5_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_5_1.append(im)
filter  = ["ds_white_frame{}_5_deg.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
ds_white_5_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_5_2.append(im)
filter  = ["ds_white_frame{}_5_deg.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
ds_white_5_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_5_3.append(im)
filter  = ["ds_black_frame{}_5_deg.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
ds_black_5_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_5_1.append(im)
filter  = ["ds_black_frame{}_5_deg.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
ds_black_5_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_5_2.append(im)
up_white_10  = []
for filename in glob.glob('45*white*_10*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_10.append(im)
up_white_10_3d = fft.fftshift(np.log(fftn(np.dstack(up_white_10))))
print(up_white_10_3d.shape)
filter  = ["45_white_frame{}_10_deg.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
up_white_10_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_10_1.append(im)
filter  = ["45_white_frame{}_10_deg.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
up_white_10_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_10_2.append(im)
filter  = ["45_white_frame{}_10_deg.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
up_white_10_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_10_3.append(im)
filter  = ["45_black_frame{}_10_deg.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
up_black_10_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_10_1.append(im)
filter  = ["45_black_frame{}_10_deg.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
up_black_10_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_10_2.append(im)
down_white_10 = []
for filename in glob.glob('315*white*_10*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_10.append(im)
filter  = ["315_white_frame{}_10_deg.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
down_white_10_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_10_1.append(im)
filter  = ["315_white_frame{}_10_deg.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
down_white_10_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_10_2.append(im)
filter  = ["315_white_frame{}_10_deg.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
down_white_10_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_10_3.append(im)
filter  = ["315_black_frame{}_10_deg.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
down_black_10_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_10_1.append(im)
filter  = ["315_black_frame{}_10_deg.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
down_black_10_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_10_2.append(im)
ds_white_10  = []
for filename in glob.glob('ds*white*_10*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_10.append(im)
ds_white_10_3d = fft.fftshift(np.log(fftn(np.dstack(ds_white_10))))
print(ds_white_10_3d.shape)
filter  = ["ds_white_frame{}_10_deg.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
ds_white_10_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_10_1.append(im)
filter  = ["ds_white_frame{}_10_deg.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
ds_white_10_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_10_2.append(im)
filter  = ["ds_white_frame{}_10_deg.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
ds_white_10_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_10_3.append(im)
filter  = ["ds_black_frame{}_10_deg.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
ds_black_10_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_10_1.append(im)
filter  = ["ds_black_frame{}_10_deg.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
ds_black_10_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_10_2.append(im)
up_white_20  = []
for filename in glob.glob('45*white*_20*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_20.append(im)
up_white_20_3d = fft.fftshift(np.log(fftn(np.dstack(up_white_20))))
print(up_white_20_3d.shape)
filter  = ["45_white_frame{}_20_deg.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
up_white_20_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_20_1.append(im)
filter  = ["45_white_frame{}_20_deg.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
up_white_20_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_20_2.append(im)
filter  = ["45_white_frame{}_20_deg.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
up_white_20_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_white_20_3.append(im)
filter  = ["45_black_frame{}_20_deg.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
up_black_20_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_20_1.append(im)
filter  = ["45_black_frame{}_20_deg.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
up_black_20_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    up_black_20_2.append(im)

down_white_20 = []
for filename in glob.glob('315*white*_20*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_20.append(im)
filter  = ["315_white_frame{}_20_deg.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
down_white_20_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_20_1.append(im)
filter  = ["315_white_frame{}_20_deg.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
down_white_20_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_20_2.append(im)
filter  = ["315_white_frame{}_20_deg.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
down_white_20_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_white_20_3.append(im)
filter  = ["315_black_frame{}_20_deg.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
down_black_20_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_20_1.append(im)
filter  = ["315_black_frame{}_20_deg.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
down_black_20_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    down_black_20_2.append(im)
ds_white_20  = []
for filename in glob.glob('ds*white*_20*'): 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_20.append(im)
ds_white_20_3d = fft.fftshift(np.log(fftn(np.dstack(ds_white_20))))
print(ds_white_20_3d.shape)
filter  = ["ds_white_frame{}_20_deg.png".format(i) for i in range(1,6)]
filter = [f for f in filter if isfile(f)]
ds_white_20_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_20_1.append(im)
filter  = ["ds_white_frame{}_20_deg.png".format(i) for i in range(11,16)]
filter = [f for f in filter if isfile(f)]
ds_white_20_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_20_2.append(im)
filter  = ["ds_white_frame{}_20_deg.png".format(i) for i in range(21,26)]
filter = [f for f in filter if isfile(f)]
ds_white_20_3 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_white_20_3.append(im)
filter  = ["ds_black_frame{}_20_deg.png".format(i) for i in range(6,11)]
filter = [f for f in filter if isfile(f)]
ds_black_20_1 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_20_1.append(im)
filter  = ["ds_black_frame{}_20_deg.png".format(i) for i in range(16,21)]
filter = [f for f in filter if isfile(f)]
ds_black_20_2 = []
for filename in filter: 
    im=pygame.image.load(filename)
    im= pygame.surfarray.array2d(im)
    ds_black_20_2.append(im)

#add white background for subsampled
subsampled_background=pygame.image.load('pygame background.png')
subsampled_background= [pygame.surfarray.array2d(subsampled_background)]*4
#add white for subsampling
#0
#up
up_im_white_1_0=pygame.image.load('45_white_frame1_0.png')
up_im_white_1_0= pygame.surfarray.array2d(up_im_white_1_0)
up_im_white_2_0=pygame.image.load('45_white_frame6_0.png')
up_im_white_2_0= pygame.surfarray.array2d(up_im_white_2_0)
up_im_white_3_0=pygame.image.load('45_white_frame11_0.png')
up_im_white_3_0= pygame.surfarray.array2d(up_im_white_3_0)
up_im_white_4_0=pygame.image.load('45_white_frame16_0.png')
up_im_white_4_0= pygame.surfarray.array2d(up_im_white_4_0)
up_im_white_5_0=pygame.image.load('45_white_frame21_0.png')
up_im_white_5_0= pygame.surfarray.array2d(up_im_white_5_0)
#down
down_im_white_1_0=pygame.image.load('315_white_frame1_0.png')
down_im_white_1_0= pygame.surfarray.array2d(down_im_white_1_0)
down_im_white_2_0=pygame.image.load('315_white_frame6_0.png')
down_im_white_2_0= pygame.surfarray.array2d(down_im_white_2_0)
down_im_white_3_0=pygame.image.load('315_white_frame11_0.png')
down_im_white_3_0= pygame.surfarray.array2d(down_im_white_3_0)
down_im_white_4_0=pygame.image.load('315_white_frame16_0.png')
down_im_white_4_0= pygame.surfarray.array2d(down_im_white_4_0)
down_im_white_5_0=pygame.image.load('315_white_frame21_0.png')
down_im_white_5_0= pygame.surfarray.array2d(down_im_white_5_0)
#ds
ds_im_white_1_0=pygame.image.load('ds_white_frame1_0.png')
ds_im_white_1_0= pygame.surfarray.array2d(ds_im_white_1_0)
ds_im_white_2_0=pygame.image.load('ds_white_frame6_0.png')
ds_im_white_2_0= pygame.surfarray.array2d(ds_im_white_2_0)
ds_im_white_3_0=pygame.image.load('ds_white_frame11_0.png')
ds_im_white_3_0= pygame.surfarray.array2d(ds_im_white_3_0)
ds_im_white_4_0=pygame.image.load('ds_white_frame16_0.png')
ds_im_white_4_0= pygame.surfarray.array2d(ds_im_white_4_0)
ds_im_white_5_0=pygame.image.load('ds_white_frame21_0.png')
ds_im_white_5_0= pygame.surfarray.array2d(ds_im_white_5_0)
#1.25
#up
up_im_white_1_125=pygame.image.load('45_white_frame1_125.png')
up_im_white_1_125= pygame.surfarray.array2d(up_im_white_1_125)
up_im_white_2_125=pygame.image.load('45_white_frame6_125.png')
up_im_white_2_125= pygame.surfarray.array2d(up_im_white_2_125)
up_im_white_3_125=pygame.image.load('45_white_frame11_125.png')
up_im_white_3_125= pygame.surfarray.array2d(up_im_white_3_125)
up_im_white_4_125=pygame.image.load('45_white_frame16_125.png')
up_im_white_4_125= pygame.surfarray.array2d(up_im_white_4_125)
up_im_white_5_125=pygame.image.load('45_white_frame21_125.png')
up_im_white_5_125= pygame.surfarray.array2d(up_im_white_5_125)
#down
down_im_white_1_125=pygame.image.load('315_white_frame1_125.png')
down_im_white_1_125= pygame.surfarray.array2d(down_im_white_1_125)
down_im_white_2_125=pygame.image.load('315_white_frame6_125.png')
down_im_white_2_125= pygame.surfarray.array2d(down_im_white_2_125)
down_im_white_3_125=pygame.image.load('315_white_frame11_125.png')
down_im_white_3_125= pygame.surfarray.array2d(down_im_white_3_125)
down_im_white_4_125=pygame.image.load('315_white_frame16_125.png')
down_im_white_4_125= pygame.surfarray.array2d(down_im_white_4_125)
down_im_white_5_125=pygame.image.load('315_white_frame21_125.png')
down_im_white_5_125= pygame.surfarray.array2d(down_im_white_5_125)
#ds
ds_im_white_1_125=pygame.image.load('ds_white_frame1_125.png')
ds_im_white_1_125= pygame.surfarray.array2d(ds_im_white_1_125)
ds_im_white_2_125=pygame.image.load('ds_white_frame6_125.png')
ds_im_white_2_125= pygame.surfarray.array2d(ds_im_white_2_125)
ds_im_white_3_125=pygame.image.load('ds_white_frame11_125.png')
ds_im_white_3_125= pygame.surfarray.array2d(ds_im_white_3_125)
ds_im_white_4_125=pygame.image.load('ds_white_frame16_125.png')
ds_im_white_4_125= pygame.surfarray.array2d(ds_im_white_4_125)
ds_im_white_5_125=pygame.image.load('ds_white_frame21_125.png')
ds_im_white_5_125= pygame.surfarray.array2d(ds_im_white_5_125)
#2.5
#up
up_im_white_1_2_5=pygame.image.load('45_white_frame1_2_5.png')
up_im_white_1_2_5= pygame.surfarray.array2d(up_im_white_1_2_5)
up_im_white_2_2_5=pygame.image.load('45_white_frame6_2_5.png')
up_im_white_2_2_5= pygame.surfarray.array2d(up_im_white_2_2_5)
up_im_white_3_2_5=pygame.image.load('45_white_frame11_2_5.png')
up_im_white_3_2_5= pygame.surfarray.array2d(up_im_white_3_2_5)
up_im_white_4_2_5=pygame.image.load('45_white_frame16_2_5.png')
up_im_white_4_2_5= pygame.surfarray.array2d(up_im_white_4_2_5)
up_im_white_5_2_5=pygame.image.load('45_white_frame21_2_5.png')
up_im_white_5_2_5= pygame.surfarray.array2d(up_im_white_5_2_5)
#down
down_im_white_1_2_5=pygame.image.load('315_white_frame1_2_5.png')
down_im_white_1_2_5= pygame.surfarray.array2d(down_im_white_1_2_5)
down_im_white_2_2_5=pygame.image.load('315_white_frame6_2_5.png')
down_im_white_2_2_5= pygame.surfarray.array2d(down_im_white_2_2_5)
down_im_white_3_2_5=pygame.image.load('315_white_frame11_2_5.png')
down_im_white_3_2_5= pygame.surfarray.array2d(down_im_white_3_2_5)
down_im_white_4_2_5=pygame.image.load('315_white_frame16_2_5.png')
down_im_white_4_2_5= pygame.surfarray.array2d(down_im_white_4_2_5)
down_im_white_5_2_5=pygame.image.load('315_white_frame21_2_5.png')
down_im_white_5_2_5= pygame.surfarray.array2d(down_im_white_5_2_5)
#ds
ds_im_white_1_2_5=pygame.image.load('ds_white_frame1_2_5.png')
ds_im_white_1_2_5= pygame.surfarray.array2d(ds_im_white_1_2_5)
ds_im_white_2_2_5=pygame.image.load('ds_white_frame6_2_5.png')
ds_im_white_2_2_5= pygame.surfarray.array2d(ds_im_white_2_2_5)
ds_im_white_3_2_5=pygame.image.load('ds_white_frame11_2_5.png')
ds_im_white_3_2_5= pygame.surfarray.array2d(ds_im_white_3_2_5)
ds_im_white_4_2_5=pygame.image.load('ds_white_frame16_2_5.png')
ds_im_white_4_2_5= pygame.surfarray.array2d(ds_im_white_4_2_5)
ds_im_white_5_2_5=pygame.image.load('ds_white_frame21_2_5.png')
ds_im_white_5_2_5= pygame.surfarray.array2d(ds_im_white_5_2_5)
#5
#up
up_im_white_1_5_deg=pygame.image.load('45_white_frame1_5_deg.png')
up_im_white_1_5_deg= pygame.surfarray.array2d(up_im_white_1_5_deg)
up_im_white_2_5_deg=pygame.image.load('45_white_frame6_5_deg.png')
up_im_white_2_5_deg= pygame.surfarray.array2d(up_im_white_2_5_deg)
up_im_white_3_5_deg=pygame.image.load('45_white_frame11_5_deg.png')
up_im_white_3_5_deg= pygame.surfarray.array2d(up_im_white_3_5_deg)
up_im_white_4_5_deg=pygame.image.load('45_white_frame16_5_deg.png')
up_im_white_4_5_deg= pygame.surfarray.array2d(up_im_white_4_5_deg)
up_im_white_5_5_deg=pygame.image.load('45_white_frame21_5_deg.png')
up_im_white_5_5_deg= pygame.surfarray.array2d(up_im_white_5_5_deg)
#down
down_im_white_1_5_deg=pygame.image.load('315_white_frame1_5_deg.png')
down_im_white_1_5_deg= pygame.surfarray.array2d(down_im_white_1_5_deg)
down_im_white_2_5_deg=pygame.image.load('315_white_frame6_5_deg.png')
down_im_white_2_5_deg= pygame.surfarray.array2d(down_im_white_2_5_deg)
down_im_white_3_5_deg=pygame.image.load('315_white_frame11_5_deg.png')
down_im_white_3_5_deg= pygame.surfarray.array2d(down_im_white_3_5_deg)
down_im_white_4_5_deg=pygame.image.load('315_white_frame16_5_deg.png')
down_im_white_4_5_deg= pygame.surfarray.array2d(down_im_white_4_5_deg)
down_im_white_5_5_deg=pygame.image.load('315_white_frame21_5_deg.png')
down_im_white_5_5_deg= pygame.surfarray.array2d(down_im_white_5_5_deg)
#ds
ds_im_white_1_5_deg=pygame.image.load('ds_white_frame1_5_deg.png')
ds_im_white_1_5_deg= pygame.surfarray.array2d(ds_im_white_1_5_deg)
ds_im_white_2_5_deg=pygame.image.load('ds_white_frame6_5_deg.png')
ds_im_white_2_5_deg= pygame.surfarray.array2d(ds_im_white_2_5_deg)
ds_im_white_3_5_deg=pygame.image.load('ds_white_frame11_5_deg.png')
ds_im_white_3_5_deg= pygame.surfarray.array2d(ds_im_white_3_5_deg)
ds_im_white_4_5_deg=pygame.image.load('ds_white_frame16_5_deg.png')
ds_im_white_4_5_deg= pygame.surfarray.array2d(ds_im_white_4_5_deg)
ds_im_white_5_5_deg=pygame.image.load('ds_white_frame21_5_deg.png')
ds_im_white_5_5_deg= pygame.surfarray.array2d(ds_im_white_5_5_deg)
#10
#up
up_im_white_1_10_deg=pygame.image.load('45_white_frame1_10_deg.png')
up_im_white_1_10_deg= pygame.surfarray.array2d(up_im_white_1_10_deg)
up_im_white_2_10_deg=pygame.image.load('45_white_frame6_10_deg.png')
up_im_white_2_10_deg= pygame.surfarray.array2d(up_im_white_2_10_deg)
up_im_white_3_10_deg=pygame.image.load('45_white_frame11_10_deg.png')
up_im_white_3_10_deg= pygame.surfarray.array2d(up_im_white_3_10_deg)
up_im_white_4_10_deg=pygame.image.load('45_white_frame16_10_deg.png')
up_im_white_4_10_deg= pygame.surfarray.array2d(up_im_white_4_10_deg)
up_im_white_5_10_deg=pygame.image.load('45_white_frame21_10_deg.png')
up_im_white_5_10_deg= pygame.surfarray.array2d(up_im_white_5_10_deg)
#down
down_im_white_1_10_deg=pygame.image.load('315_white_frame1_10_deg.png')
down_im_white_1_10_deg= pygame.surfarray.array2d(down_im_white_1_10_deg)
down_im_white_2_10_deg=pygame.image.load('315_white_frame6_10_deg.png')
down_im_white_2_10_deg= pygame.surfarray.array2d(down_im_white_2_10_deg)
down_im_white_3_10_deg=pygame.image.load('315_white_frame11_10_deg.png')
down_im_white_3_10_deg= pygame.surfarray.array2d(down_im_white_3_10_deg)
down_im_white_4_10_deg=pygame.image.load('315_white_frame16_10_deg.png')
down_im_white_4_10_deg= pygame.surfarray.array2d(down_im_white_4_10_deg)
down_im_white_5_10_deg=pygame.image.load('315_white_frame21_10_deg.png')
down_im_white_5_10_deg= pygame.surfarray.array2d(down_im_white_5_10_deg)
#ds
ds_im_white_1_10_deg=pygame.image.load('ds_white_frame1_10_deg.png')
ds_im_white_1_10_deg= pygame.surfarray.array2d(ds_im_white_1_10_deg)
ds_im_white_2_10_deg=pygame.image.load('ds_white_frame6_10_deg.png')
ds_im_white_2_10_deg= pygame.surfarray.array2d(ds_im_white_2_10_deg)
ds_im_white_3_10_deg=pygame.image.load('ds_white_frame11_10_deg.png')
ds_im_white_3_10_deg= pygame.surfarray.array2d(ds_im_white_3_10_deg)
ds_im_white_4_10_deg=pygame.image.load('ds_white_frame16_10_deg.png')
ds_im_white_4_10_deg= pygame.surfarray.array2d(ds_im_white_4_10_deg)
ds_im_white_5_10_deg=pygame.image.load('ds_white_frame21_10_deg.png')
ds_im_white_5_10_deg= pygame.surfarray.array2d(ds_im_white_5_10_deg)
#20
#up
up_im_white_1_20_deg=pygame.image.load('45_white_frame1_20_deg.png')
up_im_white_1_20_deg= pygame.surfarray.array2d(up_im_white_1_20_deg)
up_im_white_2_20_deg=pygame.image.load('45_white_frame6_20_deg.png')
up_im_white_2_20_deg= pygame.surfarray.array2d(up_im_white_2_20_deg)
up_im_white_3_20_deg=pygame.image.load('45_white_frame11_20_deg.png')
up_im_white_3_20_deg= pygame.surfarray.array2d(up_im_white_3_20_deg)
up_im_white_4_20_deg=pygame.image.load('45_white_frame16_20_deg.png')
up_im_white_4_20_deg= pygame.surfarray.array2d(up_im_white_4_20_deg)
up_im_white_5_20_deg=pygame.image.load('45_white_frame21_20_deg.png')
up_im_white_5_20_deg= pygame.surfarray.array2d(up_im_white_5_20_deg)
#down
down_im_white_1_20_deg=pygame.image.load('315_white_frame1_20_deg.png')
down_im_white_1_20_deg= pygame.surfarray.array2d(down_im_white_1_20_deg)
down_im_white_2_20_deg=pygame.image.load('315_white_frame6_20_deg.png')
down_im_white_2_20_deg= pygame.surfarray.array2d(down_im_white_2_20_deg)
down_im_white_3_20_deg=pygame.image.load('315_white_frame11_20_deg.png')
down_im_white_3_20_deg= pygame.surfarray.array2d(down_im_white_3_20_deg)
down_im_white_4_20_deg=pygame.image.load('315_white_frame16_20_deg.png')
down_im_white_4_20_deg= pygame.surfarray.array2d(down_im_white_4_20_deg)
down_im_white_5_20_deg=pygame.image.load('315_white_frame21_20_deg.png')
down_im_white_5_20_deg= pygame.surfarray.array2d(down_im_white_5_20_deg)
#ds
ds_im_white_1_20_deg=pygame.image.load('ds_white_frame1_20_deg.png')
ds_im_white_1_20_deg= pygame.surfarray.array2d(ds_im_white_1_20_deg)
ds_im_white_2_20_deg=pygame.image.load('ds_white_frame6_20_deg.png')
ds_im_white_2_20_deg= pygame.surfarray.array2d(ds_im_white_2_20_deg)
ds_im_white_3_20_deg=pygame.image.load('ds_white_frame11_20_deg.png')
ds_im_white_3_20_deg= pygame.surfarray.array2d(ds_im_white_3_20_deg)
ds_im_white_4_20_deg=pygame.image.load('ds_white_frame16_20_deg.png')
ds_im_white_4_20_deg= pygame.surfarray.array2d(ds_im_white_4_20_deg)
ds_im_white_5_20_deg=pygame.image.load('ds_white_frame21_20_deg.png')
ds_im_white_5_20_deg= pygame.surfarray.array2d(ds_im_white_5_20_deg)


#CONDITIONS PER VELOCITY
#STEPS TO DO:
#DO STACKS.DONE.
#DO TF.DONE
#DO SF.DONE
#DO A=SF*TF
#DO A= A/np.amax(A)
#DO ps_0_smooth,cp,ss fft.fftshift(np.dot(A,Z))
#DO for each condition np.sum(ps_0)
#0
    
##px_dg = np.array([22.5,18.2])
##px_dg1 = np.resize(px_dg,(1152,25))
##print(px_dg1)
smooth_0_up = up_white_0_3d
y1 = np.divide(smooth_0_up[:,2],0.33)
x1 =smooth_0_up[:,0:2]
x1= np.mean(x1,axis=1)
print(x1)
smooth_0_down = fft.fftshift(np.log(fftn(np.dstack(down_white_0))))
y2 = np.divide(smooth_0_down[:,2],0.33)
x2 = smooth_0_down[:,0:2]
x2= np.mean(x2,axis=1)
print(x2)
smooth_0_ds = fft.fftshift(np.log(fftn(np.dstack(ds_white_0))))
y3 = np.divide(smooth_0_ds[:,2],0.33)
x3 = smooth_0_ds[:,0:2]
x3= np.mean(x3,axis=1)
print(x3)
print(smooth_0_down.shape)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('Smooth-0-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('Smooth-0-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('Smooth-0-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('Smooth-0-Power vs Sensitivity.png')
plt.show()
smooth_sp_0_up = np.sum(sp1)
smooth_sp_0_down = np.sum(sp2)
smooth_sp_0_ds = np.sum(sp3)
contrast_polarity_up_0_ls = [*up_white_0_1,*up_black_0_1,*up_white_0_2,*up_black_0_2,*up_white_0_3]
contrast_polarity_0_up = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_up_0_ls))))
y1 = np.divide(contrast_polarity_0_up[:,2],0.33)
x1 = contrast_polarity_0_up[:,0:2]
x1= np.mean(x1,axis=1)
print(contrast_polarity_0_up.shape)
contrast_polarity_down_0_ls = [*down_white_0_1,*down_black_0_1,*down_white_0_2,*down_black_0_2,*down_white_0_3]
contrast_polarity_0_down = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_down_0_ls))))
y2 = np.divide(contrast_polarity_0_down[:,2],0.33)
x2 = contrast_polarity_0_down[:,0:2]
x2= np.mean(x2,axis=1)
contrast_polarity_ds_0_ls = [*ds_white_0_1,*ds_black_0_1,*ds_white_0_2,*ds_black_0_2,*ds_white_0_3]
contrast_polarity_0_ds = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_ds_0_ls))))
y3 = np.divide(contrast_polarity_0_ds[:,2],0.33)
x3 = contrast_polarity_0_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-0-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-0-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-0-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-0-Power vs Sensitivity.png')
plt.show()
cp_sp_0_up = np.sum(sp1)
cp_sp_0_down = np.sum(sp2)
cp_sp_0_ds = np.sum(sp3)
subsampled_0_up_ls= [up_im_white_1_0,*subsampled_background,up_im_white_2_0,*subsampled_background,up_im_white_3_0,*subsampled_background,up_im_white_4_0,*subsampled_background,up_im_white_5_0,*subsampled_background]
subsampled_0_up = fft.fftshift(np.log(fftn(np.dstack(subsampled_0_up_ls))))
print(subsampled_0_up.shape)
y1 = np.divide(subsampled_0_up[:,2],0.33)
x1 = subsampled_0_up[:,0:2]
x1= np.mean(x1,axis=1)
subsampled_0_down_ls =[down_im_white_1_0,*subsampled_background,down_im_white_2_0,*subsampled_background,down_im_white_3_0,*subsampled_background,down_im_white_4_0,*subsampled_background,down_im_white_5_0,*subsampled_background]
subsampled_0_down = fft.fftshift(np.log(fftn(np.dstack(subsampled_0_down_ls))))
y2 = np.divide(subsampled_0_down[:,2],0.33)
x2 = subsampled_0_down[:,0:2]
x2= np.mean(x2,axis=1)
subsampled_0_ds_ls =[ds_im_white_1_0,*subsampled_background,ds_im_white_2_0,*subsampled_background,ds_im_white_3_0,*subsampled_background,ds_im_white_4_0,*subsampled_background,ds_im_white_5_0,*subsampled_background]
subsampled_0_ds = fft.fftshift(np.log(fftn(np.dstack(subsampled_0_ds_ls))))
y3 = np.divide(subsampled_0_ds[:,2],0.33)
x3 = subsampled_0_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-0-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-0-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-0-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-0-Power vs Sensitivity.png')
plt.show()
ss_sp_0_up = np.sum(sp1)
ss_sp_0_down = np.sum(sp2)
ss_sp_0_ds = np.sum(sp3)

#1.25
smooth_125_up = up_white_125_3d
y1 = np.divide(smooth_125_up[:,2],0.33)
x1 = smooth_125_up[:,0:2]
x1= np.mean(x1,axis=1)
smooth_125_down = fft.fftshift(np.log(fftn(np.dstack(down_white_125))))
y2 = np.divide(smooth_125_down[:,2],0.33)
x2 = smooth_125_down[:,0:2]
x2= np.mean(x2,axis=1)
smooth_125_ds = fft.fftshift(np.log(fftn(np.dstack(ds_white_125))))
y3 = np.divide(smooth_125_ds[:,2],0.33)
x3 = smooth_125_ds[:,0:2]
x3= np.mean(x3,axis=1)
print(smooth_125_down.shape)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-125-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-125-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-125-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-125-Power vs Sensitivity.png')
plt.show()
smooth_sp_125_up = np.sum(sp1)
smooth_sp_125_down = np.sum(sp2)
smooth_sp_125_ds = np.sum(sp3)
contrast_polarity_up_125_ls = [*up_white_125_1,*up_black_125_1,*up_white_125_2,*up_black_125_2,*up_white_125_3]
contrast_polarity_125_up = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_up_125_ls))))
y1 = np.divide(contrast_polarity_125_up[:,2],0.33)
x1 = contrast_polarity_125_up[:,0:2]
x1= np.mean(x1,axis=1)
print(contrast_polarity_125_up.shape)
contrast_polarity_down_125_ls = [*down_white_125_1,*down_black_125_1,*down_white_125_2,*down_black_125_2,*down_white_125_3]
contrast_polarity_125_down = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_down_125_ls))))
y2 = np.divide(contrast_polarity_125_down[:,2],0.33)
x2 = contrast_polarity_125_down[:,0:2]
x2= np.mean(x2,axis=1)
contrast_polarity_ds_125_ls = [*ds_white_125_1,*ds_black_125_1,*ds_white_125_2,*ds_black_125_2,*ds_white_125_3]
contrast_polarity_125_ds = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_ds_125_ls))))
y3 = np.divide(contrast_polarity_125_ds[:,2],0.33)
x3 = contrast_polarity_125_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-125-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-125-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-125-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-125-Power vs Sensitivity.png')
plt.show()
cp_sp_125_up = np.sum(sp1)
cp_sp_125_down = np.sum(sp2)
cp_sp_125_ds = np.sum(sp3)
subsampled_125_up_ls= [up_im_white_1_125,*subsampled_background,up_im_white_2_125,*subsampled_background,up_im_white_3_125,*subsampled_background,up_im_white_4_125,*subsampled_background,up_im_white_5_125,*subsampled_background]
subsampled_125_up = fft.fftshift(np.log(fftn(np.dstack(subsampled_125_up_ls))))
print(subsampled_125_up.shape)
y1 = np.divide(subsampled_125_up[:,2],0.33)
x1 = subsampled_125_up[:,0:2]
x1= np.mean(x1,axis=1)
subsampled_125_down_ls =[down_im_white_1_125,*subsampled_background,down_im_white_2_125,*subsampled_background,down_im_white_3_125,*subsampled_background,down_im_white_4_125,*subsampled_background,down_im_white_5_125,*subsampled_background]
subsampled_125_down = fft.fftshift(np.log(fftn(np.dstack(subsampled_125_down_ls))))
y2 = np.divide(subsampled_125_down[:,2],0.33)
x2 = subsampled_125_down[:,0:2]
x2= np.mean(x2,axis=1)
subsampled_125_ds_ls =[ds_im_white_1_125,*subsampled_background,ds_im_white_2_125,*subsampled_background,ds_im_white_3_125,*subsampled_background,ds_im_white_4_125,*subsampled_background,ds_im_white_5_125,*subsampled_background]
subsampled_125_ds = fft.fftshift(np.log(fftn(np.dstack(subsampled_125_ds_ls))))
y3 = np.divide(subsampled_125_ds[:,2],0.33)
x3 = subsampled_125_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-125-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-125-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-125-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-125-Power vs Sensitivity.png')
plt.show()
ss_sp_125_up = np.sum(sp1)
ss_sp_125_down = np.sum(sp2)
ss_sp_125_ds = np.sum(sp3)
#2.5
smooth_25_up = up_white_25_3d
y1 = np.divide(smooth_25_up[:,2],0.33)
x1 = smooth_25_up[:,0:2]
x1= np.mean(x1,axis=1)
smooth_25_down = fft.fftshift(np.log(fftn(np.dstack(down_white_25))))
y2 = np.divide(smooth_25_down[:,2],0.33)
x2 = smooth_25_down[:,0:2]
x2= np.mean(x2,axis=1)
smooth_25_ds = fft.fftshift(np.log(fftn(np.dstack(ds_white_25))))
y3 = np.divide(smooth_25_ds[:,2],0.33)
x3 = smooth_25_ds[:,0:2]
x3= np.mean(x3,axis=1)
print(smooth_25_down.shape)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-25-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-25-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-25-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-25-Power vs Sensitivity.png')
plt.show()
smooth_sp_25_up = np.sum(sp1)
smooth_sp_25_down = np.sum(sp2)
smooth_sp_25_ds = np.sum(sp3)
contrast_polarity_up_25_ls = [*up_white_25_1,*up_black_25_1,*up_white_25_2,*up_black_25_2,*up_white_25_3]
contrast_polarity_25_up = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_up_25_ls))))
y1 = np.divide(contrast_polarity_25_up[:,2],0.33)
x1 = contrast_polarity_25_up[:,0:2]
x1= np.mean(x1,axis=1)
print(contrast_polarity_25_up.shape)
contrast_polarity_down_25_ls = [*down_white_25_1,*down_black_25_1,*down_white_25_2,*down_black_25_2,*down_white_25_3]
contrast_polarity_25_down = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_down_25_ls))))
y2 = np.divide(contrast_polarity_25_down[:,2],0.33)
x2 = contrast_polarity_25_down[:,0:2]
x2= np.mean(x2,axis=1)
contrast_polarity_ds_25_ls = [*ds_white_25_1,*ds_black_25_1,*ds_white_25_2,*ds_black_25_2,*ds_white_25_3]
contrast_polarity_25_ds = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_ds_25_ls))))
y3 = np.divide(contrast_polarity_25_ds[:,2],0.33)
x3 = contrast_polarity_25_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-25-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-25-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-25-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-25-Power vs Sensitivity.png')
plt.show()
cp_sp_25_up = np.sum(sp1)
cp_sp_25_down = np.sum(sp2)
cp_sp_25_ds = np.sum(sp3)
subsampled_2_5_up_ls= [up_im_white_1_2_5,*subsampled_background,up_im_white_2_2_5,*subsampled_background,up_im_white_3_2_5,*subsampled_background,up_im_white_4_2_5,*subsampled_background,up_im_white_5_2_5,*subsampled_background]
subsampled_2_5_up = fft.fftshift(np.log(fftn(np.dstack(subsampled_2_5_up_ls))))
print(subsampled_2_5_up.shape)
y1 = np.divide(subsampled_2_5_up[:,2],0.33)
x1 = subsampled_2_5_up[:,0:2]
x1= np.mean(x1,axis=1)
subsampled_2_5_down_ls =[down_im_white_1_2_5,*subsampled_background,down_im_white_2_2_5,*subsampled_background,down_im_white_3_2_5,*subsampled_background,down_im_white_4_2_5,*subsampled_background,down_im_white_5_2_5,*subsampled_background]
subsampled_2_5_down = fft.fftshift(np.log(fftn(np.dstack(subsampled_2_5_down_ls))))
y2 = np.divide(subsampled_2_5_down[:,2],0.33)
x2 = subsampled_2_5_down[:,0:2]
x2= np.mean(x2,axis=1)
subsampled_2_5_ds_ls =[ds_im_white_1_2_5,*subsampled_background,ds_im_white_2_2_5,*subsampled_background,ds_im_white_3_2_5,*subsampled_background,ds_im_white_4_2_5,*subsampled_background,ds_im_white_5_2_5,*subsampled_background]
subsampled_2_5_ds = fft.fftshift(np.log(fftn(np.dstack(subsampled_2_5_ds_ls))))
y3 = np.divide(subsampled_2_5_ds[:,2],0.33)
x3 = subsampled_2_5_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-25-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-25-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-25-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-25-Power vs Sensitivity.png')
plt.show()
ss_sp_25_up = np.sum(sp1)
ss_sp_25_down = np.sum(sp2)
ss_sp_25_ds = np.sum(sp3)
#5
smooth_5_up = up_white_5_3d
y1 = np.divide(smooth_5_up[:,2],0.33)
x1 = smooth_5_up[:,0:2]
x1= np.mean(x1,axis=1)
smooth_5_down = fft.fftshift(np.log(fftn(np.dstack(down_white_5))))
y1 = np.divide(smooth_5_down[:,2],0.33)
x2 = smooth_5_down[:,0:2]
x2= np.mean(x2,axis=1)
smooth_5_ds = fft.fftshift(np.log(fftn(np.dstack(ds_white_5))))
y1 = np.divide(smooth_5_ds[:,2],0.33)
x3 = smooth_5_ds[:,0:2]
x3= np.mean(x3,axis=1)
print(smooth_5_down.shape)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-5-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-5-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-5-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-5-Power vs Sensitivity.png')
plt.show()
smooth_sp_5_up = np.sum(sp1)
smooth_sp_5_down = np.sum(sp2)
smooth_sp_5_ds = np.sum(sp3)
contrast_polarity_up_5_ls = [*up_white_5_1,*up_black_5_1,*up_white_5_2,*up_black_5_2,*up_white_5_3]
contrast_polarity_5_up = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_up_5_ls))))
y1 = np.divide(contrast_polarity_5_up[:,2],0.33)
x1 = contrast_polarity_5_up[:,0:2]
x1= np.mean(x1,axis=1)
print(contrast_polarity_5_up.shape)
contrast_polarity_down_5_ls = [*down_white_5_1,*down_black_5_1,*down_white_5_2,*down_black_5_2,*down_white_5_3]
contrast_polarity_5_down = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_down_5_ls))))
y2 = np.divide(contrast_polarity_5_down[:,2],0.33)
x2 = contrast_polarity_5_down[:,0:2]
x2= np.mean(x2,axis=1)
contrast_polarity_ds_5_ls = [*ds_white_5_1,*ds_black_5_1,*ds_white_5_2,*ds_black_5_2,*ds_white_5_3]
contrast_polarity_5_ds = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_ds_5_ls))))
y3 = np.divide(contrast_polarity_5_ds[:,2],0.33)
x3 = contrast_polarity_5_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-5-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-5-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-5-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-5-Power vs Sensitivity.png')
plt.show()
cp_sp_5_up = np.sum(sp1)
cp_sp_5_down = np.sum(sp2)
cp_sp_5_ds = np.sum(sp3)
subsampled_5_up_ls= [up_im_white_1_5_deg,*subsampled_background,up_im_white_2_5_deg,*subsampled_background,up_im_white_3_5_deg,*subsampled_background,up_im_white_4_5_deg,*subsampled_background,up_im_white_5_5_deg,*subsampled_background]
subsampled_5_up = fft.fftshift(np.log(fftn(np.dstack(subsampled_5_up_ls))))
print(subsampled_5_up.shape)
y1 = np.divide(subsampled_5_up[:,2],0.33)
x1 = subsampled_5_up[:,0:2]
x1= np.mean(x1,axis=1)
subsampled_5_down_ls =[down_im_white_1_0,*subsampled_background,down_im_white_2_5_deg,*subsampled_background,down_im_white_3_5_deg,*subsampled_background,down_im_white_4_5_deg,*subsampled_background,down_im_white_5_5_deg,*subsampled_background]
subsampled_5_down = fft.fftshift(np.log(fftn(np.dstack(subsampled_5_down_ls))))
y2 = np.divide(subsampled_5_down[:,2],0.33)
x2 = subsampled_5_down[:,0:2]
x2= np.mean(x2,axis=1)
subsampled_5_ds_ls =[ds_im_white_1_0,*subsampled_background,ds_im_white_2_5_deg,*subsampled_background,ds_im_white_3_5_deg,*subsampled_background,ds_im_white_4_5_deg,*subsampled_background,ds_im_white_5_5_deg,*subsampled_background]
subsampled_5_ds = fft.fftshift(np.log(fftn(np.dstack(subsampled_5_ds_ls))))
y3 = np.divide(subsampled_5_ds[:,2],0.33)
x3 = subsampled_5_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-5-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-5-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-5-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-5-Power vs Sensitivity.png')
plt.show()
ss_sp_5_up = np.sum(sp1)
ss_sp_5_down = np.sum(sp2)
ss_sp_5_ds = np.sum(sp3)

#10
smooth_10_up = up_white_10_3d
y1 = np.divide(smooth_10_up[:,2],0.33)
x1 = smooth_10_up[:,0:2]
x1= np.mean(x1,axis=1)
smooth_10_down = fft.fftshift(np.log(fftn(np.dstack(down_white_10))))
y2 = np.divide(smooth_10_down[:,2],0.33)
x2 = smooth_10_down[:,0:2]
x2= np.mean(x2,axis=1)
smooth_10_ds = fft.fftshift(np.log(fftn(np.dstack(ds_white_10))))
y3 = np.divide(smooth_10_ds[:,2],0.33)
x3 = smooth_10_ds[:,0:2]
x3= np.mean(x3,axis=1)
print(smooth_10_down.shape)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-10-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-10-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-10-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-10-Power vs Sensitivity.png')
plt.show()
smooth_sp_10_up = np.sum(sp1)
smooth_sp_10_down = np.sum(sp2)
smooth_sp_10_ds = np.sum(sp3)
contrast_polarity_up_10_ls = [*up_white_10_1,*up_black_10_1,*up_white_10_2,*up_black_10_2,*up_white_10_3]
contrast_polarity_10_up = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_up_10_ls))))
y1 = np.divide(contrast_polarity_10_up[:,2],0.33)
x1 = contrast_polarity_10_up[:,0:2]
x1= np.mean(x1,axis=1)
print(contrast_polarity_10_up.shape)
contrast_polarity_down_10_ls = [*down_white_10_1,*down_black_10_1,*down_white_10_2,*down_black_10_2,*down_white_10_3]
contrast_polarity_10_down = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_down_10_ls))))
y2 = np.divide(contrast_polarity_10_down[:,2],0.33)
x2 = contrast_polarity_10_down[:,0:2]
x2= np.mean(x2,axis=1)
contrast_polarity_ds_10_ls = [*ds_white_10_1,*ds_black_10_1,*ds_white_10_2,*ds_black_10_2,*ds_white_10_3]
contrast_polarity_10_ds = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_ds_10_ls))))
y3 = np.divide(contrast_polarity_10_ds[:,2],0.33)
x3 = contrast_polarity_10_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-10-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-10-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-10-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-10-Power vs Sensitivity.png')
plt.show()
cp_sp_10_up = np.sum(sp1)
cp_sp_10_down = np.sum(sp2)
cp_sp_10_ds = np.sum(sp3)
subsampled_10_up_ls= [up_im_white_1_10_deg,*subsampled_background,up_im_white_2_10_deg,*subsampled_background,up_im_white_3_10_deg,*subsampled_background,up_im_white_4_10_deg,*subsampled_background,up_im_white_5_10_deg,*subsampled_background]
subsampled_10_up = fft.fftshift(np.log(fftn(np.dstack(subsampled_10_up_ls))))
print(subsampled_10_up.shape)
y1 = np.divide(subsampled_10_up[:,2],0.33)
x1 = subsampled_10_up[:,0:2]
x1= np.mean(x1,axis=1)
subsampled_10_down_ls =[down_im_white_1_10_deg,*subsampled_background,down_im_white_2_10_deg,*subsampled_background,down_im_white_3_10_deg,*subsampled_background,down_im_white_4_10_deg,*subsampled_background,down_im_white_5_10_deg,*subsampled_background]
subsampled_10_down = fft.fftshift(np.log(fftn(np.dstack(subsampled_10_down_ls))))
y2 = np.divide(subsampled_10_down[:,2],0.33)
x2 = subsampled_10_down[:,0:2]
x2= np.mean(x2,axis=1)
subsampled_10_ds_ls =[ds_im_white_1_10_deg,*subsampled_background,ds_im_white_2_10_deg,*subsampled_background,ds_im_white_3_10_deg,*subsampled_background,ds_im_white_4_10_deg,*subsampled_background,ds_im_white_5_10_deg,*subsampled_background]
subsampled_10_ds = fft.fftshift(np.log(fftn(np.dstack(subsampled_10_ds_ls))))
y3 = subsampled_10_ds[:,2]
x3 = subsampled_10_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-10-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-10-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-10-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-10-Power vs Sensitivity.png')
plt.show()
ss_sp_10_up = np.sum(sp1)
ss_sp_10_down = np.sum(sp2)
ss_sp_10_ds = np.sum(sp3)
#20
smooth_20_up = up_white_20_3d
y1 = np.divide(smooth_20_up[:,2],0.33)
x1 = smooth_20_up[:,0:2]
x1= np.mean(x1,axis=1)
smooth_20_down = fft.fftshift(np.log(fftn(np.dstack(down_white_20))))
y2 = np.divide(smooth_20_down[:,2],0.33)
x2 = smooth_20_down[:,0:2]
x2= np.mean(x2,axis=1)
smooth_20_ds = fft.fftshift(np.log(fftn(np.dstack(ds_white_20))))
y3 = np.divide(smooth_20_ds[:,2],0.33)
x3 = smooth_20_ds[:,0:2]
x3= np.mean(x3,axis=1)
print(smooth_20_down.shape)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-20-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-20-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-20-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('S-20-Power vs Sensitivity.png')
plt.show()
smooth_sp_20_up = np.sum(sp1)
smooth_sp_20_down = np.sum(sp2)
smooth_sp_20_ds = np.sum(sp3)
contrast_polarity_up_20_ls = [*up_white_20_1,*up_black_20_1,*up_white_20_2,*up_black_20_2,*up_white_20_3]
contrast_polarity_20_up = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_up_20_ls))))
y1 = np.divide(contrast_polarity_20_up[:,2],0.33)
x1 = contrast_polarity_20_up[:,0:2]
x1= np.mean(x1,axis=1)
print(contrast_polarity_20_up.shape)
contrast_polarity_down_20_ls = [*down_white_20_1,*down_black_20_1,*down_white_20_2,*down_black_20_2,*down_white_20_3]
contrast_polarity_20_down = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_down_20_ls))))
y2 = np.divide(contrast_polarity_20_down[:,2],0.33)
x2 = contrast_polarity_20_down[:,0:2]
x2= np.mean(x2,axis=1)
contrast_polarity_ds_20_ls = [*ds_white_20_1,*ds_black_20_1,*ds_white_20_2,*ds_black_20_2,*ds_white_20_3]
contrast_polarity_20_ds = fft.fftshift(np.log(fftn(np.dstack(contrast_polarity_ds_20_ls))))
y3 = np.divide(contrast_polarity_20_ds[:,2],0.33)
x3 = contrast_polarity_20_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-20-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-20-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-20-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('CP-20-Power vs Sensitivity.png')
plt.show()
cp_sp_20_up = np.sum(sp1)
cp_sp_20_down = np.sum(sp2)
cp_sp_20_ds = np.sum(sp3)
subsampled_20_up_ls= [up_im_white_1_20_deg,*subsampled_background,up_im_white_2_20_deg,*subsampled_background,up_im_white_3_20_deg,*subsampled_background,up_im_white_4_20_deg,*subsampled_background,up_im_white_5_20_deg,*subsampled_background]
subsampled_20_up = fft.fftshift(np.log(fftn(np.dstack(subsampled_20_up_ls))))
print(subsampled_20_up.shape)
y1 = np.divide(subsampled_20_up[:,2],0.33)
x1 = subsampled_20_up[:,0:2]
x1= np.mean(x1,axis=1)
subsampled_20_down_ls =[down_im_white_1_20_deg,*subsampled_background,down_im_white_2_20_deg,*subsampled_background,down_im_white_3_20_deg,*subsampled_background,down_im_white_4_20_deg,*subsampled_background,down_im_white_5_20_deg,*subsampled_background]
subsampled_20_down = fft.fftshift(np.log(fftn(np.dstack(subsampled_20_down_ls))))
y2 = np.divide(subsampled_20_down[:,2],0.33)
x2 = subsampled_20_down[:,0:2]
x2= np.mean(x2,axis=1)
subsampled_20_ds_ls =[ds_im_white_1_20_deg,*subsampled_background,ds_im_white_2_20_deg,*subsampled_background,ds_im_white_3_20_deg,*subsampled_background,ds_im_white_4_20_deg,*subsampled_background,ds_im_white_5_20_deg,*subsampled_background]
subsampled_20_ds = fft.fftshift(np.log(fftn(np.dstack(subsampled_20_ds_ls))))
y3 = np.divide(subsampled_20_ds[:,2],0.33)
x3 = subsampled_20_ds[:,0:2]
x3= np.mean(x3,axis=1)
up_magnitude_spectrum2 = 10*np.log(np.abs(y1))
print(up_magnitude_spectrum2.shape)
up_magnitude_spectrum3 = 10*np.log(np.abs(y2))
up_magnitude_spectrum4 = 10*np.log(np.abs(y3))
up_magnitude_spectrum5 = 10*np.log(np.abs(x1))
up_magnitude_spectrum6 = 10*np.log(np.abs(x2))
up_magnitude_spectrum7 = 10*np.log(np.abs(x3))
transform1 = scaler.fit_transform(up_magnitude_spectrum2)
transform2 = scaler.fit_transform(up_magnitude_spectrum3)
transform3 = scaler.fit_transform(up_magnitude_spectrum4)
transform4 = scaler.fit_transform(up_magnitude_spectrum5)
transform5 = scaler.fit_transform(up_magnitude_spectrum6)
transform6 = scaler.fit_transform(up_magnitude_spectrum7)
extent2 = [- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2),- np.amax(up_magnitude_spectrum2), np.amax(up_magnitude_spectrum2)]
extent3 = [- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3),- np.amax(up_magnitude_spectrum3), np.amax(up_magnitude_spectrum3)]
extent4 = [- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4),- np.amax(up_magnitude_spectrum4), np.amax(up_magnitude_spectrum4)]
extent5 = [- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5),- np.amax(up_magnitude_spectrum5), np.amax(up_magnitude_spectrum5)]
extent6 = [- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6),- np.amax(up_magnitude_spectrum6), np.amax(up_magnitude_spectrum6)]
extent7 = [- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7),- np.amax(up_magnitude_spectrum7), np.amax(up_magnitude_spectrum7)]
plt.subplot(131),plt.imshow(transform1, cmap = 'viridis',aspect='auto', extent = extent2)
plt.title('Up')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform2, cmap = 'viridis',aspect='auto', extent = extent3)
plt.title('Down')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform3, cmap = 'viridis',aspect='auto', extent = extent4)
plt.title('Difference Spectrum')
plt.xlabel('Temporal Frequency (cycles/s)',fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-20-Temporal Frequency- power.png',bbox_inches="tight")
plt.show()
plt.subplot(131),plt.imshow(transform4, cmap = 'viridis',aspect='auto', extent = extent5)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(transform5, cmap = 'viridis',aspect='auto', extent = extent6)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(transform6, cmap = 'viridis',aspect='auto', extent = extent7)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)',fontsize=8),plt.ylabel('Spatial Frequency (cycles/°)',fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-20-Spatial Frequency- power.png',bbox_inches="tight")
plt.show()
A1 = fft.fftshift(10*np.log(np.abs(np.multiply(x1,y1))))
A2 = fft.fftshift(10*np.log(np.abs(np.multiply(x2,y2))))
A3 = fft.fftshift(10*np.log(np.abs(np.multiply(x3,y3))))
extenta1 = [- np.amax(A1), np.amax(A1),- np.amax(A1), np.amax(A1)]
extenta2 = [- np.amax(A2), np.amax(A2),- np.amax(A2), np.amax(A2)]
extenta3 = [- np.amax(A3), np.amax(A3),- np.amax(A3), np.amax(A3)]
A1_transform = scaler.fit_transform(A1)
A2_transform = scaler.fit_transform(A2)
A3_transform = scaler.fit_transform(A3)
plt.subplot(131),plt.imshow(A1_transform, cmap = 'viridis',aspect='auto', extent = extenta1)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(A2_transform, cmap = 'viridis',aspect='auto', extent = extenta2)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(A3_transform, cmap = 'viridis',aspect='auto', extent = extenta3)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-20-SF vs TF- power.png')
plt.show()
A1 = np.multiply(x1,y1)
A2 = np.multiply(x2,y2)
A3 = np.multiply(x3,y3)
sp1 = fft.fftshift(10*np.log(np.abs(np.multiply(A1,Z))))
sp2 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp3 = fft.fftshift(10*np.log(np.abs(np.multiply(A2,Z))))
sp1_extent = [- np.amax(sp1), np.amax(sp1),- np.amax(sp1), np.amax(sp1)]
sp2_extent = [- np.amax(sp2), np.amax(sp2),- np.amax(sp2), np.amax(sp2)]
sp3_extent = [- np.amax(sp3), np.amax(sp3),- np.amax(sp3), np.amax(sp3)]
plt.subplot(131),plt.imshow(sp1, cmap = 'viridis',aspect='auto', extent = sp1_extent)
plt.title('Up')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(132),plt.imshow(sp2, cmap = 'viridis',aspect='auto', extent = sp2_extent)
plt.title('Down')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.subplot(133),plt.imshow(sp3, cmap = 'viridis',aspect='auto', extent = sp3_extent)
plt.title('Difference Spectrum')
plt.xlabel('Spatial Frequency (cycles/°)', fontsize=8),plt.ylabel('Temporal Frequency (cycles/s)', fontsize=8)
clb = plt.colorbar()
clb.ax.set_ylabel('Amplitude(AU)',fontsize=8)
plt.tight_layout()
plt.savefig('SS-20-Power vs Sensitivity.png')
plt.show()
ss_sp_20_up = np.sum(sp1)
ss_sp_20_down = np.sum(sp2)
ss_sp_20_ds = np.sum(sp3)

#GRAPH TO COMPARE WITH HUMAN RESOLUTION THRESHOLD
#0
smooth_sp_0_up = np.reciprocal(smooth_sp_0_up)
smooth_sp_0_down = np.reciprocal(smooth_sp_0_down)
smooth_sp_0_ds = np.reciprocal(smooth_sp_0_ds)
cp_sp_0_up = np.reciprocal(cp_sp_0_up)
cp_sp_0_down = np.reciprocal(cp_sp_0_down)
cp_sp_0_ds = np.reciprocal(cp_sp_0_ds)
ss_sp_0_up = np.reciprocal(ss_sp_0_up)
ss_sp_0_down = np.reciprocal(ss_sp_0_down)
ss_sp_0_ds = np.reciprocal(ss_sp_0_ds)
#1.25
smooth_sp_125_up = np.reciprocal(smooth_sp_125_up)
smooth_sp_125_down = np.reciprocal(smooth_sp_125_down)
smooth_sp_125_ds = np.reciprocal(smooth_sp_125_ds)
cp_sp_125_up = np.reciprocal(cp_sp_125_up)
cp_sp_125_down = np.reciprocal(cp_sp_125_down)
cp_sp_125_ds = np.reciprocal(cp_sp_125_ds)
ss_sp_125_up = np.reciprocal(ss_sp_125_up)
ss_sp_125_down = np.reciprocal(ss_sp_125_down)
ss_sp_125_ds = np.reciprocal(ss_sp_125_ds)
#2.5
smooth_sp_25_up = np.reciprocal(smooth_sp_25_up)
smooth_sp_25_down = np.reciprocal(smooth_sp_25_down)
smooth_sp_25_ds = np.reciprocal(smooth_sp_25_ds)
cp_sp_25_up = np.reciprocal(cp_sp_25_up)
cp_sp_25_down = np.reciprocal(cp_sp_25_down)
cp_sp_25_ds = np.reciprocal(cp_sp_25_ds)
ss_sp_25_up = np.reciprocal(ss_sp_25_up)
ss_sp_25_down = np.reciprocal(ss_sp_25_down)
ss_sp_25_ds = np.reciprocal(ss_sp_25_ds)
#5
smooth_sp_5_up = np.reciprocal(smooth_sp_5_up)
smooth_sp_5_down = np.reciprocal(smooth_sp_5_down)
smooth_sp_5_ds = np.reciprocal(smooth_sp_5_ds)
cp_sp_5_up = np.reciprocal(cp_sp_5_up)
cp_sp_5_down = np.reciprocal(cp_sp_5_down)
cp_sp_5_ds = np.reciprocal(cp_sp_5_ds)
ss_sp_5_up = np.reciprocal(ss_sp_5_up)
ss_sp_5_down = np.reciprocal(ss_sp_5_down)
ss_sp_5_ds = np.reciprocal(ss_sp_5_ds)
#10
smooth_sp_10_up = np.reciprocal(smooth_sp_10_up)
smooth_sp_10_down = np.reciprocal(smooth_sp_10_down)
smooth_sp_10_ds = np.reciprocal(smooth_sp_10_ds)
cp_sp_10_up = np.reciprocal(cp_sp_10_up)
cp_sp_10_down = np.reciprocal(cp_sp_10_down)
cp_sp_10_ds = np.reciprocal(cp_sp_10_ds)
ss_sp_10_up = np.reciprocal(ss_sp_10_up)
ss_sp_10_down = np.reciprocal(ss_sp_10_down)
ss_sp_10_ds = np.reciprocal(ss_sp_10_ds)
#20
smooth_sp_20_up = np.reciprocal(smooth_sp_20_up)
smooth_sp_20_down = np.reciprocal(smooth_sp_20_down)
smooth_sp_20_ds = np.reciprocal(smooth_sp_20_ds)
cp_sp_20_up = np.reciprocal(cp_sp_20_up)
cp_sp_20_down = np.reciprocal(cp_sp_20_down)
cp_sp_20_ds = np.reciprocal(cp_sp_20_ds)
ss_sp_20_up = np.reciprocal(ss_sp_20_up)
ss_sp_20_down = np.reciprocal(ss_sp_20_down)
ss_sp_20_ds = np.reciprocal(ss_sp_20_ds)

#PLOT GRAPH
#a.Organise y variables
#up
y1_up= [smooth_sp_0_up,smooth_sp_125_up,smooth_sp_25_up,smooth_sp_5_up,smooth_sp_10_up,smooth_sp_20_up]
print(y1_up)
y2_up= [cp_sp_0_up,cp_sp_125_up,cp_sp_25_up,cp_sp_5_up,cp_sp_10_up,cp_sp_20_up]
print(y2_up)
y3_up= ss_sp_0_up,ss_sp_125_up,ss_sp_25_up,ss_sp_5_up,ss_sp_10_up,ss_sp_20_up
print(y3_up)
#down
y1_down= [smooth_sp_0_down,smooth_sp_125_down,smooth_sp_25_down,smooth_sp_5_down,smooth_sp_10_down,smooth_sp_20_down]
print(y1_down)
y2_down= [cp_sp_0_down,cp_sp_125_down,cp_sp_25_down,cp_sp_5_down,cp_sp_10_down,cp_sp_20_down]
print(y2_down)
y3_down= [ss_sp_0_down,ss_sp_125_down,ss_sp_25_down,ss_sp_5_down,ss_sp_10_down,ss_sp_20_down]
print(y3_down)
#ds
y1_ds= [smooth_sp_0_ds,smooth_sp_125_ds,smooth_sp_25_ds,smooth_sp_5_ds,smooth_sp_10_ds,smooth_sp_20_ds]
print(y1_ds)
y2_ds= [cp_sp_0_ds,cp_sp_125_ds,cp_sp_25_ds,cp_sp_5_ds,cp_sp_10_ds,cp_sp_20_ds]
print(y2_ds)
y3_ds= [ss_sp_0_ds,ss_sp_125_ds,ss_sp_25_ds,ss_sp_5_ds,ss_sp_10_ds,ss_sp_20_ds]
print(y3_ds)
#names for x axis
names = ['0','1.25','2.5','5','10','20']
fig,ax = plt.subplots()
ax.plot(names,y2_up,'b^-',label= 'Contrast polarity reversal')
ax.plot(names,y1_up,'ks-',label='Unmodulated')
ax.plot(names,y3_up,'ro-',label='Subsampled')
legend = ax.legend(loc='upper right')
ax.set_ylabel('Inverse estimated relative resolvability (AU)')
ax.set_xlabel('Speed (°s⁻¹)')
plt.tight_layout()
plt.savefig('Inverse estimated relative resolvability -up.png')
plt.show()
fig,ax = plt.subplots()
ax.plot(names,y2_down,'b^-',label= 'Contrast polarity reversal')
ax.plot(names,y1_down,'ks-',label='Unmodulated')
ax.plot(names,y3_down,'ro-',label='Subsampled')
legend = ax.legend(loc='upper right')
ax.set_ylabel('Inverse estimated relative resolvability (AU)')
ax.set_xlabel('Speed (°s⁻¹)')
plt.tight_layout()
plt.savefig('Inverse estimated relative resolvability -down.png')
plt.show()
fig,ax = plt.subplots()
ax.plot(names,y2_ds,'b^-',label= 'Contrast polarity reversal')
ax.plot(names,y1_ds,'ks-',label='Unmodulated')
ax.plot(names,y3_ds,'ro-',label='Subsampled')
legend = ax.legend(loc='upper right')
ax.set_ylabel('Inverse estimated relative resolvability (AU)')
ax.set_xlabel('Speed (°s⁻¹)')
plt.tight_layout()
plt.savefig('Inverse estimated relative resolvability -ds.png')
plt.show()
