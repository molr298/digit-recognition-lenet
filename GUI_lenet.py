import pygame,sys
from numpy.lib.type_check import imag
from pygame.locals import *
import numpy as np
from numpy import testing
from pygame import image
from keras.models import load_model
import cv2
from tensorflow.python.keras.backend import constant
WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
IMGSAVE = False
PREDICT = True
MODEL = load_model("bestmodel_lenet.h5")
LABELS = {0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",
        5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"}
num_x_cord = []
num_y_cord = []

pygame.init()

FONT = pygame.font.SysFont("freesansbold", 18)
DISPLAY = pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))

pygame.display.set_caption("Handwritten Digits (Press C to clear the surface)")

iswriting = False
img_cnt = 0
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            x_cord, y_cord = event.pos
            pygame.draw.circle(DISPLAY, WHITE, (x_cord,y_cord),4,0)
            num_x_cord.append(x_cord)
            num_y_cord.append(y_cord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if not num_x_cord or not num_y_cord:
                continue
            num_x_cord = sorted(num_x_cord)
            num_y_cord = sorted(num_y_cord)

            rect_min_x, rect_max_x = max(num_x_cord[0] - 5, 0), min(WINDOWSIZEX, num_x_cord[-1] + 5)
            rect_min_y, rect_max_y = max(num_y_cord[0] - 5, 0), min(num_y_cord[-1] + 5,WINDOWSIZEX)

            num_x_cord = []
            num_y_cord = []

            img_arr = np.array(pygame.PixelArray(DISPLAY))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
            if IMGSAVE:
                cv2.imwrite("image.png")
                img_cnt +=1
            if PREDICT:
                image = cv2.resize(img_arr,(32,32))
                image = np.pad(image,(10,10),'constant',constant_values = 0)
                image = cv2.resize(image,(32,32)) / 255
                print(image.shape)
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,32,32,1)))])
                print(label)

        if event.type == KEYDOWN:
            if event.key == K_c:
                DISPLAY.fill(BLACK)

        pygame.display.update()

