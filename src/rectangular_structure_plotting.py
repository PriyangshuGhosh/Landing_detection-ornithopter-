import numpy as np
import cv2

bredth = 300

def plot_rectangular_structure(image):

cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    if not ret:
        print("Camera couldn't be accessed ")
        break
    height , width , _ = frame.shape
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray , (5,5) , 1.2)
    edges = cv2.Canny(blur,50,150)
    
