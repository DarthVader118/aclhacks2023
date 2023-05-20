

#ROAD LANE DETECTION

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math



def grey(image):
  #convert to grayscale
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  #Apply Gaussian Blur --> Reduce noise and smoothen image
def gauss(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

  #outline the strongest gradients in the image --> this is where lines in the image are
def canny(image):
    edges = cv2.Canny(image,150,150)
    return edges

def region(image):
    height, width = image.shape
    #isolate the gradients that correspond to the lane lines
    triangle = np.array([
                       [(100, height), (475, 325), (width, height)]
                       ])
    #create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    #create a mask (triangle that isolates the region of interest in our image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            #draw lines on a black image
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

def average(image, lines):
    left = []
    right = []

    if lines is not None:
      for line in lines:
        print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        print(parameters)
        slope = parameters[0]
        y_int = parameters[1]
        #lines on the right have positive slope, and lines on the left have neg slope
        
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
            
    #takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    
    return np.array([left_line, right_line])

def make_points(image, average):
    print(average)
    
    try:
        slope, y_int = average
    except TypeError:
        slope, y_int = 1,1
    
    
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


'''##### DETECTING lane lines in image ######'''
cam = cv2.VideoCapture("2023-aclhacks/hackathonThings/challenge.mp4")
oldAvg=np.array([1,1])
while True:
    ret,image = cam.read()
    copy = np.copy(image)
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    copy =gauss(gray)

    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([75, 100, 100],dtype="uint8")
    upper_yellow = np.array([255, 165, 165], dtype="uint8")
    
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray, 150, 175)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray, mask_yw)
    
    edges = canny(mask_yw_image)
    
    isolated = region(edges)
    
    #DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array, 
    lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=100)
    averaged_lines = average(copy, lines)

    if math.isnan(averaged_lines[0][0])==True or math.isnan(averaged_lines[1][0])==True:
        averaged_lines=oldAvg
    oldAvg=averaged_lines
    black_lines = display_lines(copy, averaged_lines)

    #taking wighted sum of original image and lane lines image
    lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
    
    
    cv2.imshow("iso", isolated)
    cv2.imshow("final",lanes)
    cv2.imshow("test", mask_yw_image)
    k = cv2.waitKey(1)
    if k != -1:
        break
    
cam.release()
cv2.destroyAllWindows()