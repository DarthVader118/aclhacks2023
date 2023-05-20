import cv2
import numpy as np
# this is code
cam = cv2.VideoCapture("hackathonThings/test2.mp4")

while True:
    ret, image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    edge = cv2.Canny(blur,100,200)
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([75, 100, 100],dtype="uint8")
    upper_yellow = np.array([255, 165, 165], dtype="uint8")
    
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray, 150, 1650)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray, mask_yw)
    

    
    cv2.imshow("og", image)
    cv2.imshow('gray',blur)
    
    cv2.imshow("mapper", mask_yw_image)
    #cv2.imshow("test", lines)
    
    k = cv2.waitKey(1)
    if k != -1:
        break

cv2.imwrite('testimage.jpg', image)
cam.release()
cv2.destroyAllWindows()