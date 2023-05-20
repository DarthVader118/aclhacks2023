import cv2

# this is code
cam = cv2.VideoCapture("hackathonThings/test2.mp4")

while True:
    ret, image = cam.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Imagetest',image)
    k = cv2.waitKey(1)
    if k != -1:
        break

cv2.imwrite('testimage.jpg', image)
cam.release()
cv2.destroyAllWindows()