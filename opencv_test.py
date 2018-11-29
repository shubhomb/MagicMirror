import cv2
import numpy as np

# Get filepath
filepath = 'C:/Users/shubh/Pictures/harry.jpeg'
harry = cv2.imread(filepath)
filepath = 'C:/Users/shubh/Pictures/messi.jpg'
messi = cv2.imread(filepath)
# Image details
print (harry.dtype)
print (harry.size)
print (harry.shape)
print (harry.item(10,10,2))


# Some modification
for i in range (0,harry.shape[0]):
    for j in range (0,harry.shape[1]):
        for k in range (0,1):
            harry.itemset((i,j,k),10)

messi.resize(harry.shape)
dst = cv2.addWeighted(harry,0.5,messi,0.5,0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Show image
cv2.imshow('image',harry)
cv2.waitKey(0)

