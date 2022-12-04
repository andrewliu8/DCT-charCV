# Import required packages
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import os

# Mention the installed location of Tesseract-OCR in your system
 
# Read image from which text needs to be extracted
img = cv2.imread("data/0i3/20m_good.jpeg")
 
# Preprocessing the image starts
 
# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray,(7,7),0)
# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

cv2.rectangle(img,(2840,1920),(2920,2100),(0,255,0),2)


# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
 
# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
os.chdir(r'/uavdocker/DCT-charCV')
cv2.imwrite('dilation.jpg',dilation)

cropped = dilation[1880: 2140,2800: 2960]
cv2.imwrite('cropped.jpg',cropped)

results = pytesseract.image_to_osd(thresh1, output_type=Output.DICT)
print(results)

# height, width = thresh1.shape[:2]
# # get the center coordinates of the image to create the 2D rotation matrix
# center = (width/2, height/2)
center = (130,80)
# using cv2.getRotationMatrix2D() to get the rotation matrix
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-85, scale=1)
 
# rotate the image using cv2.warpAffine
rotated_image = cv2.warpAffine(src=cropped, M=rotate_matrix, dsize=(260, 160))
cropped_rotate = rotated_image[0:80,20:130]
cv2.imwrite('rotated.jpg',cropped_rotate)
text = pytesseract.image_to_string(cropped_rotate)
print(text)


contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)

# cropped = imgPoly[600: 900, 320: 1000]
# text = pytesseract.image_to_string(cropped)
# print(text)
# cv2.imshow('blah',imgPoly)
# cv2.waitKey(0)

# Creating a copy of image
im2 = img.copy()
 
# A text file is created and flushed
file = open("recognized.txt", "w+")
file.write("")
file.close()
print(im2.shape)
 
# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
# count = 0
# cropped = thresh1[2000:2150,2680:2900]
# cv2.imwrite(f'cropped.jpg',cropped)
# text = pytesseract.image_to_string(cropped)
# print(text)
# cv2.rectangle(im2,(2680,2000),(2900,2150),(0,255,0),2)

# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     if w <= 60 and h <= 60 or x < 2650 or x > 2900 or y < 2000 or y > 2050:
#         continue
#     # Drawing a rectangle on copied image
#     rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     # cv2.imshow('sample.jpg', im2)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     print(f"{x} {y} {w} {h}")
#     cv2.rectangle(thresh1,(x, y), (x + w, y + h), (0, 255, 0), 4)
     
#     # Cropping the text block for giving input to OCR
#     cropped = thresh1[y:y + h, x:x + w]
#     cv2.imwrite(f'cropped{count}.jpg',cropped)
#     count = count+1
#     # Open the file in append mode
#     file = open("recognized.txt", "a")
     
#     # Apply OCR on the cropped image
#     text = pytesseract.image_to_string(cropped)
#     # Appending the text into file
#     file.write(text)
#     file.write("\n")
     
#     # Close the file
#     file.close()
cv2.imwrite('rects.jpg',img)
