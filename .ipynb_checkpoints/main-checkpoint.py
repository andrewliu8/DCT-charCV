# Import required packages
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import os
import math

# Mention the installed location of Tesseract-OCR in your system
def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
# Read image from which text needs to be extracted
img = cv2.imread("data/0i3/20m_good.jpeg")
os.chdir(r'/uavdocker/DCT-charCV')
# Preprocessing the image starts
 
# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.jpg",gray)
#blurred = cv2.GaussianBlur(gray,(7,7),0)
# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
cv2.imwrite("thresh.jpg",thresh1)
#cv2.rectangle(img,(2840,1920),(2920,2100),(0,255,0),2)


# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
 
# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 2)
cv2.imwrite('dilation.jpg',dilation)

# cropped = dilation[1880: 2140,2800: 2960]
# cv2.imwrite('cropped.jpg',cropped)

# height, width = thresh1.shape[:2]
# # get the center coordinates of the image to create the 2D rotation matrix
# center = (width/2, height/2)
# center = (130,80)
# using cv2.getRotationMatrix2D() to get the rotation matrix
# rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-85, scale=1)

# # rotate the image using cv2.warpAffine
# rotated_image = cv2.warpAffine(src=cropped, M=rotate_matrix, dsize=(260, 160))



contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
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
#print(im2.shape)
 
# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
count = 0
# cropped = thresh1[2000:2150,2680:2900]
# cv2.imwrite(f'cropped.jpg',cropped)
# text = pytesseract.image_to_string(cropped)
# print(text)
# cv2.rectangle(im2,(2680,2000),(2900,2150),(0,255,0),2)
# [1880: 2140,2800: 2960]
lst = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    count = count+1
    if w < 25 or h < 25 or w > 50 or h > 50:
        cv2.rectangle(dilation,(x,y), (x + w, y + h), (0, 0, 0), -1)
        continue
    # Drawing a rectangle on copied image
    # cv2.imshow('sample.jpg', im2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #print(f"{x} {y} {w} {h}")
    coords = []
    coords.append(x)
    coords.append(y)
    coords.append(w)
    coords.append(h)

    lst.append(coords)
    
    #cv2.rectangle(dilation,(x, y), (x + w, y + h), (0, 255, 0), 4)
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
    # Cropping the text block for giving input to OCR
    cropped = thresh1[y:y + h, x:x + w]
    cv2.imwrite(f'cropped{count}.jpg',cropped)
    
    # Open the file in append mode
    file = open("recognized.txt", "a")
    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped)
    # Appending the text into file
    file.write(text)
    file.write("\n")
     
    # Close the file
    file.close()
cv2.imwrite('after.jpg',dilation)
lst.sort(key = lambda x:(x[0],x[1]))
# for i in lst:
#     print(i)
numC = len(lst)
win = []
for i in range(numC):
    for j in range(i+1,numC):
        for k in range(j+1, numC):
            if lst[j][1] > lst[i][1] and lst[k][1] < lst[j][1]:
                continue
            if lst[j][1] < lst[i][1] and lst[k][1] > lst[j][1]:
                continue
            centerI = (lst[i][0]+lst[i][2]/2,lst[i][1]+lst[i][3]/2)
            centerJ = (lst[j][0]+lst[j][2]/2,lst[j][1]+lst[j][3]/2)
            centerK = (lst[k][0]+lst[k][2]/2,lst[k][1]+lst[k][3]/2)
            dist1 = dist(centerI[0],centerI[1],centerJ[0],centerJ[1])
            dist2 = dist(centerK[0],centerK[1],centerJ[0],centerJ[1])
            if dist1 > 36 or dist2 > 36:
                continue
            win.append(lst[i])
            win.append(lst[j])
            win.append(lst[k])
cv2.imwrite('rects.jpg',img)
cv2.imwrite('rects2.jpg',im2)
for k in win:
    print(k)
slope = []
slope.append((win[1][1]-win[0][1])/(win[1][0]-win[0][0]))
slope.append((win[2][1]-win[1][1])/(win[2][0]-win[1][0]))
slope.append((win[2][1]-win[0][1])/(win[2][0]-win[0][0]))
avg = -(slope[2] + slope[1] + slope[0]) / 3
angle = math.degrees(math.atan(avg))
# print(avg)
print(angle)

rangeX1 = win[0][0]-10
rangeX2 = win[2][0]+10+win[2][2]
if ((win[2][0]+10+win[2][2])-(win[0][0]-10))%2 == 1:
    rangeX1 = rangeX1-1
rangeY1 = 0
ranegY2 = 0

if avg >= 0:
    rangeY2 = win[0][1]+10+win[0][3]
    rangeY1 = win[2][1]-10
    if ((win[0][1]+10+win[0][3])-(win[2][1]-10))%2 == 1:
        rangeY1 = rangeY1-1
else:
    rangeY2 = win[2][1]+10+win[2][3]
    rangeY1 = win[0][1]-10
    if ((win[2][1]+10+win[2][3])-(win[0][1]-10))%2 == 1:
        rangeY1 = rangeY1-1
        
print(f"{rangeX1} {rangeX2} {rangeY1} {rangeY2}")

corners = np.array([[2900,2067],[2910,1970],[2920, 1971],[2910,2068]])

cv2.fillPoly(dilation, pts = [corners], color =(255,255,255))

cv2.imwrite("withunder.jpg",dilation)


cropped_img = dilation[rangeY1:rangeY2,rangeX1:rangeX2]
h, w = cropped_img.shape
rad = math.radians(angle)
newW = int((w-h*math.tan(rad))/(math.cos(rad)*(1-math.tan(rad)*math.tan(rad))))+1
newH = int((w-newW*math.cos(rad))/math.sin(rad))+1
print(newW)
print(newH)
print(cropped_img.shape)
cv2.imwrite("cropped_img.jpg",cropped_img)

centerX = int((rangeX2-rangeX1)/2)
centerY = int((rangeY2-rangeY1)/2)
print((centerX,centerY))

# using cv2.getRotationMatrix2D() to get the rotation matrix
rotate_matrix = cv2.getRotationMatrix2D(center=(centerX,centerY), angle=-angle, scale=1)
# # rotate the image using cv2.warpAffine
rotated_image = cv2.warpAffine(src=cropped_img, M=rotate_matrix, dsize=(newW,newH))
cv2.imwrite('rotated.jpg',rotated_image)
text = pytesseract.image_to_string(rotated_image)
print(text)