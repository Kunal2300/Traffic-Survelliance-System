import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image 
import pytesseract
import imutils
import easyocr

def yoloToPascalVoc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]

co="3 0.491 0.512605 0.740667 0.782713"
img=Image.open("creta.png")

def numberplateDetection(co,img):
    m=co.split(" ")
    m=img.crop(yoloToPascalVoc(float(m[1]),float(m[2]),float(m[3]),float(m[4]),img.width,img.height))
    m.save("temp.png")
    img=cv2.imread("temp.png")

    # now convert image into gray scaling
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite("result/1output.png",gray)

        # first use a kernal
    filter=cv2.bilateralFilter(gray,11,17,17)
    cv2.imwrite("result/2output.png",filter) # save filter image

    # get edged image we have to use the canny function from the cv2
    edged=cv2.Canny(filter,30,200)
    cv2.imwrite("result/3output.png",edged) #save the edgedcar image

    # we have to find rectangle bounding box containing numberplate
    # basically we want a rectangular contours 
    cnts=cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # now usinf imutils we can grab contours fro image 
    contours=imutils.grab_contours(cnts)
    # now sort the grabbed contours 
    sortedContours=sorted(contours,key=cv2.contourArea,reverse=True)[:10]

    l=None

    for i in sortedContours:
        # check that a countter is rectangle or polygon if it is we can extract it diorectly
        approx=cv2.approxPolyDP(i,10,True)
        if len(approx)==4 :
            l=approx
            # print(l)
            if (abs(l[1][0][0]-l[0][0][0])>abs(l[0][0][1]-l[2][0][1])):
                break

    # now we have our needed polygon image and lets see its configuration 
    # print(l)
    # lets draw contours on og image
    x,y,w,h=cv2.boundingRect(l) 
    bbCar=cv2.rectangle(img, (x+1, y+1), (x+w+1, y+h+1), (0, 0, 255), 2)
    cv2.imwrite("result/4output.png",bbCar)
    
    # created a mask 
    mask=np.zeros(gray.shape,np.uint8)

    # apply bounding box on image 
    plate=cv2.drawContours(mask,[l],0,255,-1)

    cv2.imwrite("result/5output.png",plate)
    # extract numberplate
    plate=cv2.bitwise_and(img,img,mask=mask)

    # lets grab the image 
    x,y,w,h=cv2.boundingRect(l)
    region_of_interest = plate[y+4:y+h+4, x+4:x+w+4]
    cv2.imwrite("result/6output.png",region_of_interest)

    ocr_result=pytesseract.image_to_string(region_of_interest)

    if ocr_result=="":
        plateImage=cv2.imread("result/7output.png")
        grayPlate=cv2.cvtColor(region_of_interest,cv2.COLOR_BGR2GRAY)

        # save Image
        cv2.imwrite("result/8output.png",grayPlate)
        thres,bwplate=cv2.threshold(grayPlate,100,230,cv2.THRESH_BINARY)
        # print(bwplate)
        # save image 
        cv2.imwrite("result/9output.png",bwplate)
        # plt.imshow(Image.open("numberplates/bwPlate.png"))
        ocr_result=pytesseract.image_to_string(bwplate)
        print(ocr_result)
    else:
        print(ocr_result)

numberplateDetection(co,img)
