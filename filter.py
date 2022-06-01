# coding=gbk
import cv2
import numpy as np
def filter1(img):
    img = cv2.GaussianBlur(img,(3,3),0)
    return img
    
def filter2(img):
    blurred = cv2.pyrMeanShiftFiltering(img, 10, 50)  # 均值迁移滤波
    return blurred

def filter3(img):
    hpf = img - cv2.GaussianBlur(img, (21, 21), 3)+127 #high pass filter 高通滤波
    return hpf

def get_binary(img):
    return cv2.threshold(img,160,255,cv2.THRESH_BINARY )#+ cv2.THRESH_OTSU

def get_binary1(img):#k-means
    pixel_values = img.reshape((-1, 1))
    # convert to float
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 0.2)
    k = 2
    _, labels, (centers) = cv2.kmeans(data=pixel_values, K=k, bestLabels=None, criteria=criteria, attempts=10,flags=cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    # convert colors
    for i in range(2):
        centers[i]=255 if centers[i]>100 else 0 
    segmented_image = centers[labels.flatten()]
    # back to the original dimension
    segmented_image = segmented_image.reshape(img.shape)
    # core = np.ones((7, 7), np.uint8)
    # segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, core)
    kernel = np.ones((20,20),np.uint8)
    segmented_image = cv2.erode(segmented_image,kernel,iterations = 1)
    return _,segmented_image


if __name__=='__main__':
    print("please run 'line_detect.py'")