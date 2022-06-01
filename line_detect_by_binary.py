# coding=gbk
import cv2
from filter import *
from kf import kf
img_path = './line/4/'
img_num = 188
my_kf = kf()
save = False
save_video = False
if save:
    train_data = []
    train_name = []

if save_video:
    size = (640,480)   
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc1 = cv2.VideoWriter_fourcc(*'mp4v')
    path_bin = "binary.mp4"
    path_bin_line = "binary_line.mp4"
    out_bin = cv2.VideoWriter( path_bin, fourcc, 10.0, size,0)
    out_bin_line= cv2.VideoWriter( path_bin_line, fourcc1, 10.0, size,0)

for i in range(img_num):
    img_name = img_path + f'{48+i:03d}.jpg'
    print(img_name)
    img_raw = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
    width = 640
    height = 480
    img_raw = cv2.resize(img_raw, (640, 480))
    img = filter1(img_raw)
    ret,binary=get_binary1(img)
    pt1 = []
    pt2 = []
    for j in range(height//20):
        if j*20<height:
            pts = [x for x in range(width)  if binary[j*20][x]>0]
            s = len(pts)
            if s>50 and s<120:
                pt1 = (pts[s//2],j*20)
                break
    
    for k in range(height//20):
        if height - k*20>j*20 + 10 and height - k*20<height*0.9 :
            pts = [x for x in range(width)  if binary[height - k*20][x]>0]
            s = len(pts)
            if s>50 and s<120:
                pt2 = (pts[s//2],height - k*20)
                break
    if pt1 != []  and pt2 != []:
        cv2.line(img, pt1, pt2, (255),3)
    
    cv2.imshow('THRESH_BINRY',binary)
    cv2.imshow('result',img)
    if save_video:
        out_bin.write(binary)
        out_bin_line.write(img)
    cv2.waitKey(1)
cv2.waitKey(0)    
cv2.destroyAllWindows()
if save:
    fp = open("label.txt","w+")
    fp1 = open("train.txt","w+")
    length = len(train_name)
    for i in range(length):
        print(train_data[i][0],train_data[i][1],file=fp)
        print(train_name[i],file=fp1)
    fp.close()
    fp1.close()
if save_video:
    out_bin.release()
    out_bin_line.release()