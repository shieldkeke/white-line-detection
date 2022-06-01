# coding=gbk
import cv2
import numpy as np
from filter import *
from kf import kf
img_path = './line/4/'#/line/4/�ļ�����ͼƬΪ0.05s�ɼ�һ�Σ�/line/�ļ�����Ϊ1s�ɼ�һ��
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
    path_canny = "canny.mp4"
    path_hough = "hough.mp4"
    out_canny = cv2.VideoWriter( path_canny, fourcc, 10.0, size,0)
    out_houghline = cv2.VideoWriter( path_hough, fourcc1, 10.0, size,0)

for i in range(img_num):
    img_name = img_path + f'{48+i:03d}.jpg'
    print(img_name)
    img_raw = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
    img_raw = cv2.resize(img_raw, (640, 480))
    img = filter1(img_raw)
    #edges = cv2.Canny(img, 90, 180, apertureSize = 3)
    edges = cv2.Canny(img, 150, 220, apertureSize = 3)#��ֵ�������½�
    lines = cv2.HoughLines(edges,1,np.pi/180,110) #��ֵ��ͼ�񣬾��뾫�ȣ��ǶȾ��ȣ�����ֵ

    result = img.copy()
    height = result.shape[0]
    pts = [0]#���������ӽ���ֱ��
    cnt_lines = 0
    avr_pt = 0
    avr_theta = 0
    if not lines is None:
        for line in lines:
            rho = line[0][0]  #��һ��Ԫ���Ǿ���rho
            theta= line[0][1] #�ڶ���Ԫ���ǽǶ�theta
            if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #��ֱֱ��
                if min([abs(pt-rho/np.cos(theta)) for pt in pts])>60:#�����ظ�
                    pts.append(rho/np.cos(theta))
                    pt1 = (int(rho/np.cos(theta)),0) #��ֱ�����һ�еĽ���
                    pt2 = (int(rho/np.cos(theta)+np.tan(-theta)*height),height)
                    #pt2 = (int((rho-height*np.sin(theta))/np.cos(theta)),height)#��ֱ�������һ�еĽ���
                    cv2.line(result, pt1, pt2, (255),3) # ���ư���
                    cnt_lines+=1    
                    avr_pt += int(rho/np.cos(theta))#���������ֱ�ߣ���¼�е�ͽǶ�
                    avr_theta += theta if theta < (np.pi/4. ) else theta-np.pi #ƽ���Ƕȣ�ȡС��
    #��������������ˣ��͸��£�����Ԥ��
    if cnt_lines == 2:
        avr_pt = avr_pt//2
        avr_theta = avr_theta / 2
        print(avr_pt)
        print(avr_theta)

        pt1 = (int(avr_pt), 0)
        pt2 = (int(avr_pt+np.tan(-avr_theta)*height),height)
        cv2.line(result,pt1,pt2,(0),2)
        if save:
            train_data.append([avr_pt,avr_theta])
            train_name.append(img_name)
        x, _ = my_kf.predict()
        cv2.circle(result, (avr_pt, 2), 1, (0), 2)
        cv2.circle(result, (x,4), 1, (255), 2)#Ԥ����ʵ�ʵ�Ա�
        my_kf.refresh(avr_pt, avr_theta)
    else:
        x, theta = my_kf.predict()
        print(x)
        print(theta)
        cv2.circle(result, (x,2), 1, (255), 2)
        pass

    cv2.imshow('Result', result)
    print(f'{cnt_lines} lines detected')
    cv2.imshow('Canny', edges)
    if save_video:
        out_canny.write(edges)
        out_houghline.write(result)
    cv2.waitKey(10)

if save_video:
    out_canny.release()
    out_houghline.release()

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
