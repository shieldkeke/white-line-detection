import torch
import torch.nn as nn
import cv2
import numpy as np
from model import MobileNet,MobileNetV3_Small
import matplotlib.pyplot as plt
class Line_detector():
    def __init__(self,v='v3',_save=False) -> None:
        
        vision = v
        if vision=="v3":
            self.modelPath = 'v3-0.322.pt'
            model = MobileNetV3_Small()
        elif vision=="v1":
            self.modelPath = 'v1-best.pt'
            model = MobileNet()
        self.save = _save 
        if _save:
            size = (640,480)   
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            path_mobile = 'mobile.mp4'
            self.out = cv2.VideoWriter( path_mobile, fourcc, 10.0, size)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            model = model.cuda()
            device = torch.device('cuda')    
        else:
            device = torch.device('cpu') 
 
        print('Loading weights...')    
        model.load_state_dict(torch.load(self.modelPath, map_location=device), strict=False)
        print(f'{self.modelPath} loaded.')
        self.net = model.eval()

    def detect(self,img_raw):
        height = img_raw.shape[0]
        width = img_raw.shape[1]
        img = np.transpose(img_raw,(2,0,1))
        img = np.expand_dims(np.array(img,dtype=np.float32)/255,0)
        with torch.no_grad():
            input = torch.from_numpy(img)
            if self.cuda:
                input = input.cuda()
            output = self.net(input).cpu().numpy()
            output = output[0]
            output[0] = max(min((output[0]*600),width),0) #restore dimension
            output[1] = output[1]/4

            pt1 = (int(output[0]), 0)
            pt2 = (int(output[0]+np.tan(-output[1])*height),height)
            result = cv2.line(img_raw,pt1,pt2,(255,255,255),2)
            result = cv2.circle(result, pt1, 1, (0,255,0), 2)
            cv2.imshow('result',result)
            if self.save:
                self.out.write(result)

    
    def feature_map(self,img_raw):
        img = np.transpose(img_raw,(2,0,1))
        img = np.expand_dims(np.array(img,dtype=np.float32)/255,0)
        with torch.no_grad():
            input = torch.from_numpy(img)
            if self.cuda:
                input = input.cuda()
            out1, out2 = self.get_feature(input)
            # print(out1.shape)
            # print(out2.shape)

            plt.figure(0)
            for i in range(16):
                plt.subplot(4,4,i+1)
                plt.axis('off')
                plt.imshow(out1[0,i])

            plt.figure(1)
            for i in range(96):
                plt.subplot(8,12,i+1)
                plt.axis('off')
                plt.imshow(out2[0,i])

            plt.show()

    def get_feature(self,input,layers = [1,2]):

        net1 = nn.Sequential(*list(self.net.children())[:layers[0]])
        out1 = net1(input)
        
        net2 = nn.Sequential(*list(self.net.children())[layers[0]:layers[1]])
        out2 = net2(out1)

        return out1.cpu().numpy() , out2.cpu().numpy() 

    def __del__(self):
        if self.save:
            self.out.release()

if __name__=="__main__":
    img_path = './line/4/'
    img_num = 188
    save = False
    display_future_map = False
    detector = Line_detector()
    if save:
        train_data = []
        train_name = []
    if not display_future_map:
        for i in range(img_num):
            img_name = img_path + f'{48+i:03d}.jpg'
            print(img_name)
            img = cv2.imread(img_name)
            img = cv2.resize(img, (640, 480))  
            detector.detect(img)
            cv2.waitKey(5)
        cv2.waitKey(0)    
        cv2.destroyAllWindows()
    else:
        img_name = img_path + f'{48:03d}.jpg'
        print(img_name)
        img = cv2.imread(img_name)
        img = cv2.resize(img, (640, 480))  
        detector.feature_map(img)

        