from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
class My_dataset(Dataset):
    def __init__(self, input_shape, _dataset_path,_label_path):
        # input_shape:image size ;num_output:number of outputs
        super(My_dataset,self).__init__()
        self.dataset_path = _dataset_path
        self.label_path = _label_path
        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channel = input_shape[2]

        self.paths = []
        self.labels = []

        self.len = 0
        self.load_dataset()

    def __len__(self):
        return self.len

    def load_dataset(self):
        with open(self.label_path,'r') as f:
            for line in f:
                line_split = line.split()
                self.labels.append(line_split)
                line_split[0] = float(line_split[0])/600 #数据量纲保持接近
                line_split[1] = float(line_split[1])*4
                self.len+=1

            self.labels = np.array(self.labels,dtype=np.float32)
        
        with open(self.dataset_path,'r') as f:
            for line in f:
                line = line.strip()
                #line = 'F:/pic'+line[1:]#换成全局变量，防止被修改环境变量
                self.paths.append(line)
            self.paths = np.array(self.paths,dtype=np.object)

    def __getitem__(self, index):
        #print("+++",self.paths[index])
        img = cv2.imread(self.paths[index])
        img = cv2.resize(img, (640, 480))
        img = np.transpose(img,(2,0,1))
        img = np.array(img,dtype=np.float32)/255
        label = self.labels[index]
        return img,label

