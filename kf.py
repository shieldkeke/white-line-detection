from cv2 import KalmanFilter as KF
import numpy as np
class kf():
    def __init__(self):
        kalman = KF(4, 2) # 4��״̬����������x��theta��dx��dtheta����2���۲������ܿ�������x,theta
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # ϵͳ��������
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # ״̬ת�ƾ���
        #kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03 # ϵͳ��������Э����
        self.kalman = kalman
    def refresh(self, x, theta):
        measured = np.array([[np.float32(x)], [np.float32(theta)]])
        self.kalman.correct(measured)
    def predict(self):
        predicted = self.kalman.predict()
        x, theta = int(predicted[0]), predicted[1]
        return x, theta
if __name__=='__main__':
    print("please run 'line_detect.py'")