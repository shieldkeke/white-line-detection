# 基于视觉的白线识别

写得很烂，仅供参考。



主要是基于三块：

1、滤波+canny+hough；

2、滤波+二值化/聚类；

3、端到端神经网络（这个写烂了）

## 相关文件：

文件结构如下：

├─line

│  └─4

├─other_weights

└─utils


可以通过[百度网盘](https://pan.baidu.com/s/1f29Vw--blKuaV_zmUPTFrg?pwd=zjur)下载line文件夹，密码zjur

## 环境配置

```bash
pip3 install -r requirements.txt
```

## 运行结果

```bash
python line_detect_by_filter.py

python line_detect_by_binary.py

python line_detect_by_mobilenet.py
```

## 其他小功能

### 存检测结果为视频

在第一、第二个文件里把save_video改成True，第三个文件里，创建 Line_detector类的时候，设置save为True

### 存检测结果（中心线与第一排交点+角度）

仅限第一个文件，把save设为True

## 训练网络

```bash
python train.py
```

## 联系方式：

见主页
