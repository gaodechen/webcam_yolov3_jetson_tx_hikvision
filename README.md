## 简介

* Jetson TX2 ARM上拉流、YOLO v3对象检测
* 视频源为海康摄像头（多个网络摄像头）

实验环境为ARM架构的Jetson TX2，从Jetson TX2上获取海康摄像头的画面（此处为4个），分别套用YOLO V3模型进行对象检测并且输出检测画面。操作过程中较繁琐的步骤是ARM上配环境、稳定的拉流实现。

- YOLO v3 implementation: [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)
- HIKVISION multiple IP cameras: [Yonv1943/Python](https://github.com/Yonv1943/Python/tree/master/Demo_camera_and_network)

![demo](https://github.com/gaodechen/webcam_yolov3_jetson_tx_hikvision/blob/master/demo.png)

## 使用

### 多个网络摄像头拉流 + YOLO v3 运行

首先clone [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)以及YOLO v3预训练模型，直接将本项目的.py文件放进`pytorch-yolo-v3`文件夹当中。

- `yolov3.py`是对`pytorch-yolo-v3`模型推断的二次封装，不需要变动
- `settings.py`当中修改IP camera地址列表

运行：

```c++
python run.py
```

### 多摄像头拉流

如果仅需要对多个海康摄像头拉流，直接跳过推断操作pop_image()展示图片继可。例如，修改前：

```python
processes = [
    mp.Process(target=push_image, args=(raw_q, cam_addr)),
    mp.Process(target=predict, args=(raw_q, pred_q)),
    mp.Process(target=pop_image, args=(pred_q, window_name)),
]
```

修改后：

```python
processes = [
    mp.Process(target=push_image, args=(raw_q, cam_addr)),
    # display images in raw_q instead
    mp.Process(target=pop_image, args=(raw_q, window_name)),
]
```

## 细节

### OpenCV后端选择

系统默认给出了GStreamer进行后端解码。这里我们改为FFMPEG作为后端：

```python
cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
```

### 多个网络摄像头如何拉流

[Yonv1943/Python](https://github.com/Yonv1943/Python/tree/master/Demo_camera_and_network)项目当中已经给出了很棒的解答。

考虑对于一个海康摄像头，由于H.264编码的问题使用多进程实现：

* push_image进程负责拉流，将图片送入raw_queue（未经推断的图片队列）
* predict进程负责将raw_queue中的图片取出，经过模型推断放进pred_queue（处理后的图片队列）
* pop_image进程负责将pred_queue图片弹出显示

多进程间的同步采用共享队列(multiprocessing.Queue)。在此基础上，多进程程序同步问题导致的BUG可能有：

* 推断速度慢，与拉流速度不一致，导致raw_queue当中累积图片多，从而产生了延时，乃至队列溢出
* 网络原因，断流导致程序崩溃

代码当中给出了针对多进程处理速度不一致的解决方案。即push_image进程不断抛弃队首图片。然而实验过程中，程序还会因为断流的原因产生崩溃，故而修改加入重连，得到最终能够长时间稳定运行的push_image()如下：

```python
def push_image(raw_q, cam_addr):
    cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
    while True:
        is_opened, frame = cap.read()
        if is_opened:
            raw_q.put(frame)
        else:
            # reconnect
            cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
        while raw_q.qsize() > 1:
            # drop old images
            raw_q.get()
        else:
            # wait for streaming
            time.sleep(0.01)
```

虽然可以长期运行不至于崩溃，但依旧不够优雅，因为重连依旧耗时。今后有机会继续修改。

### Jetson TX2 ARM PyTorch环境的搭建

git clone --recursive以及compile太耗时了！而且板子上网速也慢，读写也慢。好在最终找到了可以用的whl安装包，直接pip install一次成功。

PyTorch .whl downloading & installation on Jetson TX2: [CSDN Blog](https://blog.csdn.net/beckhans/article/details/91386429)

