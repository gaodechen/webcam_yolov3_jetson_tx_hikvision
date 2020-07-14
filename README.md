## Introduction

* Python多个网络摄像头拉流
* 拉流后使用PyTorch实现的YOLO v3做对象检测

最后需要运行在ARM架构的Jetson TX 2平台上，并且以海康摄像头作为视频源。

- **YOLO v3 implementation**: [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)
- **HIKVISION multiple IP cameras**: [Yonv1943/Python](https://github.com/Yonv1943/Python/tree/master/Demo_camera_and_network)

## 配合其他模型

同样的拉流代码同样可以复用在EfficientDet、YoloV5等模型上，不过模型需要二次封装。

- **EfficientDet implementation**: [gaodechen/EfficientDet-Webcam](https://github.com/gaodechen/EfficientDet-Webcam)
- **YoloV5**: YoloV5工作对性能的对比缺乏参照，但其实现依旧有启发性。官方实现当中使用的DataLoader，故而修改时去掉了拉流取图的进程。

## 测试环境

* Intel CPU + NVIDA GPU
* 视频源：RTSP/RTMP网络摄像头

<div align=center><img width="500px" src="https://github.com/gaodechen/webcam_yolov3_jetson_tx_hikvision/blob/master/demo_1.png" /></div>


## 使用

### 多网络摄像头拉流 + YOLO v3对象检测

首先clone [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)的对象检测实现，下载YOLO v3预训练模型，直接将本项目文件覆盖放入`pytorch-yolo-v3`文件夹当中。

- `yolov3.py`是对`pytorch-yolo-v3`模型推断的二次封装，不需要变动
- `settings.py`当中修改IP camera地址列表，画面大小
- `preprocess.py`添加了prep_frame函数，与prep_image不同只是图片输入从文件改为cv2图像

运行：

```
python run.py --single_window=True (or False)
```

使用`single_window`则所有画面合并显示到同一个窗口当中。

### 仅多摄像头拉流

去掉`predict()`进程的调用，并且`pop_image()`显示`raw_q`中的原图像。

```python
processes = [
    mp.Process(target=push_image, args=(raw_q, cam_addr)),
    # display images in raw_q instead
    mp.Process(target=pop_image, args=(raw_q, window_name)),
]
```

## 其他细节

### OpenCV后端选择

系统默认给出了GStreamer进行后端解码。这里我们改为FFMPEG作为后端：

```python
cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
```

### 多个网络摄像头如何拉流？

[Yonv1943/Python](https://github.com/Yonv1943/Python/tree/master/Demo_camera_and_network)项目当中已经给出了很棒的解答。我们在此基础上进行修改。

考虑对于一个摄像头，由于H.264编码的问题使用**多进程实现**，多进程间的同步采用**共享队列**(multiprocessing.Queue)。

* `push_image()`: 拉流，将图片送入`raw_queue`（未经推断的图片队列）
* `predict()`: 将`raw_queue`中的图片取出，经模型推断放进`pred_queue()`（处理后的图片队列）
* `pop_image()`: 将`pred_queue`中的图片通过OpenCV显示

总结起来，导致程序崩溃的原因可能有：

* **推断速度慢，与拉流速度不一致**: 即进程同步问题。`raw_queue`当中累积图片多，延时高，甚至队列溢出程序崩溃
* **网络原因**: 断流导致程序崩溃
* **机器硬件性能不足**: 导致创建某些进程失败

最初代码中，作者通过`push_image()`进程不断抛弃队首图片解决同步问题。

但是实验过程中，程序还会因为断流的原因产生崩溃，故而这里加入重连，得到最终能够长时间稳定运行的`push_image()`如下：

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
        if raw_q.qsize() > 1:
            # drop old images
            raw_q.get()
        else:
            # wait for streaming
            time.sleep(0.01)
```

经过上述修改可以长时间稳定运行，缺点是断流时重连耗时，画面卡顿1s左右。

此外，在NVIDIA Jetson TX2上运行时，出现了**程序启动后，无法加载几个摄像头画面**的问题，例如分开窗口显示时，有一两个摄像头的窗口一直没有创建。但是程序在本地却正常运行。

最后观察发现，由于**Jetson TX2内存不足**，导致进程无法创建。**解决方法：创建内存交换区**。目测4个海康摄像头进行YOLO对象检测时，6G交换区足够。

## Jetson TX 2 ARM

### 运行环境

* 硬件平台: Jetson TX2 ARM
* 操作系统：Ubuntu 18.04
* 视频源: 多个海康网络摄像头(RTSP TCP)

<div align=center><img width="500px" src="https://github.com/gaodechen/webcam_yolov3_jetson_tx_hikvision/blob/master/demo_2.jpg" /></div>

### Jetson TX2 ARM PyTorch环境的搭建

`git clone --recursive`以及compile太耗时了！而且板子上网速也慢，读写也慢。好在最终找到了可以用的whl安装包，直接`pip install`一次成功。

- **PyTorch .whl downloading & installation on Jetson TX2**: [Nvidia Forum](https://forums.developer.nvidia.com/t/pytorch-for-jetson-nano-version-1-5-0-now-available/72048)
