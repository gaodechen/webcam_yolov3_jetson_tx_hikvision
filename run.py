'''
# webcam_yolov3_jetson_tx_hikvision
# https://github.com/gaodechen/webcam_yolov3_jetson_tx_hikvision
'''

import cv2
import sys
import time
import numpy as np
# import gc
import multiprocessing as mp
from argparse import ArgumentParser

from yolov3 import Model


def push_image(raw_q, cam_addr):
    cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
    while True:
        is_opened, frame = cap.read()
        if is_opened:
            raw_q.put(frame)
        else:
            cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
        if raw_q.qsize() > 1:
            # drop old images
            raw_q.get()
        else:
            # wait for stremaing
            time.sleep(0.01)


def predict(raw_q, pred_q):
    model = Model()
    while True:
        raw_img = raw_q.get()
        pred_img = model.predict(raw_img)
        pred_q.put(pred_img)


def pop_image(pred_q, window_name, img_shape):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = pred_q.get()
        frame = cv2.resize(frame, img_shape)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def display(cam_addrs, window_names, img_shape=(300, 300)):
    raw_queues = [mp.Queue(maxsize=2) for _ in cam_addrs]
    pred_queues = [mp.Queue(maxsize=4) for _ in cam_addrs]
    processes = []

    for raw_q, pred_q, cam_addr, window_name in zip(raw_queues, pred_queues, cam_addrs, window_names):
        processes.append(mp.Process(target=push_image, args=(raw_q, cam_addr)))
        processes.append(mp.Process(target=predict, args=(raw_q, pred_q)))
        processes.append(mp.Process(target=pop_image, args=(pred_q, window_name, img_shape)))

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


def combine_images(queue_list, window_name, img_shape):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    num_cameras = len(queue_list)

    while True:
        imgs = [cv2.resize(q.get(), img_shape) for q in queue_list]
        x = np.concatenate(imgs[:num_cameras//2], axis=1)
        y = np.concatenate(imgs[-(num_cameras-num_cameras//2):], axis=1)
        imgs = np.concatenate([x, y], axis=0)
        cv2.imshow(window_name, imgs)
        # cv2.imwrite('test.jpg', imgs)
        cv2.waitKey(1)


def display_single_window(cam_addrs, window_name='camera',  img_shape=(300, 300)):
    raw_queues = [mp.Queue(maxsize=2) for _ in cam_addrs]
    pred_queues = [mp.Queue(maxsize=4) for _ in cam_addrs]
    processes = []

    processes.append(mp.Process(target=combine_images, args=(pred_queues, window_name, img_shape)))
    for raw_q, pred_q, cam_addr in zip(raw_queues, pred_queues, cam_addrs):
        processes.append(mp.Process(target=push_image, args=(raw_q, cam_addr)))
        processes.append(mp.Process(target=predict, args=(raw_q, pred_q)))

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    mp.set_start_method(method='spawn')

    parser = ArgumentParser()
    parser.add_argument('--num_cameras', '-n', type=int,
                        help='number of cameras to process')
    parser.add_argument('--single_window', '-s', type=str, default='True',
                        help='should multiple cameras displayed in one single window,\
                            only used when number of cameras > 1')
    args = parser.parse_args()

    # load configurations
    from settings_ import cam_addrs, img_shape
    args.single_window = True if args.single_window.lower() == 'true' else False
    args.num_cameras = len(cam_addrs) if args.num_cameras is None else args.num_cameras

    print(args.num_cameras)

    if args.single_window is False or args.num_cameras is 1:
        # display each of the cameras in separate window
        display(cam_addrs[:args.num_cameras], ['camera' for _ in cam_addrs], img_shape)
    else:
        # combine all images and display in one single window
        display_single_window(cam_addrs[:args.num_cameras], 'camera', img_shape)