import time
import multiprocessing as mp
import cv2
import gc
import numpy as np
import sys

from yolov3 import Model


def push_image(raw_q, cam_addr):
    cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
    while True:
        is_opened, frame = cap.read()
        if is_opened:
            raw_q.put(frame)
        else:
            cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
        while raw_q.qsize() > 1:
            # drop old images
            raw_q.get()
        else:
            # wait for stremaing
            time.sleep(0.01)


def predict(raw_q, pred_q):
    model = Model()
    is_opened = True
    while is_opened:
        raw_img = raw_q.get()
        pred_img = model.predict(raw_img)
        pred_q.put((pred_img))


def pop_image(pred_q, window_name, img_shape=(416, 416)):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = pred_q.get()
        frame = cv2.resize(frame, img_shape)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def run_single_camera(cam_addr, window_name, img_shape=(450, 450)):
    raw_q = mp.Queue(maxsize=2)
    pred_q = mp.Queue(maxsize=4)

    processes = [
        mp.Process(target=push_image, args=(raw_q, cam_addr)),
        mp.Process(target=predict, args=(raw_q, pred_q)),
        mp.Process(target=pop_image, args=(pred_q, window_name)),
    ]

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


def run_multi_camera(cam_addrs, window_names, img_shape=(300, 300)):
    raw_queues = [mp.Queue(maxsize=2) for _ in cam_addrs]
    pred_queues = [mp.Queue(maxsize=4) for _ in cam_addrs]

    processes = []
    for raw_q, pred_q, cam_addr, window_name in zip(raw_queues, pred_queues, cam_addrs, window_names):
        processes.append(mp.Process(target=push_image, args=(raw_q, cam_addr)))
        processes.append(mp.Process(target=predict, args=(raw_q, pred_q)))
        processes.append(mp.Process(target=pop_image,
                                    args=(pred_q, window_name, img_shape)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


def combine_images(queue_list, window_name, img_shape=(300, 300)):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        imgs = [cv2.resize(q.get(), img_shape) for q in queue_list]
        x = np.concatenate(imgs[:2], axis=1)
        y = np.concatenate(imgs[-2:], axis=1)
        imgs = np.concatenate([x, y], axis=0)
        cv2.imshow(window_name, imgs)
        cv2.imwrite('test.jpg', imgs)
        cv2.waitKey(1)


def run_multi_camera_in_a_window(cam_addrs, img_shape=(300, 300)):
    raw_queues = [mp.Queue(maxsize=2) for _ in cam_addrs]
    pred_queues = [mp.Queue(maxsize=4) for _ in cam_addrs]

    processes = [mp.Process(target=combine_images, args=(pred_queues, 'CAMs'))]
    for raw_q, pred_q, cam_addr in zip(raw_queues, pred_queues, cam_addrs):
        processes.append(mp.Process(target=push_image, args=(raw_q, cam_addr)))
        processes.append(mp.Process(target=predict, args=(raw_q, pred_q)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


def run():
    from settings import cam_addrs, img_shape
    # run_single_camera(cam_addrs[0], 'Test')
    # run_multi_camera(cam_addrs, ['Test' for _ in cam_addrs], img_shape)
    run_multi_camera_in_a_window(cam_addrs, img_shape)


if __name__ == '__main__':
    mp.set_start_method(method='spawn')
    run()
