# import subprocess
# import threading
#
# ps = subprocess.Popen([
#     "./libs/darknet/bin/linux_x64_cpu/main",
#     "-cfg", "./libs/darknet/weights/yolov4.cfg",
#     "-weights", "./libs/darknet/weights/yolov4.weights",
#     "-names", "./libs/darknet/weights/coco.names",
#     "-url", "rtsp://iottalk:iottalk2019@140.113.237.220:554/live2.sdp",
# ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#
#
# def job1():
#     for li in ps.stderr:
#         print("stderr:", li)
#
#
# def job2():
#     for li in ps.stdout:
#         print("stdout:", li)
#
#
# # 建立一個子執行緒
# t1 = threading.Thread(target=job1)
# t2 = threading.Thread(target=job2)
#
# # 執行該子執行緒
# t1.start()
# t2.start()

import time
from libs.darknet.YOLO import YOLO
import signal
import sys


run = True
# yolo = YOLO("rtsp://iottalk:iottalk2019@140.113.237.220:554/live2.sdp")
yolo = YOLO("person.mp4", thresh=0.25, display_message=True)


def on_data(detections):
    print("-----")
    for det in detections:
        print(det.get_class_name(), det.get_confidence(), det.get_center())
    print("-----")


yolo.set_listener(on_data)
yolo.start()


def signal_handler(sig, frame):
    global run
    run = False
    yolo.stop()
    yolo.join()


signal.signal(signal.SIGINT, signal_handler)
signal.pause()


while run:
    pass