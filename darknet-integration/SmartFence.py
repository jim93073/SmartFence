import signal
from libs.darknet.yolo_device import YoloDevice
from libs.darknet import utils
from libs.deep_sort.wrapper import DeepSortWrapper
from libs.deep_sort.wrapper import DEEP_SORT_ENCODER_MODEL_PATH_PERSON

import time
import datetime
import LineNotify
import DAN
import os
import shutil
import requests
from threading import Thread
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

if __name__ == '__main__':
       
    ServerURL = 'https://edgecore.iottalk.tw'    
    Reg_addr = '555642434' #if None, Reg_addr = MAC address
    DAN.profile['dm_name']='Yolo_Device'
    DAN.profile['df_list']=['yPerson-I',]
    DAN.profile['d_name']= 'YOLOjim'

    DAN.device_registration_with_retry(ServerURL, Reg_addr)
    #DAN.deregister()  #if you want to deregister this device, uncomment this line
    #exit()            #if you want to deregister this device, uncomment this line

    run = True
    output_dir = "/home/jim/SmartFence/darknet-integration/output/"    
    post_img_URL = 'http://panettone.iottalk.tw:12051'

    yolo1 = YoloDevice(        
        video_url="rtsp://user:VIDEO2021@140.113.169.201:554/profile1",
        # video_url="rtsp://admin:edgecore123@120.110.124.23:554/profile1",
        gpu=True,
        gpu_id=0,
        display_message=True,
        config_file=utils.CONFIG_FILE_YOLO_V4,
        names_file=utils.NAMES_COCO,
        thresh=0.45,
        weights_file=utils.WEIGHTS_YOLO_V4_COCO,
        output_dir=output_dir,
        use_polygon=True,
        vertex=[(0,0), (0,1060), (1100,1060), (1100,0) ],
        # vertex=[(0,0), (0,10), (11,10), (11,0) ],
        target_classes=["person"],
        draw_bbox=True,
        draw_polygon=True,
    )

    def del_yesterday_dir():
        yesterday_dir =  output_dir + (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        if os.path.exists(yesterday_dir):
            try:
                shutil.rmtree(yesterday_dir)
                print("[Info] Delete ", yesterday_dir)
            except OSError as e:
                print(e)            

    def post_img(img_path):
        url = post_img_URL
        server_img_path = "https://panettone.iottalk.tw/~jim93073/receive_image/outputs/" + \
        get_current_date_string() + "/" + get_current_hour_string() + "/" + os.path.basename(img_path)
        print("Img path:",img_path)

        try:            
            files = {'file': open(img_path, 'rb')}
            requests.request("POST",url, files=files)
            msg = "[EECS] Time:"+ str(datetime.datetime.now())+", "+ server_img_path
            LineNotify.line_notify(msg)
            print("[Info] POST ",img_path," to panettone successfully")
        except Exception as e:
            print("Error. Failed to post image:",e)        

    def threading_post(q):
        print("Successfully call threading_post")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
        # with ProcessPoolExecutor(max_workers=2) as executor:
            while(True):                
                if(q.empty() == False):
                    executor.submit(post_img, q.get())
                if(q.qsize() > 5):
                    q.get()
                time.sleep(0.1)
            

    def get_current_date_string():
        now_dt = datetime.datetime.now()
        return "{:04d}-{:02d}-{:02d}".format(now_dt.year, now_dt.month, now_dt.day)

    def get_current_hour_string():
        now_dt = datetime.datetime.now()
        return "{:02d}".format(now_dt.hour)

    def on_data(frame_id, img, bboxes, img_path):     
        del_yesterday_dir()
        global q
        for det in bboxes:
            now = datetime.datetime.now()
            class_name = det.get_class_name()
            confidence = det.get_confidence()
            center_x, center_y = det.get_center()
            print(class_name, confidence, center_x, center_y, det.get_class_id(), det.get_obj_id())
            # DAN.push('yPerson-I', str(class_name), center_x, center_y, server_img_path)                                

        if len(bboxes) > 0:
            q.put(img_path)
    
    def signal_handler(sig, frame):
        """
        A simple handler to handle ctrl + c
        """
        global run
        run = False
        # Use `yolo.stop()` to stop the darknet program
        yolo1.stop()
        # Use `yolo.join()` to wait for darknet closing
        yolo1.join()

    
    yolo1.set_listener(on_data)
    # yolo1.enable_tracking(True)
    # yolo1.add_deep_sort_tracker("person", DeepSortWrapper(encoder_model_path=DEEP_SORT_ENCODER_MODEL_PATH_PERSON))
    yolo1.start()

    q = mp.Queue()
    p = mp.Process(target=threading_post, args=((q),))
    p.daemon = True
    p.start()


    print("press ctrl + c to exit.")
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()