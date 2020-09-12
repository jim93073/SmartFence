import signal
from libs.darknet.yolo_device import YoloDevice
from libs.darknet import utils
#from libs.deep_sort.wrapper import DeepSortWrapper
#from libs.deep_sort.wrapper import DEEP_SORT_ENCODER_MODEL_PATH_PERSON

import time,datetime
import LineNotify
import DAN

# Note: this program can only run on Linux
if __name__ == '__main__':
    # Register
    #ServerURL = 'http://IP:9999'      #with non-secure connection
    ServerURL = ''
    Reg_addr = '' #if None, Reg_addr = MAC address

    DAN.profile['dm_name']='Yolo_Device'
    DAN.profile['df_list']=['yPerson-I',]
    DAN.profile['d_name']= ''

    DAN.device_registration_with_retry(ServerURL, Reg_addr)
    #DAN.deregister()  #if you want to deregister this device, uncomment this line
    #exit()            #if you want to deregister this device, uncomment this line

    # Define the variables
    tmp = "19970101000000"
    tmp = datetime.datetime.strptime(tmp, "%Y%m%d%H%M%S")

    yolo1 = YoloDevice(
        video_url="",
        gpu=True,
        gpu_id=0,
        display_message=True,
        config_file=utils.CONFIG_FILE_YOLO_V4,
        names_file=utils.NAMES_COCO,
        thresh=0.45,
        weights_file=utils.WEIGHTS_YOLO_V4_COCO,
        output_dir="./output",
        #use_polygon=True,
        #vertex=[(185,256), (250,365), (585,530), (850,180), (380,130)],
        target_classes=["person"],
        draw_bbox=True,
        draw_polygon=True,
    )

    run = True
                                    
    def on_data(frame_id, img, bboxes, img_path):
        """
        When objects are detected, this function will be called.

        Args:
            frame_id:
                the frame number
            img:
                a numpy array which stored the frame
            bboxes:
                A list contains several `libs.darknet.yolo_device.ExtendedBoundingBox` object. The list contains
                all detected objects in a single frame.
            img_path:
                The path of the stored frame. If `output_dir` is None, this parameter will be None too.
        """
        global tmp
        for det in bboxes:
            now = datetime.datetime.now()
            # You can push these variables to IoTtalk sever
            class_name = det.get_class_name()
            confidence = det.get_confidence()
            center_x, center_y = det.get_center()
            
            if int(now.strftime("%Y%m%d%H%M%S"))>int((tmp+datetime.timedelta(seconds=10)).strftime("%Y%m%d%H%M%S")):
                msg = "\nPeople detected, see: https://"+img_path.split('www')[1]+"\nToday's snapshots: https://"+(img_path.split('www')[1]).rsplit("/", 2)[0]+"/"
                tmp = now
                line_notify(msg)
                DAN.push('yPerson-I', str(det.get_obj_id()), center_x, center_y, img_path.split('www')[1])
                time.sleep(1)
                print(confidence, center_x, center_y)
                print(img_path)

    # `yolo.set_listener()` must be before `yolo.start()`
    yolo1.set_listener(on_data)
    #yolo.enable_tracking(True)
    #yolo.add_deep_sort_tracker("person", DeepSortWrapper(encoder_model_path=DEEP_SORT_ENCODER_MODEL_PATH_PERSON))
    yolo1.start()

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

    print("press ctrl + c to exit.")
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()

    while run:
        pass

    print("Exit.")
