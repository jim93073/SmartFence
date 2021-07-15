import os
import urllib.request
import shutil
import hashlib
import traceback
from . import __current_dir

CONFIG_FILE_YOLO_V4 = os.path.join(__current_dir, 'weights/yolov4.cfg')
DATA_FILE_COCO = os.path.join(__current_dir, 'weights/coco.data')
NAMES_COCO = os.path.join(__current_dir, 'weights/coco.names')
WEIGHTS_YOLO_V4_COCO = os.path.join(__current_dir, 'weights/yolov4.weights')
MD5_WEIGHTS_YOLO_V4_COCO = "00a4878d05f4d832ab38861b32061283"
URL_WEIGHTS_YOLO_V4_COCO = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"


def print_info(message):
    print("[YoloTalk]", "[INFO ]", message)
    pass


def print_error(message):
    print("[YoloTalk]", "[ERROR]", message)
    pass


def download(url, file_path):
    def show_progress(count, block_size, total_size):
        width = shutil.get_terminal_size((80, 20)).columns
        max_len = width - 5
        percent = count * block_size / total_size
        percent_str = "{:.02f}%".format(percent*100)
        line = percent_str + " ["
        line += "=" * int((max_len-len(line)-len(percent_str)) * percent)
        closure = "]"
        line += (" " * int(max_len-len(line)-len(closure))) + closure
        print(line, end="\r")

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent',
                          'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)

    print("Start download file '{}' from '{}'".format(file_path, url))
    urllib.request.urlretrieve(url, file_path, show_progress)
    print("\nDone.")


def check_file_and_fix(to_test, target, target_checksum, remote_url):
    if to_test == target:
        file_ok = True
        if not os.path.exists(to_test):
            file_ok = False
        '''
        # check model hash
        else:
            check_sum = hashlib.md5(open(to_test, 'rb').read()).hexdigest()
            if check_sum != target_checksum:
                file_ok = False
                print("Weights file checksum error.")
        '''

        if not file_ok:
            print("{} not exist. Try to download.".format(target))
            try:
                download(remote_url, target)
            except Exception as e:
                print("Error. Faild to download file.")
                print(e)
                traceback.print_exc()
                exit(1)
