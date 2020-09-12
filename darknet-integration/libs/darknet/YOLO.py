import os
import stat
import hashlib
import subprocess
import threading
import traceback

from . import utils
from .utils import print_info as _print_info
from .utils import print_error as _print_error
from . import __current_dir as _current_dir

YOLOTALK_PREFIX = "[yolotalk]"
YOLOTALK_CMD_PREFIX = "[yolotalk_cmd]"

YOLOTALK_CMD_PREFIX_MODEL_FPS = "[yolotalk_model_fps]"
YOLOTALK_CMD_PREFIX_VIDEO_FPS = "[yolotalk_video_fps]"
YOLOTALK_CMD_PREFIX_QUEUE_SIZE = "[yolotalk_queue_size]"

YOLOTALK_CMD_LOADING_MODEL = "loading_model"
YOLOTALK_CMD_LOADING_MODEL_FINISH = "loading_model_finish"
YOLOTALK_CMD_WARMING_UP = "warming_up"
YOLOTALK_CMD_WARMING_UP_FINISH = "warming_up_finish"
YOLOTALK_CMD_LOADING_VIDEO = "loading_video"
YOLOTALK_CMD_LOADING_VIDEO_FINISH = "loading_video_finish"
YOLOTALK_CMD_START_PREDICT = "start_predict"
YOLOTALK_CMD_PREDICT_NO_OBJECT = "predict_no_object"
YOLOTALK_CMD_VIDEO_CLOSED = "video_closed"
YOLOTALK_CMD_PROGRAM_EXITED = "program_exited"
YOLOTALK_CMD_QUEUE_OVERFLOW = "queue_overflow"

YOLOTALK_CMD_ERROR_OPEN_VIDEO = "error_open_video"
YOLOTALK_CMD_ERROR_READ_VIDEO_FRAME = "error_read_video_frame"


class BoundingBox:
    def __init__(self, confidence, class_name, min_x, min_y, max_x, max_y, image_path):
        self.__confidence = confidence
        self.__class_name = class_name
        self.__min_x = min_x
        self.__min_y = min_y
        self.__max_x = max_x
        self.__max_y = max_y
        self.image_path = image_path

    def get_confidence(self):
        return self.__confidence

    def get_class_name(self):
        return self.__class_name

    def get_min_x(self):
        return self.__min_x

    def get_min_y(self):
        return self.__min_y

    def get_max_x(self):
        return self.__max_x

    def get_max_y(self):
        return self.__max_y

    def get_center(self):
        x = round((self.__min_x + self.__max_x) / 2)
        y = round((self.__min_y + self.__max_y) / 2)
        return x, y

    def get_image_path(self):
        return self.image_path


class YOLO:
    """ A wrapper for YOLO model

    Args:
        config_file: the path of the model config file
        gpu: enable GPU acceleration
        gpu_id: ID of GPU
        display_message: show message from darknet
        names_file: the path of the names file
        weights_file: the path of the weights file
        video_url: the url of video
    """

    def __init__(self, video_url, gpu=False, gpu_id=0, display_message=False,
                 config_file=utils.CONFIG_FILE_YOLO_V4,
                 names_file=utils.NAMES_COCO,
                 thresh=0.25,
                 weights_file=utils.WEIGHTS_YOLO_V4_COCO,
                 output_dir=None,
                 use_polygon=False,
                 vertex=None):

        self.executable = os.path.join(_current_dir, "bin/linux_x64_cpu/main")
        self.gpu = gpu
        self.gpu_id = gpu_id
        if gpu:
            self.executable = os.path.join(_current_dir, "bin/linux_x64_gpu/main")

        st = os.stat(self.executable)
        os.chmod(self.executable, st.st_mode | stat.S_IEXEC)

        self.display_message = display_message
        self.config_file = config_file
        self.weights_file = weights_file
        self.thresh = thresh
        self.names_file = names_file
        self.video_url = video_url
        self.output_dir = output_dir

        self.use_polygon = use_polygon
        self.vertex = vertex

        utils.check_file_and_fix(weights_file, utils.WEIGHTS_YOLO_V4_COCO, utils.MD5_WEIGHTS_YOLO_V4_COCO, utils.URL_WEIGHTS_YOLO_V4_COCO)

        self.ps = None
        self.thread_std_err = None
        self.thread_std_out = None

        self.run = True

        self.model_fps = 0
        self.video_fps = 0
        self.queue_size = 0
        self.max_queue_size = 0
        self.current_frame_id = -1
        self.detection_listener = None
        self.current_detections = []

    def start(self):
        if self.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        args = [
            self.executable,
            "-cfg", self.config_file,
            "-weights", self.weights_file,
            "-names", self.names_file,
            "-thresh", str(self.thresh),
            "-url", self.video_url,
            # "-url", "rtsp://iottalk:iottalk2019@140.113.237.220:554/live2.sdp",
        ]

        if self.output_dir is not None:
            args += [
                "-output_dir", self.output_dir
            ]

        if self.use_polygon and self.vertex is not None:
            vertices = []
            for tup in self.vertex:
                vertices.append("{},{}".format(tup[0], tup[1]))
            vertices_str = ",".join(vertices)

            args += [
                "-use_polygon",
                "-vertex", vertices_str,
            ]

        self.ps = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.thread_std_err = threading.Thread(target=self.__parse_std_err)
        self.thread_std_out = threading.Thread(target=self.__parse_std_out)
        self.thread_std_err.start()
        self.thread_std_out.start()

    def join(self):
        if self.thread_std_err is not None:
            self.thread_std_err.join()
        self.thread_std_err = None
        if self.thread_std_out is not None:
            self.thread_std_out.join()
        self.thread_std_out = None

    def stop(self):
        self.run = False
        if self.ps is not None:
            self.ps.kill()
        self.join()

    def print_info(self, message):
        if self.display_message:
            _print_info(message)

    def print_error(self, message):
        if self.display_message:
            _print_error(message)

    def set_listener(self, on_detection):
        self.detection_listener = on_detection

    def __parse_std_err(self):
        if self.ps is None:
            return
        for line in self.ps.stderr:
            line_str = line.decode("utf-8").replace("\n", "")
            if YOLOTALK_CMD_PREFIX not in line_str:
                continue
            flag = self.__check_and_parse_cmd(line_str)
            if not flag:
                self.__display_yolotalk_cmd(line_str)
            if not self.run:
                break

    def __parse_std_out(self):
        if self.ps is None:
            return
        for line in self.ps.stdout:
            line_str = line.decode("utf-8").replace("\n", "")
            if YOLOTALK_CMD_PREFIX not in line_str:
                continue
            self.__parse_detection(line_str)
            if not self.run:
                break

    def __check_and_parse_cmd(self, line_str):
        temp = line_str.split(" ")
        if len(temp) < 2:
            return False

        prefix, cmd = temp[0], temp[1]
        if prefix != YOLOTALK_CMD_PREFIX:
            return False

        if len(temp) == 4:
            if temp[1] == YOLOTALK_CMD_PREFIX_QUEUE_SIZE:
                # example:
                # [yolotalk_cmd] [yolotalk_queue_size] 0 180
                try:
                    self.queue_size = int(temp[2])
                    self.max_queue_size = int(temp[3])
                    self.print_info("Queue size: {} / {}".format(self.queue_size, self.max_queue_size))
                except ValueError as e:
                    return False
                return True
        elif len(temp) == 3:
            if temp[1] == YOLOTALK_CMD_PREFIX_MODEL_FPS:
                # example:
                # [yolotalk_cmd] [yolotalk_model_fps] 50.84818
                try:
                    self.model_fps = float(temp[2])
                    self.print_info("Model FPS: {} ".format(self.model_fps))
                except ValueError as e:
                    return False
                return True
            elif temp[1] == YOLOTALK_CMD_PREFIX_VIDEO_FPS:
                # example:
                # [yolotalk_cmd] [yolotalk_video_fps] 5
                try:
                    self.video_fps = int(temp[2])
                    self.print_info("Video FPS: {} ".format(self.video_fps))
                except ValueError as e:
                    return False
                return True
        return False

    def __parse_detection(self, line_str):
        # examples:
        # [yolotalk_cmd] predict_no_object
        # [yolotalk_cmd] 46 chair 53.831 216 0 248 29
        temp = line_str.split(" ")

        if len(temp) < 2:
            return False

        prefix, data = temp[0], temp[1:]

        if (self.output_dir is not None and len(data) >= 8) or (self.output_dir is None and len(data) >= 7):
            try:
                if self.output_dir is not None:
                    frame_id = int(data[0])
                    class_name = " ".join(data[1:-6])
                    confidence = float(data[-6])
                    min_x = int(data[-5])
                    min_y = int(data[-4])
                    max_x = int(data[-3])
                    max_y = int(data[-2])
                    image_path = data[-1]
                else:
                    frame_id = int(data[0])
                    class_name = " ".join(data[1:-5])
                    confidence = float(data[-5])
                    min_x = int(data[-4])
                    min_y = int(data[-3])
                    max_x = int(data[-2])
                    max_y = int(data[-1])
                    image_path = None

                data = BoundingBox(confidence, class_name, min_x, min_y, max_x, max_y, image_path)
            except ValueError as e:
                self.print_error("Error. Bad data format. '{}'".format(line_str))
                return False

            if self.current_frame_id == frame_id:
                self.current_detections.append(data)
            else:
                if self.current_frame_id != -1:
                    self.__trigger_callback()
                self.current_frame_id = frame_id
                self.current_detections.append(data)
            return True
        elif len(data) == 1 and data[0] == YOLOTALK_CMD_PREDICT_NO_OBJECT:
            self.__trigger_callback()
            return True
        else:
            self.print_error("Error. Bad data format. '{}'".format(line_str))
        return False

    def __trigger_callback(self):
        if self.detection_listener is None:
            return
        if len(self.current_detections) > 0:
            self.detection_listener(self.current_detections)
            self.current_detections = []

    def __display_yolotalk_cmd(self, line_str):
        temp = line_str.split(" ")
        if len(temp) < 2:
            self.print_error("Error. Bad yolotalk_cmd message.")
            return
        prefix, cmd = temp[0], temp[1]

        if prefix != YOLOTALK_CMD_PREFIX:
            self.print_error("Error. Bad yolotalk_cmd message prefix.")
            return

        if cmd == YOLOTALK_CMD_LOADING_MODEL:
            self.print_info("Loading model.")
        elif cmd == YOLOTALK_CMD_LOADING_MODEL_FINISH:
            self.print_info("Model loaded.")
        elif cmd == YOLOTALK_CMD_WARMING_UP:
            self.print_info("Warming up model.")
        elif cmd == YOLOTALK_CMD_WARMING_UP_FINISH:
            self.print_info("Model Warming up finished.")
        elif cmd == YOLOTALK_CMD_LOADING_VIDEO:
            self.print_info("Loading video.")
        elif cmd == YOLOTALK_CMD_LOADING_VIDEO_FINISH:
            self.print_info("Video loaded.")
        elif cmd == YOLOTALK_CMD_START_PREDICT:
            self.print_info("Prediction thread started.")
        elif cmd == YOLOTALK_CMD_PREDICT_NO_OBJECT:
            self.print_info("No object was detected in this frame.")
        elif cmd == YOLOTALK_CMD_VIDEO_CLOSED:
            self.print_info("Video closed.")
        elif cmd == YOLOTALK_CMD_PROGRAM_EXITED:
            self.print_info("Darknet exited.")
        elif cmd == YOLOTALK_CMD_QUEUE_OVERFLOW:
            self.print_error("Queue overflow.")
        elif cmd == YOLOTALK_CMD_ERROR_OPEN_VIDEO:
            self.print_error("Cannot open video.")
        elif cmd == YOLOTALK_CMD_ERROR_READ_VIDEO_FRAME:
            self.print_error("Cannot read frame from video.")
