import os
import cv2
import numpy as np
from datetime import datetime

from . import utils
from ..deep_sort.wrapper import DeepSortWrapper


class ExtendedBoundingBox:
    def __init__(self, confidence, class_name, class_id, min_x, min_y, max_x, max_y, center_x, center_y, width, height, obj_id=None):
        self.__confidence = confidence
        self.__class_name = class_name
        self.__class_id = class_id
        self.__min_x = min_x
        self.__min_y = min_y
        self.__max_x = max_x
        self.__max_y = max_y
        self.__center_x = center_x
        self.__center_y = center_y
        self.__obj_id = obj_id
        self.__width = width
        self.__height = height

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
        x = self.__center_x
        y = self.__center_y
        return x, y

    def get_obj_id(self):
        return self.__obj_id

    def set_obj_id(self, obj_id):
        self.__obj_id = obj_id

    def get_width(self):
        return self.__width

    def get_height(self):
        return self.__height

    def get_class_id(self):
        return self.__class_id


class YoloDevice:
    """ A wrapper for YOLO Device

        Args:
            config_file: the path of the model config file
            gpu: enable GPU acceleration
            gpu_id: ID of GPU
            display_message: show message from darknet
            names_file: the path of the names file
            weights_file: the path of the weights file
            video_url: the url of video
            thresh: threshold for YOLO
            output_dir: the folder to save output image
        """

    def __init__(self, video_url, gpu=False, gpu_id=0, display_message=False,
                 config_file=utils.CONFIG_FILE_YOLO_V4,
                 names_file=utils.NAMES_COCO,
                 thresh=0.25,
                 weights_file=utils.WEIGHTS_YOLO_V4_COCO,
                 output_dir=None,
                 use_polygon=False,
                 vertex=None,
                 target_classes=None,
                 draw_bbox=False,
                 draw_polygon=False,
                 enable_tracking=False):
        if gpu:
            os.environ["YOLOTALK_USE_GPU"] = "1"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        from . import libyolotalk as ly
        self.ly = ly

        if not use_polygon or vertex is None or (vertex is not None and len(vertex) == 0):
            vertex = None

        utils.check_file_and_fix(weights_file, utils.WEIGHTS_YOLO_V4_COCO, utils.MD5_WEIGHTS_YOLO_V4_COCO, utils.URL_WEIGHTS_YOLO_V4_COCO)

        self.device = self.ly.YoloDevice(
            cfg=config_file,
            weights=weights_file,
            name_list=names_file,
            url=video_url,
            thresh=thresh,
            polygon=vertex,
            output_folder=None,  # File output will handle by python code
            max_video_queue_size=18,
            show_msg=display_message)

        self.output_dir = output_dir
        self.__prediction_listener = None
        self.__listener = None
        self.listener = None
        self.polygon = self.device.getPolygon()

        self.target_classes = target_classes
        self.draw_bbox = draw_bbox
        self.draw_polygon = draw_polygon

        self.__enable_tracking = enable_tracking
        self.__deep_sort_tracker = {}

    def enable_tracking(self, flag):
        self.__enable_tracking = flag

    def add_deep_sort_tracker(self, name, wrapper: DeepSortWrapper):
        self.__deep_sort_tracker[name] = wrapper

    def set_polygon(self, vertex):
        self.device.setPolygon(vertex)
        self.polygon = self.device.getPolygon()

    def start(self):
        self.device.start()

    def stop(self, force=True):
        self.device.stop(force=force)

    def join(self):
        self.device.join()

    def __draw_polygon(self, img):
        if len(self.polygon) == 0:
            return
        pre = None
        for p in self.polygon:
            if pre is None:
                pre = p
                continue
            cv2.line(img, (int(pre.x), int(pre.y)), (int(p.x), int(p.y)), (0, 0, 255), 2)
            pre = p
        cv2.line(img, (int(self.polygon[-1].x), int(self.polygon[-1].y)), (int(self.polygon[0].x), int(self.polygon[0].y)), (0, 0, 255), 2)

    @staticmethod
    def get_current_date_string():
        now_dt = datetime.now()
        return "{:04d}-{:02d}-{:02d}".format(now_dt.year, now_dt.month, now_dt.day)

    @staticmethod
    def get_current_hour_string():
        now_dt = datetime.now()
        return "{:02d}".format(now_dt.hour)

    def __get_output_image_name(self, frame_id, ensure_dir=True):
        folder = os.path.join(
            self.output_dir,
            self.get_current_date_string(),
            self.get_current_hour_string()            
        )
        # print("Folder:",folder)
        if ensure_dir:
          try:
              original_umask = os.umask(0)
               #os.makedirs(folder, exist_ok=True)
              os.makedirs(folder, mode=0o777,exist_ok=True)
          finally:
              os.umask(original_umask)
            
        return os.path.join(folder, "{:05d}.jpg".format(frame_id))

    def __draw_detections(self, image, _detections):
        for detection in _detections:
            detection: ExtendedBoundingBox = detection
            left, top, right, bottom = detection.get_min_x(), detection.get_min_y(), detection.get_max_x(), detection.get_max_y()
            color = self.device.getColors(detection.get_class_id())
            cv2.rectangle(image, (left, top), (right, bottom), color, 1)

            if detection.get_obj_id() is not None:
                label_str = "{} id: {} [{:.2f}]".format(
                    detection.get_class_name(),
                    detection.get_obj_id(),
                    float(detection.get_confidence()))
            else:
                label_str = "{} [{:.2f}]".format(detection.get_class_name(), float(detection.get_confidence()))

            cv2.putText(image, label_str,
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1)

    def __tracking(self, frame, new_bboxes):
        bbox_map = {}
        for key in self.__deep_sort_tracker:
            bbox_map[key] = []

        for bbox in new_bboxes:
            bbox: ExtendedBoundingBox = bbox
            if bbox.get_class_name() in bbox_map:
                bbox_map[bbox.get_class_name()].append(bbox)

        for key in bbox_map:
            value = bbox_map[key]
            _bboxes = []
            for v in value:
                v: ExtendedBoundingBox = v
                x, y = v.get_center()
                w, h = v.get_width(), v.get_height()
                _bboxes.append((x, y, w, h))
            tracker: DeepSortWrapper = self.__deep_sort_tracker[key]
            ids = tracker.get_ids(frame, _bboxes)
            for i, v in enumerate(value):
                v.set_obj_id(ids[i])

    def set_listener(self, listener):
        self.__listener = listener

        def prediction_listener(frame_id, mat, bboxes, file_path):
            # Check if target_classes match
            #print("FRAME ID:"+str(frame_id))
            if self.target_classes is not None:
                bboxes_temp = []
                for b in bboxes:
                    if b.get_name() in self.target_classes:
                        bboxes_temp.append(b)
                bboxes = bboxes_temp

            new_bbox = []
            for b in bboxes:
                box = b.get_box()
                new_bbox.append(ExtendedBoundingBox(
                    confidence=b.get_confidence(),
                    class_name=b.get_name(),
                    class_id=b.get_class_id(),
                    min_x=b.get_x_min(),
                    min_y=b.get_y_min(),
                    max_x=b.get_x_max(),
                    max_y=b.get_y_max(),
                    center_x=box.x,
                    center_y=box.y,
                    width=box.w,
                    height=box.h,
                    obj_id=None
                ))

            if self.__enable_tracking:
                self.__tracking(mat.getData(), new_bbox)

            # Saving image
            img_path = None
            img = mat.getData()
            if self.output_dir is not None and len(new_bbox) != 0:
                if self.draw_polygon:
                    self.__draw_polygon(img)
                if self.draw_bbox:
                    self.__draw_detections(img, new_bbox)
                img_path = self.__get_output_image_name(frame_id)
                cv2.imwrite(img_path, img)
            
            self.__listener(frame_id, img, new_bbox, img_path)
        
        self.__prediction_listener = prediction_listener
        self.device.setPredictionListener(self.__prediction_listener)

