import os
import numpy as np
from ctypes import *
from .utils import __current_dir


if os.environ.get('YOLOTALK_USE_GPU'):
    so = cdll.LoadLibrary(os.path.join(__current_dir, "bin/linux_x64_gpu/libyolotalk.so.1.0.1"))
else:
    so = cdll.LoadLibrary(os.path.join(__current_dir, "bin/linux_x64_cpu/libyolotalk.so.1.0.1"))


# struct
class box(Structure):
    _fields_ = [('x', c_float),
                ('y', c_float),
                ('w', c_float),
                ('h', c_float), ]


class point(Structure):
    _fields_ = [('x', c_float),
                ('y', c_float), ]


# class BoundingBox
so.BoundingBox_new.argtypes = [c_char_p, c_int, c_float, box]
so.BoundingBox_new.restype = c_void_p

so.BoundingBox_getXmin.argtypes = [c_void_p]
so.BoundingBox_getXmin.restype = c_int

so.BoundingBox_getXmax.argtypes = [c_void_p]
so.BoundingBox_getXmax.restype = c_int

so.BoundingBox_getYmin.argtypes = [c_void_p]
so.BoundingBox_getYmin.restype = c_int

so.BoundingBox_getYmax.argtypes = [c_void_p]
so.BoundingBox_getYmax.restype = c_int

so.BoundingBox_getClassId.argtypes = [c_void_p]
so.BoundingBox_getClassId.restype = c_int

so.BoundingBox_getName.argtypes = [c_void_p]
so.BoundingBox_getName.restype = c_char_p

so.BoundingBox_getConfidence.argtypes = [c_void_p]
so.BoundingBox_getConfidence.restype = c_float

so.BoundingBox_getBox.argtypes = [c_void_p]
so.BoundingBox_getBox.restype = box

# class YoloDevice
so.YoloDevice_new.argtypes = [
    c_char_p, c_char_p, c_char_p, c_char_p, c_float,
    np.ctypeslib.ndpointer(np.single, flags='aligned, contiguous'), c_int,
    c_char_p, c_int, c_bool
]
so.YoloDevice_new.restype = c_void_p

so.YoloDevice_start.argtypes = [c_void_p]
so.YoloDevice_start.restype = None

so.YoloDevice_setPolygon.argtypes = [
    c_void_p,
    np.ctypeslib.ndpointer(np.single, flags='aligned, contiguous'),
    c_int
]
so.YoloDevice_setPolygon.restype = None

so.YoloDevice_setPredictionListener.argtypes = [c_void_p, c_void_p]
so.YoloDevice_setPredictionListener.restype = None

so.YoloDevice_join.argtypes = [c_void_p]
so.YoloDevice_join.restype = None

so.YoloDevice_stop.argtypes = [c_void_p, c_bool]
so.YoloDevice_stop.restype = None

so.YoloDevice_getVideoFps.argtypes = [c_void_p]
so.YoloDevice_getVideoFps.restype = c_float

so.YoloDevice_getModelFps.argtypes = [c_void_p]
so.YoloDevice_getModelFps.restype = c_float

so.YoloDevice_getFps.argtypes = [c_void_p]
so.YoloDevice_getFps.restype = c_float

so.YoloDevice_getColors.argtypes = [c_void_p, c_int, POINTER(c_float), POINTER(c_float), POINTER(c_float)]
so.YoloDevice_getColors.restype = None

so.YoloDevice_getPolygon.argtypes = [c_void_p, POINTER(c_int)]
so.YoloDevice_getPolygon.restype = POINTER(point)

# Helper functions
so.releaseMat.argtypes = [c_void_p]
so.releaseMat.restype = None

so.matToArray.argtypes = [
    c_void_p,
    np.ctypeslib.ndpointer(np.ubyte, flags='aligned, contiguous, writeable'),
    c_int
]
so.matToArray.restype = None

so.getMatLength.argtypes = [c_void_p]
so.getMatLength.restype = c_int

so.getMatInfo.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int), POINTER(c_int)]
so.getMatInfo.restype = None


class BoundingBox(object):
    def __init__(self, obj):
        self.obj = obj

    def get_x_min(self):
        return so.BoundingBox_getXmin(self.obj)

    def get_x_max(self):
        return so.BoundingBox_getXmax(self.obj)

    def get_y_min(self):
        return so.BoundingBox_getYmin(self.obj)

    def get_y_max(self):
        return so.BoundingBox_getYmax(self.obj)

    def get_class_id(self):
        return so.BoundingBox_getClassId(self.obj)

    def get_name(self):
        return so.BoundingBox_getName(self.obj).decode("utf-8")

    def get_confidence(self):
        return so.BoundingBox_getConfidence(self.obj)

    def get_box(self):
        return so.BoundingBox_getBox(self.obj)


class Mat:
    def __init__(self, mat_pointer):
        self.pointer = mat_pointer

        info = [pointer(c_int(0)), pointer(c_int(0)), pointer(c_int(0))]
        so.getMatInfo(self.pointer, info[0], info[1], info[2])
        self.rows, self.cols, self.channels = info[0][0], info[1][0], info[2][0]
        self.length = so.getMatLength(self.pointer)

        result = np.empty(self.length)
        requires = ['CONTIGUOUS', 'ALIGNED', 'WRITEABLE']
        result = np.require(result, np.ubyte, requires)

        so.matToArray(self.pointer, result, self.length)

        self.data = result.reshape((self.rows, self.cols, self.channels))

    def rows(self):
        return self.rows

    def cols(self):
        return self.cols

    def length(self):
        return self.length

    def getData(self):
        return self.data

    def getPointer(self):
        return self.pointer


class YoloDevice:
    def __init__(self, cfg, weights, name_list, url,
                 thresh=0.25,
                 polygon=None,
                 output_folder=None,
                 max_video_queue_size=180,
                 show_msg=False):

        c_vertices = None
        c_vertices_len = 0

        if polygon is None:
            polygon = []

        vertices_list = []
        for p in polygon:
            vertices_list.append(p[0])
            vertices_list.append(p[1])
        c_vertices = np.array(vertices_list)
        requires = ['CONTIGUOUS', 'ALIGNED']
        c_vertices = np.require(c_vertices, np.single, requires)
        c_vertices_len = len(vertices_list)

        if output_folder is not None and type(output_folder) == str:
            output_folder = str.encode(output_folder)
        else:
            output_folder = None

        self.obj = so.YoloDevice_new(
            str.encode(cfg),
            str.encode(weights),
            str.encode(name_list),
            str.encode(url),
            thresh,
            c_vertices,
            c_vertices_len,
            output_folder,
            max_video_queue_size,
            show_msg
        )

        self.__prediction_listener = None

    def start(self):
        so.YoloDevice_start(self.obj)

    def setPolygon(self, polygon):
        c_vertices = None
        c_vertices_len = 0

        if polygon is not None and type(polygon) == list:
            vertices_list = []
            for p in polygon:
                vertices_list.append(p[0])
                vertices_list.append(p[1])
            c_vertices = np.array(vertices_list)
            requires = ['CONTIGUOUS', 'ALIGNED']
            c_vertices = np.require(c_vertices, np.single, requires)
            c_vertices_len = len(vertices_list)

        so.YoloDevice_setPolygon(self.obj, c_vertices, c_vertices_len)

    def setPredictionListener(self, listener):
        """
        example of `listener`
        def listener(frame_id, mat, bboxes, file_path):
            pass
        """

        @CFUNCTYPE(None, c_int, c_void_p, POINTER(c_void_p), c_int, c_char_p)
        def predictionListener(frame_id, mat_ptr, bboxes_ptr, bbox_len, file_path):
            mat = Mat(mat_ptr)
            bboxes = []
            for i in range(bbox_len):
                bboxes.append(BoundingBox(bboxes_ptr[i]))

            if file_path is not None:
                file_path = file_path.encode("utf-8")

            listener(frame_id, mat, bboxes, file_path)

        self.__prediction_listener = predictionListener
        so.YoloDevice_setPredictionListener(self.obj, self.__prediction_listener)

    def join(self):
        so.YoloDevice_join(self.obj)

    def stop(self, force=False):
        so.YoloDevice_stop(self.obj, force)

    def getVideoFps(self):
        return so.YoloDevice_getVideoFps(self.obj)

    def getModelFps(self):
        return so.YoloDevice_getModelFps(self.obj)

    def getFps(self):
        return so.YoloDevice_getFps(self.obj)

    def getColors(self, class_id):
        colors = [pointer(c_float(0)), pointer(c_float(0)), pointer(c_float(0))]
        so.YoloDevice_getColors(self.obj, class_id, colors[0], colors[1], colors[2])
        return [colors[0][0], colors[1][0], colors[2][0]]

    def getPolygon(self):
        p_num = pointer(c_int(0))
        p_points = so.YoloDevice_getPolygon(self.obj, p_num)

        points = []

        for i in range(p_num[0]):
            points.append(p_points[i])

        return points
