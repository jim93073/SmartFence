# The source code of Deep SORT is from: https://github.com/nwojke/deep_sort
# Deep SORT Wrapper by Li-Xian Chen at NCTU
# 2020/08/30
import os
import numpy as np

from .tools import generate_detections as gdet
from .deep_sort import nn_matching
from .application_util import preprocessing
from .deep_sort.tracker import Tracker
from .deep_sort.detection import Detection
from . import __current_dir

DEEP_SORT_ENCODER_MODEL_PATH_PERSON = os.path.join(__current_dir, "./models/market1501.pb")


class DeepSortWrapper:
    # TODO Add docs and test script.
    def __init__(self, max_cosine_distance=0.5, nn_budget=None, nms_max_overlap=0.3,
                 encoder_model_path=DEEP_SORT_ENCODER_MODEL_PATH_PERSON):
        self.encoder = gdet.create_box_encoder(encoder_model_path, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
        self.detections = None

        self.nms_max_overlap = nms_max_overlap

    def get_ids(self, frame, _boxes):
        """ Predict Object ID by Deep SORT

        Args:
            frame:
                A image in numpy array
            _boxes:
                A list contains all bounding box. Each bounding box is in format (x, y, width, height).

        Returns:
            The object ID for each classes.
        """
        features = self.encoder(frame, _boxes)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(_boxes, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        self.detections = detections

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        to_return = []

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                to_return.append(None)
            else:
                to_return.append(track.track_id)

        return to_return

    def get_detections(self):
        return self.detections



