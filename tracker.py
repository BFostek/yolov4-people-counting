import numpy as np
from deep_sort import DeepSort
from yolo_detector import YOLOv4Detector  # Assuming you have YOLOv4Detector implemented


class Tracker:
    def __init__(self, neural_network: YOLOv4Detector):
        self.nn_detector = neural_network

        # Initialize Deep SORT tracker
        deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
        self.tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

    def track(self, image):
        final_boxes, coordinates, confidence_scores, ids = self.nn_detector.detect_person(image)
        bboxes_xywh = []
        con = []
        if len(final_boxes) == 0:
            return
        for item in final_boxes.flatten():
            x1, y1, w, h = coordinates[item]
            cx = x1 + w / 2
            cy = y1 + h / 2
            bboxes_xywh.append([cx, cy, w, h])
            con.append(confidence_scores[item])

        bboxes_xywh = np.array(bboxes_xywh, dtype=float)
        self.tracker.update(bboxes_xywh, con, image)

    def bounding_boxes(self):
        tracks = self.tracker.tracker.tracks
        if len(tracks) == 0:
            return []
        return [
            (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            for track in tracks
            for x1, y1, x2, y2 in [track.to_tlbr()]
        ]

