import cv2 as cv
import numpy as np


class YOLOv4Detector:
    def __init__(self, config_path, weights_path, class_names_path):
        # Load YOLOv4
        self.neural_network = cv.dnn.readNetFromDarknet(config_path, weights_path)
        self.neural_network.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

        # Load class names
        with open(class_names_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]


    def detect_person(self, img):
        blob, height, width = self.to_image_blob(img)
        self.neural_network.setInput(blob)
        output_layer_names = self.neural_network.getUnconnectedOutLayersNames()
        outputs = self.neural_network.forward(output_layer_names)
        final_boxes, coordinates, confidence_scores, ids = self._get_bounding_boxes(outputs, width, height)
        return final_boxes, coordinates, confidence_scores, ids

    def to_image_blob(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        blob = cv.dnn.blobFromImage(img_rgb, 1 / 255.0, size=(608, 608), swapRB=False, crop=False)
        return blob, height, width

    def _get_bounding_boxes(self, detections, width, height, confidence_threshold=0.7, nms_threshold=0.4):
        confidence_scores = []
        ids = []
        coordinates = []

        for detection in detections:
            for prediction in detection:
                scores = prediction[5:]
                class_id = np.argmax(scores)
                if class_id != 0:
                    continue
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    center_x = int(prediction[0] * width)
                    center_y = int(prediction[1] * height)
                    w = int(prediction[2] * width)
                    h = int(prediction[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    coordinates.append([x, y, w, h])
                    ids.append(class_id)
                    confidence_scores.append(float(confidence))

        indices = cv.dnn.NMSBoxes(coordinates, confidence_scores, confidence_threshold, nms_threshold)

        return indices, coordinates, confidence_scores, ids
