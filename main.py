import argparse
import os

from image.draw_bounding_box import draw_bounding_box
from tracker import Tracker
from yolo_detector import YOLOv4Detector
import cv2 as cv


def main():
    args = parse_arguments()
    detector = YOLOv4Detector(args.cfg, args.weights, args.names)
    tracker = Tracker(detector)

    if os.path.exists(args.video):
        cap = cv.VideoCapture(args.video)
    else:
        raise FileNotFoundError(f"Video file '{args.video}' not found.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracker.track(frame)
        draw_bounding_box(frame, tracker.bounding_boxes())
        cv.imshow('People tracking', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Object Tracking and Detection using YOLOv4")
    parser.add_argument('--cfg', type=str, required=True, help='Path to YOLOv4 config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv4 weights file')
    parser.add_argument('--names', type=str, required=True, help='Path to YOLOv4 class names file')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
