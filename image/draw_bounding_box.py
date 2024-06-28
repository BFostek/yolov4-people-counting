import cv2 as cv


def draw_bounding_box(image, box_list):
    cv.putText(image, f"Total de pessoas: {len(box_list)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv.LINE_AA)
    for (x1, y1, w, h) in box_list:
        cv.rectangle(image, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
