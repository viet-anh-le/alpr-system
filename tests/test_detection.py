import cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

ocr = PaddleOCR(use_textline_orientation=False, lang="en")


def analyze_plate_with_dbnet(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không tìm thấy ảnh tại: {image_path}")
        return

    img_draw = img.copy()

    results = ocr.ocr(image_path, det=True, rec=False)

    boxes = results[0] if results and results[0] else []

    print("-" * 40)
    print(f"File: {image_path.split('/')[-1]}")
    print(f"DBNet tìm thấy: {len(boxes)} dòng chữ")

    if len(boxes) > 0:
        boxes = sorted(boxes, key=lambda box: np.mean([pt[1] for pt in box]))
        for i, box in enumerate(boxes):
            box_points = np.array(box).astype(np.int32)

            cv2.polylines(img_draw, [box_points], isClosed=True, color=(0, 255, 0), thickness=2)

            top_left_x, top_left_y = box_points[0]
            cv2.putText(
                img_draw,
                f"Line {i+1}",
                (top_left_x, top_left_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            x, y, w, h = cv2.boundingRect(box_points)
            cropped_line = img[y : y + h, x : x + w]

            print(
                f"  + Dòng {i+1}: Tọa độ Y={np.mean([pt[1] for pt in box]):.1f} | Kích thước Crop: {w}x{h}"
            )

    cv2.imshow("DBNet Detection", img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_plate_with_dbnet(
        "/home/vietanh/Documents/DATN/datasets/OCR/images/train/1PlateBaza487.jpg"
    )

    analyze_plate_with_dbnet("/home/vietanh/Documents/DATN/datasets/OCR/images/train/1xemay278.jpg")
