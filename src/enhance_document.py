import cv2
import numpy as np


def optimized_morphology(image_path):

    img = cv2.imread(image_path, 0)
    if img is None:
        print("Error: load img fail")
        return

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    kernel_size = 100
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(img_clahe, cv2.MORPH_TOPHAT, kernel)

    blurred = cv2.GaussianBlur(tophat, (15, 15), 0)

    enhanced = cv2.addWeighted(img, 0.4, blurred, 0.6, 5)

    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12
    )

    kernel_closure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_closure, iterations=1)

    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_opening, iterations=1)

    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(opened, -1, kernel_sharpen)

    final = cv2.medianBlur(sharpened, 1)

    final = cv2.bitwise_not(final)

    processing_flow = np.vstack(
        [np.hstack([img, img_clahe]), np.hstack([thresh, final])]
    )

    cv2.imshow("Optimized Processing Flow", processing_flow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


optimized_morphology("./img/handwritten.png")
