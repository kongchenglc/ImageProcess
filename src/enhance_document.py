import cv2
import numpy as np

def add_title(image, title):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        title, font, font_scale, thickness
    )

    background_color = (0, 0, 0)
    cv2.rectangle(
        image,
        (position[0] - 5, position[1] - text_height - 5),
        (position[0] + text_width + 5, position[1] + baseline + 5),
        background_color,
        -1,
    )

    image_with_title = cv2.putText(
        image, title, position, font, font_scale, color, thickness, cv2.LINE_AA
    )
    return image_with_title

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

    img_with_title = add_title(img, "Original Image")
    img_clahe_with_title = add_title(img_clahe, "CLAHE")
    tophat_with_title = add_title(tophat, "Tophat")
    blurred_with_title = add_title(blurred, "Blurred")
    enhanced_with_title = add_title(enhanced, "Enhanced")
    thresh_with_title = add_title(thresh, "Thresholded")
    closed_with_title = add_title(closed, "Closed")
    opened_with_title = add_title(opened, "Opened")
    sharpened_with_title = add_title(sharpened, "Sharpened")
    final_with_title = add_title(final, "Final")

    processing_flow = np.vstack(
        [
            np.hstack(
                [
                    img_with_title,
                    img_clahe_with_title,
                    tophat_with_title,
                    blurred_with_title,
                    enhanced_with_title,
                ]
            ),
            np.hstack(
                [
                    thresh_with_title,
                    closed_with_title,
                    opened_with_title,
                    sharpened_with_title,
                    final_with_title,
                ]
            ),
        ]
    )

    cv2.imshow("Optimized Processing Flow", processing_flow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

optimized_morphology("./img/handwritten.png")
