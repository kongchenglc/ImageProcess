import cv2
import numpy as np


# Function to add a title on the image
def add_title(image, title):
    # Font settings for the title text
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)  # Position of the title
    font_scale = 1
    color = (255, 255, 255)  # White color for the title text
    thickness = 2

    # Calculate the width and height of the text to draw a rectangle for the background
    (text_width, text_height), baseline = cv2.getTextSize(
        title, font, font_scale, thickness
    )

    # Background color for the title rectangle (black)
    background_color = (0, 0, 0)

    # Draw a rectangle behind the title text for better visibility
    cv2.rectangle(
        image,
        (position[0] - 5, position[1] - text_height - 5),
        (position[0] + text_width + 5, position[1] + baseline + 5),
        background_color,
        -1,  # -1 means filled rectangle
    )

    # Place the title text over the background rectangle
    image_with_title = cv2.putText(
        image, title, position, font, font_scale, color, thickness, cv2.LINE_AA
    )
    return image_with_title


# Main function to perform the optimized morphological image processing
def optimized_morphology(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, 0)
    if img is None:
        print("Error: load img fail")  # Error message if image cannot be loaded
        return

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # Define the kernel size for morphological operations (100x100)
    kernel_size = 100
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Perform morphological Top-hat transformation to highlight bright areas
    tophat = cv2.morphologyEx(img_clahe, cv2.MORPH_TOPHAT, kernel)

    # Apply Gaussian blur to the Top-hat image to remove noise
    blurred = cv2.GaussianBlur(tophat, (15, 15), 0)

    # Combine the original image with the blurred version using weighted addition
    enhanced = cv2.addWeighted(img, 0.4, blurred, 0.6, 5)

    # Apply adaptive thresholding to create a binary image (inverted)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12
    )

    # Perform morphological closing to remove small dark spots in the binary image
    kernel_closure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_closure, iterations=1)

    # Perform morphological opening to remove small white regions and noise
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_opening, iterations=1)

    # Apply sharpening filter to enhance the details of the text
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(opened, -1, kernel_sharpen)

    # Apply median blur to reduce noise and make the image smoother
    final = cv2.medianBlur(sharpened, 1)

    # Invert the final image (so the text is black on a white background)
    final = cv2.bitwise_not(final)

    # Add titles to different stages of the image processing for visualization
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

    # Stack all images in a grid format for side-by-side comparison
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

    # Display the entire processing flow in a window
    cv2.imshow("Optimized Processing Flow", processing_flow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call the function with the path to an image file
optimized_morphology("./img/handwritten.png")
