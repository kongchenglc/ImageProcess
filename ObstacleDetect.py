import cv2
import numpy as np

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to open the camera")
        return

    print("Press 'q' to exit the program")
    
    while True:
        # Read a video frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video frame")
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the HSV range for red color (to adapt to different brightness conditions)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create a mask to capture pixels within the red color range
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Use morphological operations to clean up noise (optional)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of the red regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Ignore regions that are too small
                # Calculate the centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Draw the centroid and the contour
                    cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

                    # Control logic
                    frame_center = frame.shape[1] // 2
                    if cX < frame_center - 50:
                        direction = "Turn Left"
                    elif cX > frame_center + 50:
                        direction = "Turn Right"
                    else:
                        direction = "Move Forward"

                    # Display the direction
                    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    print(direction)

        # Display the original video and the mask
        cv2.imshow("Video Stream", frame)
        cv2.imshow("Red Object Mask", mask)

        # Press 'q' to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
