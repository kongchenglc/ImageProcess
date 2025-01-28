import cv2
import numpy as np
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")


# Function to detect the color of the traffic light
def detect_traffic_light_color(frame, xmin, ymin, xmax, ymax):
    # Extract the region of interest (ROI) for the traffic light
    roi = frame[ymin:ymax, xmin:xmax]
    
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define the HSV ranges for red, green, and yellow
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create masks for red, green, and yellow colors
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Calculate the area (number of pixels) for each color
    red_area = np.sum(red_mask)
    green_area = np.sum(green_mask)
    yellow_area = np.sum(yellow_mask)

    # Determine the traffic light status based on the color area
    if red_area > green_area and red_area > yellow_area:
        return "Red"  # Stop command
    elif green_area > red_area and green_area > yellow_area:
        return "Green"  # Accelerate command
    elif yellow_area > red_area and yellow_area > green_area:
        return "Yellow"
    else:
        return "Unknown"


# Function to detect traffic lights in the frame
def detect_traffic_lights(frame):
    # Perform inference using the YOLO model
    results = model.predict(frame, conf=0.5)
    boxes = results[0].boxes
    traffic_lights = []

    # Iterate over the detection results
    for box in boxes:
        cls = int(box.cls)  # Get the class of the detected object
        conf = float(box.conf)  # Get the confidence of the detection
        if cls == 9:  # Class ID 9 corresponds to traffic lights
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
            color = detect_traffic_light_color(frame, xmin, ymin, xmax, ymax)
            traffic_lights.append((xmin, ymin, xmax, ymax, conf, color))

    return traffic_lights


# Function to draw detection boxes and show the traffic light status
def draw_boxes(frame, traffic_lights):
    for (xmin, ymin, xmax, ymax, conf, color) in traffic_lights:
        # Draw the bounding box around the traffic light
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Display the traffic light color and confidence score
        cv2.putText(
            frame,
            f"{color} ({conf:.2f})",
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return frame


# Real-time detection of traffic lights
def detect_video():
    cap = cv2.VideoCapture(0)  # Open the camera
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from camera.")
            break

        # Resize the video frame to improve inference speed
        frame = cv2.resize(frame, (640, 360))

        # Detect traffic lights in the frame
        traffic_lights = detect_traffic_lights(frame)

        # Draw the detection results on the frame
        frame_with_boxes = draw_boxes(frame, traffic_lights)

        # Display the processed frame
        cv2.imshow("YOLOv8 Traffic Light Detection", frame_with_boxes)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    detect_video()  # Start the real-time traffic light detection
