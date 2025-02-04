# Traffic Light Detection using YOLOv8

This project demonstrates real-time traffic light detection using the YOLOv8 model. The system detects traffic lights in a video stream from a camera and determines their color (red, green, or yellow) using color thresholding in the HSV color space.

# Features

- **Real-time Detection**: Detects traffic lights in real-time using a webcam.
- **Color Detection**: Determines the color of the detected traffic lights (red, green, or yellow).
- **Bounding Boxes**: Draws bounding boxes around detected traffic lights and displays their color and confidence score.

# Usage

1. **Clone the Repository**:

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Script**:

   ```bash
   python traffic_light_detection.py
   ```

4. **Real-time Detection**:
   - The script will open your webcam and start detecting traffic lights.
   - Detected traffic lights will be highlighted with bounding boxes, and their color will be displayed.
   - Press `q` to quit the application.

# Code Overview

- **YOLOv8 Model**: The script uses a pre-trained YOLOv8 model (`yolov8n.pt`) to detect traffic lights in the video stream.
- **Color Detection**: The color of the detected traffic light is determined by analyzing the HSV color space within the bounding box.
- **Real-time Video Processing**: The video stream is processed frame-by-frame, and the results are displayed in real-time.

# Functions

- `detect_traffic_light_color(frame, xmin, ymin, xmax, ymax)`: Detects the color of the traffic light within the specified bounding box.
- `detect_traffic_lights(frame)`: Detects traffic lights in the frame using the YOLOv8 model.
- `draw_boxes(frame, traffic_lights)`: Draws bounding boxes and displays the traffic light color and confidence score.
- `detect_video()`: Captures video from the webcam and performs real-time traffic light detection.
