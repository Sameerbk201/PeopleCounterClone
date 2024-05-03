# People Counter using YOLO and SORT

This Python script combines YOLO (You Only Look Once) for object detection and SORT (Simple Online and Realtime Tracking) for object tracking to count people in a video stream. It detects people entering and exiting a specified region in the video frame.

## Dependencies

- Python 3.x
- numpy
- ultralytics
- OpenCV (cv2)
- cvzone
- math
- sort

## Installation

1. Clone this repository:

```
# Note make sure to install 
driver https://www.nvidia.com/download/index.aspx
cuda tool kit https://developer.nvidia.com/cudnn
cuDN https://developer.nvidia.com/cudnn
git clone https://github.com/your_username/your_repository.git
git clone https://github.com/your_username/your_repository.git
```

2. Install the required dependencies:

```
pip install numpy ultralytics opencv-python-headless cvzone
```

## Usage

1. Ensure you have a video file named `people.mp4` in the same directory as the script.
2. Run the script:

```
python people_counter.py
```

3. The script will display the video stream with people counted as they cross predefined lines.
4. Press any key to advance the video frame by frame or uncomment the line for real-time processing (line 118) to run the script continuously.

## Description

The script performs the following tasks:

1. Imports necessary libraries including YOLO, OpenCV, and SORT.
2. Initializes a SORT tracker with specified parameters.
3. Defines the main function which:
   - Loads YOLO model weights.
   - Reads video frames from `people.mp4`.
   - Detects people using YOLO.
   - Filters detected people based on confidence threshold.
   - Tracks filtered people using SORT.
   - Counts people crossing predefined lines in the video frame.
   - Displays the video stream with people counts.

## Notes

- Adjust the paths to the YOLO model weights (`yolov8l.pt`) and mask image (`Mask.png`) as necessary.
- Ensure your system has sufficient resources to run the script, especially for real-time processing.

---

