import cv2
import os
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time

model = YOLO('yolov8s.pt')

# Create object
cap = cv2.VideoCapture('highway.mp4')

# Defines the list of object classes YOLO will detect
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench']

count = 0
tracker = Tracker()
down = {}
up = {}
counter_down = []
counter_up = []

red_line_y = 198
blue_line_y = 268
offset = 6

# Create a folder to save frames
if not os.path.exists('detected_frames'):
    os.makedirs('detected_frames')

# Initializes a VideoWriter object to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1020, 500))

while True:
    # Reads a single frame from the video capture object
    # ret is a boolean value indicating whether the frame was successfully read
    ret, frame = cap.read()
    if not ret:
        break

    # Use to save each frame image
    count += 1

    # Resize frame
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    boxes = results[0].boxes.data.detach().cpu().numpy()

    # Converts the boxes Numpy array into a Pandas Dataframe
    px = pd.DataFrame(boxes).astype("float")
    
    # Storesbounding boxes of detected objects
    list_of_bboxes = []

    # Iterates each rows
    for index, row in px.iterrows():
        # Extract bounding box coordinates and class id
        x1, y1, x2, y2, _, class_id = row

        # Convert class id to class name refering from class list
        class_name = class_list[int(class_id)]
        if 'car' in class_name or 'truck' in class_name:
            list_of_bboxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    print(list_of_bboxes)
    bbox_id = tracker.update(list_of_bboxes)

    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        cx = int((x3 + x4) / 2)
        cy = int((y3 + y4) / 2)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2) 

        if red_line_y - offset < cy < red_line_y + offset:
            down[obj_id] = time.time()
        if obj_id in down and blue_line_y - offset < cy < blue_line_y + offset:
            elapsed_time = time.time() - down[obj_id]
            if obj_id not in counter_down:
                counter_down.append(obj_id)
                distance = 10  # meters
                speed_ms = distance / elapsed_time
                speed_kmh = speed_ms * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, f'{obj_id}', (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                #cv2.putText(frame, f'{int(speed_kmh)} Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if blue_line_y - offset < cy < blue_line_y + offset:
            up[obj_id] = time.time()
        if obj_id in up and red_line_y - offset < cy < red_line_y + offset:
            elapsed_time = time.time() - up[obj_id]
            if obj_id not in counter_up:
                counter_up.append(obj_id)
                distance = 10  # meters
                speed_ms = distance / elapsed_time
                speed_kmh = speed_ms * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, f'{obj_id}', (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                #cv2.putText(frame, f'{int(speed_kmh)} Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    text_color = (0, 0, 0)  # Black color for text
    yellow_color = (0, 255, 255)  # Yellow color for background
    red_color = (0, 0, 255)  # Red color for lines
    blue_color = (255, 0, 0)  # Blue color for lines

    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)
    cv2.line(frame, (172, red_line_y), (774, red_line_y), red_color, 2)
    cv2.putText(frame, 'Red Line', (172, red_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(frame, (8, blue_line_y), (927, blue_line_y), blue_color, 2)
    cv2.putText(frame, 'Blue Line', (8, blue_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f'Going Down - {len(counter_down)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f'Going Up - {len(counter_up)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Save frame
    frame_filename = f'detected_frames/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)

    out.write(frame)
    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
