import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
from util import read_license_plate, preprocess_license_plate
import mysql.connector
from datetime import datetime

model = YOLO('assets/yolov8n.pt')
license_plate_detector = YOLO('assets/best.pt')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
vehicles = [2,3,5,7]
tracker = Tracker()
count = 0
cap = cv2.VideoCapture('highway.mp4')

counter_red = set()
counter_blue = set()

 # Connect to MySQL
conn = mysql.connector.connect(
host="localhost",
user="root",
password="1234",
database="license_plates_db")
cur = conn.cursor()

def insert_license_plate_data(frame_number, car_id, license_plate, time_of_offense, vehicle_class):
       
    sql = '''INSERT INTO license_plates (frame_number, car_id, license_plate, timestamp, vehicle_type) 
             VALUES (%s, %s, %s, %s, %s)'''
    val = (frame_number, car_id, license_plate, time_of_offense, vehicle_class)
    cur.execute(sql, val)
    
    # Commit changes
    conn.commit()
    
dict_of_vehicles = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    list = []
             
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        confidence = float(row[4])
        class_id = int(row[5])
        if class_id in vehicles:
            list.append([x1, y1, x2, y2, class_id])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id, class_id = bbox
        cx = int((x3 + x4) // 2)
        cy = int((y3 + y4) // 2)

        # cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 1)
        # cv2.putText(frame, class_list[class_id], (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # uncomment above lines if you want to see bounding boxes and vehicle labels
        line_y = 350
        red_xleft = 250
        red_xright = 500
        blue_xleft = 515
        blue_xright = 750
        offset = 7
        line_reached = line_y < (cy + offset) and line_y > (cy - offset)  # named condition
        
        if class_id == 5 or class_id == 7:
            if line_reached :
                if red_xleft <= cx <= red_xright:
                    if obj_id not in counter_red:
                        counter_red.add(obj_id)

                    car_image = frame[y3:y4, x3:x4]
                    license_plate_results = license_plate_detector(car_image)[0]

                    for license_plate in license_plate_results.boxes.data.tolist():
                        lx1, ly1, lx2, ly2, lscore, lclass_id = license_plate
                        license_plate_crop = car_image[int(ly1):int(ly2), int(lx1):int(lx2), :]
                        processed_license_plate = preprocess_license_plate(license_plate_crop)
                    
                        # cv2.imshow("Preprocessed License Plate", processed_license_plate)
                        # cv2.waitKey(0)
                    
                        license_plate_text, license_plate_text_score = read_license_plate(processed_license_plate)
                        print(f"OCR Output: {license_plate_text}")


                        if license_plate_text != 0:
                            dict_of_vehicles[obj_id] = [count, obj_id ,license_plate_text, datetime.now().strftime('%Y-%M-%D %H:%M:%S'), class_list[class_id]]
                            # using a dictionary ensures that only one license plate is added to database per obj_id.

        if class_id == 2 or class_id == 3: 
            if line_reached:
                if blue_xleft <= cx <= blue_xright:
                    if obj_id not in counter_blue:
                        counter_blue.add(obj_id)
                        
                    car_image = frame[y3:y4, x3:x4]
                    license_plate_results = license_plate_detector(car_image)[0]

                    for license_plate in license_plate_results.boxes.data.tolist():
                        lx1, ly1, lx2, ly2, lscore, lclass_id = license_plate
                        license_plate_crop = car_image[int(ly1):int(ly2), int(lx1):int(lx2), :]
                    
                        processed_license_plate = preprocess_license_plate(license_plate_crop)
                    
                        # cv2.imshow("Preprocessed License Plate", processed_license_plate)
                        # cv2.waitKey(0)
                    
                        license_plate_text, license_plate_text_score = read_license_plate(processed_license_plate)
                        print(f"OCR Output: {license_plate_text}")


                        if license_plate_text != 0:
                            dict_of_vehicles[obj_id] = [count, obj_id ,license_plate_text, datetime.now().strftime('%Y-%M-%D %H:%M:%S'), class_list[class_id]]
                            # Dictionary ensures that one obj_id can have only one license plate
                            
    text_color = (255, 255, 255)  # white color for text
    red_color = (0, 0, 255)  # (B, G, R)   
    blue_color = (255, 0, 0)  # (B, G, R)
    green_color = (0, 255, 0)  # (B, G, R)  

    cv2.line(frame, (250, 350), (500, 350), red_color, 3)  # starting coordinates and end of line coordinates
    
    cv2.line(frame, (515, 350), (750, 350), blue_color, 3)  # second line   

    red_lane = len(counter_red)
    cv2.putText(frame, 'red lane - ' + str(red_lane), (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1, cv2.LINE_AA)    

    blue_lane = len(counter_blue)
    cv2.putText(frame, 'blue lane - ' + str(blue_lane), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)  

    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
for (frame_number, obj_id, license_plate_text, time ,class_name ) in dict_of_vehicles.values():
    insert_license_plate_data(frame_number, obj_id, license_plate_text, time,class_name)
    
cur.close()
conn.close()
cap.release()
cv2.destroyAllWindows()

# additional tips: 

''' if you want to rename all instance of a variable:
 highlight word, press fn + f2 and rename '''

'''_ is for placeholder'''
