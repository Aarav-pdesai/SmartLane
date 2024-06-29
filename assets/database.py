'''
DATABASE AND TABLE CREATION CODE:

CREATE DATABASE license_plates_db;
USE license_plates_db;

CREATE TABLE IF NOT EXISTS license_plates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    frame_number INT,
    car_id INT,
    license_plate VARCHAR(20),
    timestamp TEXT,
    vehicle_type VARCHAR(50)
);

'''