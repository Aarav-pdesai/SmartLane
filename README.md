# SmartLane 
![](https://github.com/Aarav-pdesai/SmartLane/blob/main/assets/demo.gif)

## Overview

SmartLane is a project developed using YOLOv8 and opencv. It provides real-time monitoring of vehicles by detecting vehicles driving in the wrong lanes and storing the license plates of offending vehicles in a MySQL database.

## Features

- **Automated Lane Detection**: Detects and classifies vehicles in specified lanes.
- **License Plate Recognition**: Recognizes license plates using a YOLO-based model.
- **Database Integration**: Stores detected license plate information in a MySQL database.
- **Real-time Processing**: Provides real-time feedback on vehicle movements and infractions.

## Use Cases

- **Traffic Management**: Monitor and manage traffic flow on busy roads.
- **Law Enforcement**: Identify and penalize vehicles violating traffic rules.
- **Toll Collection**: Automate toll collection by recognizing license plates.
- **Parking Management**: Manage and monitor vehicle parking in large facilities.

## Benefits

- **Enhanced Traffic Safety**: Reduces accidents by enforcing lane discipline.
- **Improved Efficiency**: Automates the detection and recognition process, reducing manual effort.
- **Data Collection**: Gathers valuable data for traffic analysis and planning.
- **Scalability**: Can be scaled to monitor multiple lanes and roads simultaneously.

## Requirements

### Dependencies:
- opencv-python
- pandas
- ultralytics
- mysql-connector-python
- numpy
 
## Installation

1. Clone the repository:

```sh
   git clone https://github.com/your-username/smartlane.git
```
2. Run below command to install all requirements for this project:
  ```bash
  pip install -r requirements.txt
  ```

## Dataset
The license plate detection model was trained on the following [dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) from roboflow.

## Database setup
The SQL queries for setting up the database are given in database.py in the assets folder

## Major Improvements for the Future
- **Automated Lane Classification:** Improve the system to automatically classify lanes without predefined lines.
- **Enhanced License Plate Recognition:** Replace traditional image processing techniques (Gaussian blur, thresholding) with a machine  learning model for better accuracy.
- **Performance Improvement:** Optimize the system to handle and display frames faster.
- **Offense Detection:** Extend the system to detect various traffic offenses beyond wrong lane driving.

## Usage
once you have finished with the installation process simply run the main file:
```sh
python main.py
```
## Contribution
We welcome contributions to enhance SmartLane. Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.