from PyQt5 import QtCore, QtGui, QtWidgets
from instruct_window import Ui_Instructions
from main_window import Ui_MainWindow
from status_window import Ui_Status
from about_window import Ui_AboutWin
from error import Ui_ERROR

# importing the multiprocessing module
import multiprocessing

import cv2
import time
import HandDetectorModule as hdm


from LightMode import lightMode as lm
#import Gesture as ges

###################################################################################################
#Came Inside of Count_Test Function
import depthai as dai
import blobconverter
import Depth as dep
import dlib
import numpy as np
#Hand, Mouth, and Depth

###################################################################################################
### Imports for Servo ###


import Adafruit_PCA9685
import time
import tkinter as tk
import threading


###################################################################################################


def Count_Test():
    
    detector = hdm.handDetector(detectionCon=0.8, maxHands=1)
    #hand_gesture = ges.Gesture()

    
    depth = dep.detectDepth()
    
    #Define Frame size
    FRAME_SIZE = (640, 360)

    #Define the NN model name and input size
    DET_INPUT_SIZE = (300, 300)
    model_name = "face-detection-retail-0004"
    zoo_type = "depthai"
    blob_path = None

    pipeline = dai.Pipeline()
    
    #cam,mono_left,mono_right,stereo=depth.source_cam(pipeline,FRAME_SIZE)
    #face_spac_det_nn,face_det_manip=depth.face_detect(pipeline,DET_INPUT_SIZE,model_name,zoo_type,blob_path)
    #Define a source – RGB camera
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(FRAME_SIZE[0], FRAME_SIZE[1])
    cam.setInterleaved(False)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setFps(35)
    
    #Define mono camera sources for stereo depth
    mono_left = pipeline.createMonoCamera()
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right = pipeline.createMonoCamera()
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    #Create stereo node
    stereo = pipeline.createStereoDepth()
    
    #Linking mono cam outputs to stereo node
    
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    
    #Use blobconverter to get the blob of the required model
    if model_name is not None:
        blob_path = blobconverter.from_zoo(
        name=model_name,
        #The ‘shaves’ argument in blobconverter determines the number of SHAVE cores used to compile the neural network. The higher the value, the faster network can run.
        shaves=6,
        zoo_type=zoo_type
        ) 
    
    #Define face detection NN node   
    face_spac_det_nn = pipeline.createMobileNetSpatialDetectionNetwork()
    face_spac_det_nn.setConfidenceThreshold(0.75)
    face_spac_det_nn.setBlobPath(blob_path)
    face_spac_det_nn.setDepthLowerThreshold(100)
    face_spac_det_nn.setDepthUpperThreshold(5000)
    
    #Define face detection input config
    face_det_manip = pipeline.createImageManip()
    face_det_manip.initialConfig.setResize(DET_INPUT_SIZE[0], DET_INPUT_SIZE[1])
    face_det_manip.initialConfig.setKeepAspectRatio(False)
    
    #Linking
    cam.preview.link(face_det_manip.inputImage)
    face_det_manip.out.link(face_spac_det_nn.input)
    stereo.depth.link(face_spac_det_nn.inputDepth)

    #Preview Output
    x_preview_out = pipeline.createXLinkOut()
    x_preview_out.setStreamName("preview")
    cam.preview.link(x_preview_out.input)

    #Detection Output
    det_out = pipeline.createXLinkOut()
    det_out.setStreamName('det_out')
    face_spac_det_nn.out.link(det_out.input)
    
    
    # Frame count
    frame_count = 0
     
    # Placeholder fps value
    fps = 0
     
    # Used to record the time when we processed last frames
    prev_frame_time = 0
     
    # Used to record the time at which we processed current frames
    new_frame_time = 0
     
    # Set status colors
    status_color = {
        'Face Detected': (0, 255, 0),
        'No Face Detected': (0, 0, 255)}
        
    # Load face detector
    #face_detector = dlib.get_frontal_face_detector()
    # Load shape predictor for 68 landmarks
    shape_predictor = dlib.shape_predictor('68_points/shape_predictor_68_face_landmarks.dat')

    # Camera intrinsic parameters
    focal_length = 250.0  
    principal_point = (0, 250)  

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        #print('Connected cameras:', device.getConnectedCameraFeatures())
        # Print out usb speed
        #print('Usb speed:', device.getUsbSpeed().name)
        # Bootloader version
        if device.getBootloaderVersion() is not None:
            print('Bootloader version:', device.getBootloaderVersion())
        # Device name
        print('Device name:', device.getDeviceName())

        
         # Output queue will be used to get the right camera frames from the outputs defined above
        q_cam = device.getOutputQueue(name="preview", maxSize=1, blocking=False)
     
        # Output queue will be used to get nn data from the video frames.
        q_det = device.getOutputQueue(name="det_out", maxSize=1, blocking=False)

        while True:
             # Get right camera frame
            in_cam = q_cam.get()
            frame = in_cam.getCvFrame()
            #inRgb = q_cam.get()  # blocking call, will wait until a new data has arrived
             
             # Retrieve 'bgr' (opencv format) frame
            img = frame
           
            hands,img = detector.findHands(img, flipType=True)
            
            lmList = detector.findPosition(img,draw=False)
            # If a hand is detected
            if len(lmList)!= 0:
                hand = hands[0]
                # List of which fingers are up
                fingers,count = detector.fingersUp(hand) 
                # Show total count of fingers
                cv2.putText(img,f'Count: {int(count)}', (450, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                #####################################################################
                print(count)

                ##################################################################################################33
                ##################################################################################################33
                ##################################################################################################33
                if count == 1:
                    loopBase()
                elif count == 5:
                    start_arm_postion()
                #print(totalFingers)
                #hand_gesture.Fingers(count)

            
            bbox = None
            coordinates = None
     
            inDet = q_det.tryGet()
            
     
            if inDet is not None:
                
                detections = inDet.detections
               
     
                # if face detected
                if len(detections) !=0:
                    detection = detections[0]
     
                    # Correct bounding box
                    xmin = max(0, detection.xmin)
                    ymin = max(0, detection.ymin)
                    xmax = min(detection.xmax, 1)
                    ymax = min(detection.ymax, 1)
     
                    # Calculate coordinates
                    x = int(xmin*FRAME_SIZE[0])
                    y = int(ymin*FRAME_SIZE[1])
                    w = int(xmax*FRAME_SIZE[0]-xmin*FRAME_SIZE[0])
                    h = int(ymax*FRAME_SIZE[1]-ymin*FRAME_SIZE[1])
                    
                    x2= x+w
                    y2= y+h
                   
                    bbox = (x, y, w, h)
    
                    # Get spacial coordinates
                    coord_x = detection.spatialCoordinates.x
                    coord_y = detection.spatialCoordinates.y
                    coord_z = detection.spatialCoordinates.z
     
                    coordinates = (coord_x, coord_y, coord_z)
                    #cv2.putText(img,f'coordinates: {coordinates}', (10, 240), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0))
                    
                    #coordinates to be used for shape_predictor()
                    face=dlib.rectangle(int(x),int(y),int(x2),int(y2))
                    #cv2.putText(img,f'detections: {face}', (10, 230), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 0, 0))
     
            # Check if a face was detected in the frame
            if bbox:
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Face detected
                status = 'Face Detected'
                # Get landmarks for face
                landmarks = shape_predictor(gray, face)
                
                # Extract coordinates for mouth
                mouth_points = landmarks.parts()[48:68]
                mouth_points = [[p.x, p.y] for p in mouth_points]
                
                # Calculate average distance of mouth from camera
                avg_depth = np.mean(focal_length * np.ones(len(mouth_points), dtype=np.float32))
                #print("Average depth of mouth from camera:", avg_depth)
                #cv2.putText(img,f'depth: {avg_depth}', (10, 290), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255))
                
                # Draw mouth on frame
                for i, point in enumerate(mouth_points[:-1]):
                    cv2.line(img, tuple(point), tuple(mouth_points[i + 1]), (255, 0, 0), 2)
                
                # Print only the specified mouth points in terminal
                cv2.putText(img,f'1st Quadrant:, {mouth_points[2]}, 2nd Quadrant: {mouth_points[4]}',(20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                cv2.putText(img,f'3rd Quadrant: {mouth_points[8]} 4th Quadrant: {mouth_points[10]}',(20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                
                
            else:
                # No face detected
                status = 'No Face Detected'
     
            # Display info on frame
            depth.display_info(img, bbox, coordinates, status, status_color, fps)
     
            # Calculate average fps
            if frame_count % 10 == 0:
                # Time when we finish processing last 100 frames
                new_frame_time = time.time()
     
                # Fps will be number of frame processed in one second
                fps = 1 / ((new_frame_time - prev_frame_time)/10)
                prev_frame_time = new_frame_time
     
            # Capture the key pressed
            key_pressed = cv2.waitKey(1) & 0xff
     
            # Stop the program if Esc key was pressed
            if key_pressed == 'q':
                break
     
            # Display the final frame
            cv2.imshow("Face Cam", frame)
            #cv2.imshow("Image", img)
     
            # Increment frame count
            frame_count += 1       
    



###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################


    # Initialize the PCA9685 PWM device
    pwm = Adafruit_PCA9685.PCA9685()

    # Frequency of the PWM signal
    pwm.set_pwm_freq(60)

    # Set up Servo motor range
    servo_min = 150  # Min pulse length out of 4096
    servo_max = 600  # Max pulse length out of 4096
    servo_range = servo_max - servo_min

    # Set up the servo channels
    servo_channels = [0, 1, 2, 3, 4, 5, 6, 7]

    # Speed of the servo motors
    servo_speed = 1

    # Dictionary to store the servo angle values
    servo_angles = {channel: 90 for channel in servo_channels}



    # Dictionary to store the label text
    servo_labels = {}

    # Function to control servo motor
    def set_servo_angle(channel, angle):
        duty_cycle = int(((angle / 180) * (servo_max - servo_min)) + servo_min)
        pwm.set_pwm(channel, 0, duty_cycle)
        print("Servo angle:", angle, "Servo channel:", channel)
        servo_angles[channel] = angle
        update_servo_angle(channel)

    #Starting of servo motor position
    def start_arm_postion():
        set_servo_angle(0, 110)
        set_servo_angle(1, 70)
        set_servo_angle(2, 90)
        set_servo_angle(3, 140)
        set_servo_angle(4, 74)
        set_servo_angle(5, 105)
        set_servo_angle(6, 30)

    # Function to increase the servo "SHOULDER" (channel 0 & 1) angle by 1 degree 
    """"
    'W' Key is used to increase the angle of the servo motor
    """
    def increase_servo0and1_angle():
        newAngle0 = servo_angles[0] + servo_speed
        print(newAngle0)
        if newAngle0 <= 90:
            set_servo_angle(0, newAngle0)
            servo_angles[0] = newAngle0
        newAngle1 = servo_angles[1] - servo_speed
        if newAngle1 >= 70:
            set_servo_angle(1, newAngle1)
            servo_angles[1] = newAngle1
                    
    # Function to decrease the servo "SHOULDER" (channel 0 & 1) angle by 1 degree
    """
    'S' Key is used to decrease the angle of the servo motor
    """
    def decrease_servo0and1_angle():
        newAngle1 = servo_angles[1] + servo_speed
        if newAngle1 <= 90:
            set_servo_angle(1, newAngle1)
            servo_angles[1] = newAngle1
        newAngle0 = servo_angles[0] - servo_speed
        if newAngle0 >=70:
            set_servo_angle(0, newAngle0)
            servo_angles[0] = newAngle0
            
    # Function to increase the servo "BASE" (channel 2) base by 1 degree
    """
    'A' Key is used to increase the angle of the servo motor
    """
    def increase_servo2_base_angle():
        new_angle_base = servo_angles[2] + servo_speed
        if new_angle_base <= 180:
            set_servo_angle(2, new_angle_base)
            servo_angles[2] = new_angle_base
            

    # Function to decrease the servo "BASE" (channel 2) base by 1 degree
    """
    'D' Key is used to decrease the angle of the servo motor
    """
    def decrease_servo2_base_angle():
        new_angle_base = servo_angles[2] - servo_speed
        if new_angle_base >= 0:
            set_servo_angle(2, new_angle_base)
            servo_angles[2] = new_angle_base
            

    # Function to increase the servo "ELBOW" (channel 3) angle by 1 degree
    """
    'Q' Key is used to increase the angle of the servo motor
    """
    def increase_servo3_angle():
        new_angle = servo_angles[3] + servo_speed
        if new_angle <= 140:
            set_servo_angle(3, new_angle)
            servo_angles[3] = new_angle

    # Function to decrease the servo "ELBOW" (channel 3) angle by 1 degree
    """
    'E' Key is used to decrease the angle of the servo motor
    """
    def decrease_servo3_angle():
        new_angle = servo_angles[3] - servo_speed
        if new_angle >= 70:
            set_servo_angle(3, new_angle)
            servo_angles[3] = new_angle

    # Function to incerase the servo "WRIST"(channel 4) angle by 1 degree
    """
    'R' Key is used to increase the angle of the servo motor 
    """
    def increase_servo4_angle():
        new_angle = servo_angles[4] + servo_speed
        if new_angle <= 135:
            set_servo_angle(4, new_angle)
            servo_angles[4] = new_angle

    # Function to decrease the servo "WRIST" (channel 4) angle by 1 degree
    """
    'F' Key is used to decrease the angle of the servo motor
    """
    def decrease_servo4_angle():
        new_angle = servo_angles[4] - servo_speed
        if new_angle >= 45:
            set_servo_angle(4, new_angle)
            servo_angles[4] = new_angle

    # Function to incerase the servo "5TH AXIS" (channel 5) angle by 1 degree
    """
    'T' Key is used to increase the angle of the servo motor
    """
    def increase_servo5_angle():
        new_angle = servo_angles[5] + servo_speed
        if new_angle <= 180:
            set_servo_angle(5, new_angle)
            servo_angles[5] = new_angle

    # Function to decrease the servo "5TH AXIS" (channel 5) angle by 1 degree
    """
    'G' Key is used to decrease the angle of the servo motor
    """
    def decrease_servo5_angle():
        new_angle = servo_angles[5] - servo_speed
        if new_angle >= 0:
            set_servo_angle(5, new_angle)
            servo_angles[5] = new_angle

    # Function to incerase the servo "6TH AXIS" (channel 6) angle by 1 degree
    """
    'Y' Key is used to increase the angle of the servo motor
    """
    def increase_servo6_angle():
        new_angle = servo_angles[6] + servo_speed
        if new_angle <= 70:
            set_servo_angle(6, new_angle)
            servo_angles[6] = new_angle
            

    # Function to decrease the servo 6TH AXIS (channel 6) angle by 1 degree
    """
    'H' Key is used to decrease the angle of the servo motor
    """
    def decrease_servo6_angle():
        new_angle = servo_angles[6] - servo_speed
        if new_angle >= 30:
            set_servo_angle(6, new_angle)
            servo_angles[6] = new_angle
            

    # Fuction First Movement
    def first_movement():
        set_servo_angle(0, 0) # Servo shoulder (0) 
        set_servo_angle(1, 180) # Servo shoulder (1)
        set_servo_angle(2, 90) #Servo base (2)
        set_servo_angle(3, 90) # Servo elbow (3)
        set_servo_angle(4, 90) # Servo wrist (4)
        set_servo_angle(5, 90) #Servo 5th Axis
        set_servo_angle(6, 90) #Servo 6th Axis

    # Fuction Second Movement
    def second_movement():
        set_servo_angle(0, 90) # Servo shoulder (0)
        set_servo_angle(1, 90) # Servo shoulder (1)
        set_servo_angle(2, 90) #Servo base (2)
        set_servo_angle(3, 90) # Servo elbow (3)
        set_servo_angle(4, 90) # Servo wrist (4)
        set_servo_angle(5, 90) #Servo 5th Axis
        set_servo_angle(6, 90) #Servo 6th Axis

    def loopFifth():
                for angle in range(0, 181, 1):
                    set_servo_angle(5, angle)
                    time.sleep(0.005)

                # Loop from 180 degrees to 0 degrees
                for angle in range(180, -1, -1):
                    set_servo_angle(5, angle)
                    time.sleep(0.005)

    def loopBase():
                for angle in range(0, 181, 1):
                    set_servo_angle(2, angle)
                    time.sleep(0.1)

                # Loop from 180 degrees to 0 degrees
                for angle in range(180, -1, -1):
                    set_servo_angle(2, angle)
                    time.sleep(0.1)


    #Tkinter GUI
    root = tk.Tk()
    root.title("Servo Control")
    root.geometry("400x300")


    #'W' and 'S' keys to the increase_angle and decrease_angle of the Shoulder functions 
    root.bind('<Key-w>', lambda event: increase_servo0and1_angle())
    root.bind('<Key-s>', lambda event: decrease_servo0and1_angle())
    #'A' and 'D' keys to the increase_angle and decrease_angle of the Base functions
    root.bind('<Key-a>', lambda event: increase_servo2_base_angle())
    root.bind('<Key-d>', lambda event: decrease_servo2_base_angle())
    #'Q' and 'E' keys to the increase_angle and decrease_angle of the Elbow functions
    root.bind('<Key-q>', lambda event: increase_servo3_angle())
    root.bind('<Key-e>', lambda event: decrease_servo3_angle())
    #'R' and 'F' keys to the increase_angle and decrease_angle of the Wrist functions
    root.bind('<Key-r>', lambda event: increase_servo4_angle())
    root.bind('<Key-f>', lambda event: decrease_servo4_angle())
    #'T' and 'G' keys to the increase_angle and decrease_angle of the 5th Axis functions
    root.bind('<Key-t>', lambda event: increase_servo5_angle())
    root.bind('<Key-g>', lambda event: decrease_servo5_angle())
    #'Y' and 'H' keys to the increase_angle and decrease_angle of the 6th Axis functions
    root.bind('<Key-y>', lambda event: increase_servo6_angle())
    root.bind('<Key-h>', lambda event: decrease_servo6_angle())

    # Initialized process arm
    root.bind('<Key-z>', lambda event: start_arm_postion())

    #GUI that the list the servo motors and updates the angle of the servo motors
    for i in range(8):  # Updated range to start from 0 and go up to 7
        servo_label = tk.Label(root, text="Servo " + str(i+1) + " Angle: " + str(servo_angles[i]))
        servo_label.grid(row=i, column=0, padx=10, pady=10)
        servo_labels[i] = servo_label  # Add label to dictionary with key i
        

    # Create a function to update the angle of a servo
    def update_servo_angle(servo_number):
        servo_labels[servo_number].config(text=f"Servo  {servo_number+1} Angle: {servo_angles[servo_number]}°")
       
    

if __name__ == "__main__":
    import sys
    Count_Test()

    sys.exit(app.exec_())

########