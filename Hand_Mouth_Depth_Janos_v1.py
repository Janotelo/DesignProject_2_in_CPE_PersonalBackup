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
import Gesture as ges

###################################################################################################
#Came Inside of Count_Test Function
import depthai as dai
import blobconverter
import Depth as dep
import dlib
import numpy as np
#Hand, Mouth, and Depth

###################################################################################################
def Count_Test():
    
    detector = hdm.handDetector(detectionCon=0.8, maxHands=1)
    hand_gesture = ges.Gesture()

    
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
                #print(totalFingers)
                hand_gesture.Fingers(count)

            
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
    
       
 
if __name__ == "__main__":
    import sys
    Count_Test()


    
   

    sys.exit(app.exec_())