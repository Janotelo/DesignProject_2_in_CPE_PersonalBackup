import Adafruit_PCA9685
import time
import tkinter as tk
import threading

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

def start_servo():
    new_angle2_reversed = servo_angles[0] + 20
    new_angle2_correct = servo_angles[1] - 20
    new_angle_base = servo_angles[2]
    new_angle_elbow = servo_angles[3] + 50
    new_angle_wrist = servo_angles[4] - 16
    new_angle_fifth = servo_angles[5] + 15
    new_angle_sixth = servo_angles[6] - 60
    set_servo_angle(0, new_angle2_reversed)
    set_servo_angle(1, new_angle2_correct)
    set_servo_angle(2, new_angle_base)
    set_servo_angle(3, new_angle_elbow)
    set_servo_angle(4, new_angle_wrist)
    set_servo_angle(5, new_angle_fifth)
    set_servo_angle(6, new_angle_sixth)
    print("Shoulder_R :", new_angle2_reversed)
    print("Shoulder_C :", new_angle2_correct)
    print("Base       :", new_angle_base)
    print("Elbow      :", new_angle_elbow)
    print("Wrist      :", new_angle_wrist)
    print("Fifth      :", new_angle_fifth)
    print("Sixth      :", new_angle_sixth)



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
    

root.mainloop()
