import cv2
import numpy as np
import urllib.request
import pyttsx3  # For text-to-speech
import time  # For cooldown between alerts
import requests
import threading  # For non-blocking Arduino communication
import queue
import os

# Camera IP Address
#ESP32_IP = "http://192.168.8.100"  # Replace this with your ESP32-CAM IP address
# Camera URL
#url = 'http://192.168.8.100/cam-hi.jpg'
#cap = cv2.VideoCapture(url)

def find_active_esp32_ip(start=100, end=200):
    base_ip = "192.168.8."
    for i in range(start, end + 1):
        test_url = f"http://{base_ip}{i}/cam-hi.jpg"
        try:
            img_resp = urllib.request.urlopen(test_url, timeout=2)
            if img_resp.status == 200:
                print(f"ESP32 found at {base_ip}{i}")
                return base_ip + str(i)
        except:
            continue
    print("No active ESP32 found in range.")
    return None

# Attempt to find ESP32 in IP range 192.168.8.100–110
found_ip = find_active_esp32_ip()
if found_ip is None:
    exit("No ESP32 found. Exiting...")

ESP32_IP = f"http://{found_ip}"
url = f"{ESP32_IP}/cam-hi.jpg"
cap = cv2.VideoCapture(url)


# YOLO configuration
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile = 'coco.names'
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Create a speech queue
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break  # Exit signal
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start the speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


# Define animals to alert for
animal_classes = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

# Variables for alert cooldown
last_alert_time = 0
alert_cooldown = 3  # seconds between alerts

# Arduino communication setup

def send_to_arduino(message):
    try:
        if message.startswith("A:"):
            animal = message.split(":")[1]
            response = requests.get(f"{ESP32_IP}/alert?animal={animal}")
            print("HTTP Request sent to Arduino. Response:")
            print(response.text)
        elif message.startswith("S:START"):
            response = requests.get(f"{ESP32_IP}/start")
            print("System Start message sent. Arduino says:")
            print(response.text)
        elif message.startswith("S:STOP"):
            response = requests.get(f"{ESP32_IP}/stop")
            print("System Stop message sent. Arduino says:")
            print(response.text)
    except Exception as e:
        print(f"Error sending HTTP request to Arduino: {e}")


def speak_alert(animal_type):
    """Generate voice alert for detected animal and send to Arduino"""
    global last_alert_time
    current_time = time.time()
    
    if current_time - last_alert_time > alert_cooldown:
        alert_text = f"{animal_type} detected ahead, please slow down your vehicle"
        print(f"ALERT: {alert_text}")
        
        # Start text-to-speech in a separate thread to avoid blocking
        #threading.Thread(target=lambda: engine.say(alert_text) or engine.runAndWait()).start()


        # Add alert to speech queue instead of using threads
        speech_queue.put(alert_text)
        
        # Send alert to Arduino
        #requests.get(f"{ESP32_IP}/alert?animal={animal_type}")
        send_to_arduino(f"A:{animal_type}")
                 
        last_alert_time = current_time

# Load YOLO model
modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    detected_animals = []  # Track detected animals for alerts
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    indexes = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    # Different colors for different animal types (BGR format)
    color_map = {
        'dog': (0, 255, 0),      # Green
        'cat': (255, 0, 0),      # Blue
        'bird': (0, 0, 255),     # Red
        'horse': (255, 255, 0),  # Cyan
        'sheep': (255, 0, 255),  # Magenta
        'cow': (0, 255, 255),    # Yellow
        'elephant': (128, 0, 0), # Dark blue
        'bear': (0, 128, 0),     # Dark green
        'zebra': (0, 0, 128),    # Dark red
        'giraffe': (128, 128, 0) # Dark cyan
    }
    
    # Default color for non-animal objects
    default_color = (255, 0, 255)  # Magenta
   
    for i in indexes:
        # Handle different OpenCV versions
        #i = i if isinstance(i, int) else i[0]
        i = int(i)  # This safely converts [0], [[0]], or scalar to plain int
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        class_name = classNames[classIds[i]]
        
        # Set color based on class
        if class_name.lower() in animal_classes:
            color = color_map.get(class_name.lower(), default_color)
            detected_animals.append(class_name)
        else:
            color = default_color
            
        cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
        cv2.putText(im, f'{class_name.upper()} {int(confs[i] * 100)}%', 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add detection stats to the image
    cv2.putText(im, f"Animals detected: {len(detected_animals)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Generate voice alerts for detected animals
    if detected_animals:
        # Alert for the first animal detected (to avoid multiple alerts)
        speak_alert(detected_animals[0])

def cleanup():
    """Clean up resources before exit"""
    """if arduino_connected:
        print("Closing Arduino connection...")
        arduino.close()"""
    cv2.destroyAllWindows()
    # Stop the speech thread gracefully
    speech_queue.put(None)
    speech_thread.join()
    print("Resources cleaned up")

# Main program
if __name__ == "__main__":
    print("Starting Animal Detection System with Arduino integration...")
    
    # Send startup message to Arduino
    #if arduino_connected:
    send_to_arduino("S:START")  # S for System
    
    try:
        while True:
            try:
                img_resp = urllib.request.urlopen(url)
                imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                im = cv2.imdecode(imgnp, -1)
                
                if im is None:
                    print("Warning: Failed to get image from camera")
                    time.sleep(1)
                    continue
                
                blob = cv2.dnn.blobFromImage(im, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                
                layernames = net.getLayerNames()
                # Handle different OpenCV versions
                try:
                    outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
                except:
                    outputNames = [layernames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                
                outputs = net.forward(outputNames)
                findObject(outputs, im)
                
                cv2.imshow('Animal Detection', im)
                
                key = cv2.waitKey(1)
                if key == ord('q'):  # Press 'q' to quit
                    break
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if arduino_connected:
            send_to_arduino("S:STOP")  # Notify Arduino that we're shutting down
        cleanup()

