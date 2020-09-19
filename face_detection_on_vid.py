# Import required libraries
import cv2
import numpy as np

# An instance of the class cv2.VideoCapture() to capture the frame
cap = cv2.VideoCapture("./Data/Megamind.avi")

# PropIds of the frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# An instance of the class cv2.VideoWriter() to save the frame
fourcc = cv2.VideoWriter_fourcc(*"XVID")
saved_frame = cv2.VideoWriter("face_detection.avi", fourcc, 20.0, (frame_width, frame_height))

# Loading the Haar Cascade Classifier 
model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Capturing frames
while cap.isOpened():
    _, frame = cap.read()                                      # Reading the frame
    
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Converting the frame to grayscale
        faces = model.detectMultiScale(gray_frame, 1.1, 10)    # Detecting faces and storing the co-ordinates in the variable faces

        for x, y, w, h in faces:                               # Drawing rectangles around the detected cars
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        saved_frame.write(frame)                               # Save the frame
        cv2.imshow("frame", frame)                             # Show the frame 

        if cv2.waitKey(1) & 0xFF == 27:                        # Stop capturing the frames when esc key is pressed
            break
    
    except Exception:
        break


cap.release()                                                 # Release the captured frame
saved_frame.release()                                         # Release the saved frame'
cv2.destroyAllWindows()                                       # Close the window


