import cv2
import numpy as np

# Load Model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Loading image and converting to grayscale
img = cv2.imread("./Data/messi5.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Face Detection
faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

# Drawing Rectangle around detected faces
for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
