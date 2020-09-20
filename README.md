# Face-detection
## Face detection using OpenCV in Python

### This project detects faces in an image and in a video frame using a pre-trained Haar Cascade Classifier.    
#### Here are some snippets

#### In a video
![face_detection](https://user-images.githubusercontent.com/61016383/93594940-69b3de80-f9d4-11ea-8df3-41ec459a653a.gif)

#### In an image
![messi_face](https://user-images.githubusercontent.com/61016383/93595051-9f58c780-f9d4-11ea-88fe-0a97f90282a3.jpg)

### Libraries used
  - OpenCV
  - Numpy

### Steps to use this project:
 1. [Clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) this repo
 
 2. Install Python 3.6 or greater `pip install python`
 
 3. Intsall OpenCV for Python `pip install opencv-python`. This package includes Numpy too.
 
 4. Go to the directory of this repo cloned onto your local machine and open cmd.
 
 5. Run the python files of your choice by typing the file name in the cmd as `face_detection.py` or `face_detection_on_vid.py`
 
### You can try this project to detect your own face through your webcam by following these steps:
  1. Open file face_detection_on_vid.py in any Python editor of your choice
  
  2. Go to line 4 of the python script which reads `cap = cv2.VideoCapture("./Data/Megamind.avi")`
  
  3. Replace the argument specified as `"./Data/Megamind.avi"` with `0`
  
  4. Save the file and run it.
