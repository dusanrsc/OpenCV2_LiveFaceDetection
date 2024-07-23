# importing modules
import cv2
import numpy
import os

# importing sub-modules
from time import strftime

# reading video image from camera
cap = cv2.VideoCapture(0)

# importing detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# image path
image_path = "detected_images"

# creating directory for image storage
os.system(f"mkdir {image_path}")

# main program loop
while True:
	# returning success and frame
	ret, frame = cap.read()

	# converting video capture in grayscale
	grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# face detection
	faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)

	# draw red circle indicator if there is not face detected
	detection_success_circle = cv2.circle(frame, (15, 15), (10), (0, 0, 255), -1)

	# looping through face detection
	for face in faces:

		# draw green circle indicator if there is face detected
		detection_success_circle = cv2.circle(frame, (15, 15), (10), (0, 255, 0), -1)

		# if there is face save as image
		detected = cv2.imwrite(f"{image_path}/face_detected_{strftime("_%Y%m%d_%H%M%S")}.jpg", frame)

	# displaying the video image
	cv2.imshow("Live Face Detection With Python3 And OpenCV2", frame)

	# destroying the window if specific key (event) is pressed
	if cv2.waitKey(1) == ord("q"):
		break

# liberating camera source for other programs
cap.release()

# destroying windows
cv2.destroyAllWindows()