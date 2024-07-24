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
while cap.isOpened():
	# returning success and frame
	ret, frame = cap.read()

	# converting video capture in grayscale
	grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# face detection
	faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)

	# draw red circle indicator if there is not face detected
	detection_success_circle = cv2.circle(frame, (15, 15), (10), (0, 0, 255), -1)

	# text must have font instance and drawing text
	# font instance
	font = cv2.FONT_HERSHEY_SIMPLEX

	# drawing date and time (imageVariable, "textString", (startingPointX, startingPointY), fontInstance, fontSize, (blueColor, greenColor, redColor), tickness, drawingMethod)
	drawing_date_and_time_on_frame_img = cv2.putText(frame, f"{strftime("%Y-%m-%d %H:%M:%S")}", (30, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

	# drawing frame width and height (imageVariable, "textString", (startingPointX, startingPointY), fontInstance, fontSize, (blueColor, greenColor, redColor), tickness, drawingMethod)
	drawing_frame_width_and_height_on_frame_img = cv2.putText(frame, f"w:{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}, h:{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}", (10, 40), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

	# looping through face detection
	for (x, y, width, height) in faces:

		# draw rectangle on face
		cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

		# draw green circle indicator if there is face detected
		detection_success_circle = cv2.circle(frame, (15, 15), (10), (0, 255, 0), -1)

		# if there is face save as image
		detected = cv2.imwrite(f"{image_path}/face_detected_{strftime("%Y%m%d_%H%M%S")}.jpg", frame)

	# displaying the video image
	cv2.imshow("Live Face Detection With Python3 And OpenCV2", frame)

	# destroying the window if specific key (event) is pressed
	if cv2.waitKey(1) == ord("q"):
		break

# liberating camera source for other programs
cap.release()

# destroying windows
cv2.destroyAllWindows()