import cv2
# import sys
# import logging as log
# import datetime as dt
from time import sleep

##load the face cascade into  memory ready for use - this contains the
##data to detect common faces
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
# log.basicConfig(filename='webcam.log', level=log.INFO)

# see https://docs.opencv.org/4.5.2/dd/d43/tutorial_py_video_display.html for details on what I've done here
# open the video camera - 0 is the first camera in the system - can provide a video file here if needed
video_capture = cv2.VideoCapture(0)

##
anterior = 0

##here's the loop to check that the camera is opened and loop capturing the video frame by frame
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture one frame of video
    ret, frame = video_capture.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ##https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    ##Draw a circle around the faces
    ##   for (x, y, w, h) in faces:
    ##   	cv2.circle(frame,(x, y), (x+w, y+h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_PLAIN
    #    cv2.putText(frame,'Curtiss-Wright Face Detection',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

    if anterior != len(faces):
        anterior = len(faces)
        # log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# release the capture
video_capture.release()
cv2.destroyAllWindows()
