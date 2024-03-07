import numpy as np
import cv2 as cv
# import cv2.aruco

# Host command: rpicam-vid -t 0 -n --inline --framerate 60 --intra 1 -o - | gst-launch-1.0 fdsrc fd=0 ! h264parse ! rtph264pay pt=96 ! udpsink host=192.168.5.2 port=5000 sync=false
# cap = cv.VideoCapture("udpsrc address=192.168.100.101 port=5000 caps=application/x-rtp ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true", cv.CAP_GSTREAMER)
cap = cv.VideoCapture("udpsrc address=192.168.5.2 port=5000 ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv.CAP_GSTREAMER)
# cap = cv.VideoCapture("videotestsrc ! autovideoconvert ! appsink", cv.CAP_GSTREAMER)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

# print(cv.getBuildInformation())