import cv2
import time

cap = cv2.VideoCapture("udpsrc address=192.168.5.2 port=5000 ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()  # ret is True if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow('frame', frame)
    if cv2.pollKey() == ord('q'):
        break
    
    if cv2.pollKey() == ord('c'):
        cv2.imwrite(f'camera_images/cam_5/{time.strftime("%Y-%m-%d %H-%M-%S")}.jpg', frame)

cap.release()
cv2.destroyAllWindows()

