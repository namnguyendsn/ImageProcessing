import cv2
from time import sleep
import numpy as np

kernel = np.ones((5, 5), np.uint8)

def empty(val):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 97, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 116, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 222, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

Webcam_capture = cv2.VideoCapture("http://192.168.2.27:8080/?action=stream")
Webcam_capture.set(3, 1280)
Webcam_capture.set(4, 720)
Webcam_capture.set(10, 0)
while True:
    # read image from local
    success, img = Webcam_capture.read()

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    hue_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    val_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    val_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(hue_min, hue_max, sat_min, sat_max, val_min, val_max)

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    mask = cv2.inRange(imgHSV, lower, upper)

    imgResult = cv2.bitwise_and(img, img, mask=mask)

    # show in a window
    cv2.imshow("Day la khung anh", img)
    cv2.imshow("Day la khung anh HSV", imgHSV)
    cv2.imshow("Day la khung anh HSV thay doi", mask)
    cv2.imshow("Day la khung anh imgResult", imgResult)


    # wait for keys
    cv2.waitKey(1)