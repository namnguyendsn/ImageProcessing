import cv2
from time import sleep
import numpy as np

kernel = np.ones((5, 5), np.uint8)

Webcam_capture = cv2.VideoCapture("http://192.168.2.27:8080/?action=stream")
Webcam_capture.set(3, 1280)
Webcam_capture.set(4, 720)
Webcam_capture.set(10, 0)

myColors = [[55, 116, 64, 255, 110, 255], # blue
            [0, 83, 84, 191, 118, 255]] # orange

ColorValue = [[255, 0, 0], [0, 255, 0]]

myPoint = [] # x, y, colorID

def drawOnCanvas(mypoint, mycolorvalue):
    for point in mypoint:
        cv2.circle(imgResult, (point[0], point[1]), 10, ColorValue[point[2]], cv2.FILLED)

def findColor(img, myColors_):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    newPoint = []
    for index, color in enumerate(myColors_):
        lower = np.array([color[0], color[2], color[4]])
        upper = np.array([color[1], color[3], color[5]])

        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getConlouts(mask)
        cv2.circle(imgResult, (x, y), 10, ColorValue[index], cv2.FILLED)
        if x != 0 and y != 0:
            newPoint.append([x, y, index])
        #cv2.imshow(str(color[0]), mask)
    return newPoint

def getConlouts(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area > 500:
            cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w//2, y

while 1:
    success, img = Webcam_capture.read()
    imgResult = img.copy()
    newPoint = findColor(img, myColors)
    if len(newPoint) != 0:
        for newP in newPoint:
            myPoint.append(newP)

    if len(myPoint) != 0:
        drawOnCanvas(myPoint, ColorValue)

    else:
        pass

    cv2.imshow("Video from webcam", imgResult)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break