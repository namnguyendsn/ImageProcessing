import cv2
from time import sleep
import numpy as np

kernel = np.ones((5, 5), np.uint8)
imgW = 640
imgH = 480

Webcam_capture = cv2.VideoCapture("http://192.168.2.27:8080/?action=stream")
Webcam_capture.set(3, 1280)
Webcam_capture.set(4, 720)
Webcam_capture.set(10, 0)


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    imgDial = cv2.dilate(imgCanny, kernel, iterations = 2)
    imgThres = cv2.erode(imgDial, kernel, iterations = 1)

    return imgThres


def getConlouts(img):
    x, y, w, h = 0, 0, 0, 0
    maxArea = 0
    biggest = np.array([])
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area > 5000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if (area > maxArea) and (len(approx) == 4):
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 30)
    return biggest
            #x, y, w, h = cv2.boundingRect(approx)

def reorder(myPoints):
#    if len(myPoints) < 1:
#        return myPoints
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    #print("add", add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    #print("NewPoints", myPointsNew)

    diff = np.diff(myPoints, axis = 1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def getWarp(img, biggest):
    biggest = reorder(biggest)
    # toa do 4 canh cua anh can lay
    pts1 = np.float32(biggest)

    pts2 = np.float32([[0, 0],
                      [imgW, 0],
                      [0, imgH],
                      [imgW, imgH]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (imgW, imgH))

    return imgOutput

while True:
    # read image from local
    success, img = Webcam_capture.read()
    imgContour = img.copy()
    cv2.resize(img, (imgW, imgH))
    imgThres = preProcessing(img)
    biggest = getConlouts(imgThres)
    imgWarped = getWarp(img, biggest)


    cv2.imshow("Day la khung anh", img)
    cv2.imshow("Day la imgContour", imgContour)
    cv2.imshow("Day la imgThres", imgWarped)
    cv2.imshow("Day la imgThres", imgThres)

    # wait for keys
    cv2.waitKey(1)