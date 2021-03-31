import cv2
from time import sleep
import numpy as np

kernel = np.ones((5, 5), np.uint8)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                         scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getConlouts(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            #print(len(approx))
            objColor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objColor == 3:
                ObjectType_ = "Tri"
            elif objColor == 4:
                aspRatio = w/float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    ObjectType_ = "Square"
                else:
                    ObjectType_ = "Rec"
            else:
                ObjectType_ = "Circles"

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (106, 13, 173), 4)
            cv2.putText(imgContour,
                        ObjectType_,
                        (x + (w//2) - 10, y + (h//2) - 10),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (0, 0, 0),
                        2)




if 0: # read an image from local
    # read image from local
    img = cv2.imread(r"/home/u2004/Desktop/AI_Image_Classification/sample/img_2.jpg")

    # show in a window
    cv2.imshow("Day la khung anh", img)


    # wait for keys
    cv2.waitKey()

if 0: # Read video from local
    path = r"/home/u2004/Desktop/AI_Image_Classification/sample/testLivestream.mp4"
    # define video's path in cv2
    VideoCapture = cv2.VideoCapture(path)
    while 1:
        # read video from local
        success, img = VideoCapture.read()
        cv2.imshow("day la video", img)
        sleep(0.03)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if 0: # read video from webcam (or livestream url)
    Webcam_capture = cv2.VideoCapture("http://192.168.2.27:8080/?action=stream")
    Webcam_capture.set(3, 1280)
    Webcam_capture.set(4, 720)
    Webcam_capture.set(10, 0)
    while 1:
        success, img = Webcam_capture.read()
        cv2.imshow("Video from webcam", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if 0: # mot vai ham bien doi hinh anh
    img = cv2.imread(r"/home/u2004/Desktop/AI_Image_Classification/sample/img_1.jpg")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # anh den trang
    imgBlur = cv2.GaussianBlur(imgGray, (11, 111), 0)  # anh bi lam mo. y nghia tham so 111,111 ?
    imgCanny1 = cv2.Canny(imgBlur, 200, 200)  # anh chi co cac duong vien, cac tham so 200, 200 y nghia nhu nao ?
    imgCanny2 = cv2.Canny(img, 350, 350)  # anh chi co cac duong vien, cac tham so 200, 200 y nghia nhu nao ?
    imgDialation = cv2.dilate(imgCanny2, kernel, iterations=10)  # tang doi do rong cua duong vien sau khi canny
    imgEroded = cv2.erode(imgCanny2, kernel, iterations=1)  # ham nay nguoc lai voi dilate, lam giam do rong cua duong vien sau khi canny

    cv2.imshow("day la anh den trang", imgGray)
    cv2.imshow("day la anh bi lam mo", imgBlur)
    cv2.imshow("day la anh chi con cac duong vien (src=img)", imgCanny1)  # image cua Canny phai la hinh anh ro net
    cv2.imshow("day la anh chi con cac duong vien (src=imgBlur)", imgCanny2)
    cv2.imshow("day la anh sau khi canny va dialation", imgDialation)
    cv2.imshow("day la anh sau khi canny va erode", imgEroded)  # hinh anh sau khi erode se bi mat rat nhieu chi tiet

    cv2.waitKey(0)

if 0: # resize and crop
    img = cv2.imread(r"/home/u2004/Desktop/AI_Image_Classification/sample/img_1.jpg")
    print(img.shape)
    imgResize = cv2.resize(img, (150, 150))
    print(imgResize.shape)

    img_cropped_arr = img[0:200, 200:300]  # xu ly nhu index trong array

    cv2.imshow("day la anh", img)
    cv2.imshow("day la anh resize1", imgResize)
    cv2.imshow("day la anh resize2", img_cropped_arr)

    cv2.waitKey(0)

if 0: # ve hinh va text tren anh
    img = np.zeros((512, 512, 3), np.uint8)
    print(img.shape)

    img[200:300, 300:500] = 255, 100, 100  # fill background color
    cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)  # ve 1 duong cheo tren img, mau (0, 255, 0)
    cv2.rectangle(img,
                  (100, 100),
                  (200, 200),
                  (255, 0, 0),
                  cv2.FILLED)  # ve 1 hinh chu nhat tren img, start port (100, 100), stop point (200, 200), mau (255, 0, 0), fill
    cv2.rectangle(img,
                  (300, 300),
                  (400, 400),
                  (255, 255, 0),
                  10)  # ve 1 hinh chu nhat tren img, start port (100, 100), stop point (200, 200), mau (255, 255, 0), do day vien 10

    cv2.circle(img, (256, 256), 100, (0, 0, 255), 20)
    cv2.putText(img,
                "OPENCV",
                (10, 500),
                cv2.FONT_HERSHEY_COMPLEX,
                2,  # Font size ?
                (0, 150, 0),
                5)  # them doan text OPENCV vao img, vi tri (100, 500), mau (0, 150, 0), chu beo 5

    cv2.imshow("Blank image", img)

    cv2.waitKey(0)

if 0: # prespective cha biet dich la gi
    img = cv2.imread(r"/home/u2004/Desktop/AI_Image_Classification/sample/cards.jpg")
    print(img.shape)
    cv2.imshow("cards", img)

    # toa do 4 canh cua anh can lay
    pts1 = np.float32([[148, 105],
                      [216, 83],
                      [174, 201],
                      [246, 179]])

    width, height = 100, 150
    pts2 = np.float32([[0, 0],
                      [width, 0],
                      [0, height],
                      [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))

    # neu apply them Gray va Canny thi sao ?
    imgGray = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2GRAY)  # anh den trang
    imgCanny2 = cv2.Canny(imgOutput, 90, 90)  # anh chi co cac duong vien, cac tham so 200, 200 y nghia nhu nao ?

    cv2.imshow("cards transformed", imgOutput)
    cv2.imshow("cards Gray", imgGray)
    cv2.imshow("cards canny", imgCanny2)

    cv2.waitKey(0)

if 0: # ghep nhieu anh vao 1 khung
    img = cv2.imread(r"/home/u2004/Desktop/AI_Image_Classification/sample/img_5.jpg")
    img1 = cv2.imread(r"/home/u2004/Desktop/AI_Image_Classification/sample/img_1.jpg")
    print(img.shape)
    imgHor = np.hstack((img, img))
    imgVer = np.vstack((img, img))

    multiImg = stackImages(0.5, ([img, img1, img], [img1, img, img1]))

    cv2.imshow("Horizontal image", imgHor)
    cv2.imshow("Vertical image", imgVer)
    cv2.imshow("Multi image", multiImg)
    cv2.waitKey(0)

if 1: # color detection from image
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


    while True:
        # read image from local
        img = cv2.imread(r"/home/u2004/Desktop/AI_Image_Classification/sample/img_1.jpg")
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

if 0: # contour and shape detection from Image
    # read image from local
    img = cv2.imread(r"/home/u2004/Desktop/AI_Image_Classification/sample/shape.jpg")
    imgContour = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    imgCannyGray = cv2.Canny(imgGray, 50, 50)
    getConlouts(imgCanny)

    imgStack = stackImages(0.75, ([img, imgGray, imgBlur],
                                 [imgCanny, imgCannyGray, imgContour]))

    # show in a window
    #cv2.imshow("Anh goc", img)
    #cv2.imshow("Gray", imgGray)
    cv2.imshow("Blue", imgStack)


    # wait for keys
    cv2.waitKey()

if 0: # contour and shape detection from livestream

    Webcam_capture = cv2.VideoCapture("http://192.168.2.27:8080/?action=stream")
    Webcam_capture.set(3, 1280)
    Webcam_capture.set(4, 720)
    Webcam_capture.set(10, 0)
    while 1:
        success, img = Webcam_capture.read()
        cv2.imshow("Video from webcam", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        imgContour = img.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
        imgCanny = cv2.Canny(imgBlur, 50, 50)
        imgCannyGray = cv2.Canny(imgGray, 50, 50)
        getConlouts(imgCanny)

        imgStack = stackImages(0.75, ([img, imgGray, imgBlur],
                                     [imgCanny, imgCannyGray, imgContour]))

        # show in a window
        #cv2.imshow("Anh goc", img)
        #cv2.imshow("Gray", imgGray)
        cv2.imshow("Blue", imgStack)


        # wait for keys
        cv2.waitKey(1)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

if 0: # face detection
    faceCascade = cv2.CascadeClassifier("/home/u2004/Desktop/AI_Image_Classification/haarcascades/haarcascade_frontalface_default.xml")
    path = r"/home/u2004/Desktop/AI_Image_Classification/1interview_1080p.mp4"
    # define video's path in cv2
    #Webcam_capture = cv2.VideoCapture(path)
    Webcam_capture = cv2.VideoCapture("http://192.168.2.27:8080/?action=stream")

    Webcam_capture.set(3, 1280)
    Webcam_capture.set(4, 720)
    #Webcam_capture.set(10, 0)
    from time import sleep
    while 1:
        success, img = Webcam_capture.read()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        img = rescale_frame(img, percent=50)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Result", img)

        # wait for keys
        cv2.waitKey(1)
        sleep(0.03)


