# import the necessary packages
import argparse
import cv2
import os
import random
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

input_dir = r"/home/u2004/Desktop/AI_Image_Classification/opencv_create_sample/input_image"

output_dir =           r"/home/u2004/Desktop/AI_Image_Classification/opencv_create_sample/output/pos"
opencv_face_pos_path = r"/home/u2004/Desktop/AI_Image_Classification/opencv_create_sample/output/face_pos.info"
# file name format: soluongObject_startX_startY_width_height_randomID.jpg
processed_image_path = r"/home/u2004/Desktop/AI_Image_Classification/opencv_create_sample/output/origin_face_sample"
counter_file_path =    r"/home/u2004/Desktop/AI_Image_Classification/opencv_create_sample/counter.txt"
input_arr = []
start_pos = [0 ,0]

def click_and_crop(event, x, y, flags, param):
    global start_pos, image
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        start_pos[0] = x
        start_pos[1] = y
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        #cv2.imshow("image", image)

    if cropping is True:
        image = clone.copy()
        cv2.rectangle(image, (start_pos[0], start_pos[1]), (x, y), (255, 0, 0), 2)
        cv2.imshow("cv2_image", image)
        print(x, "---", y)

def update_counter(cnt):
    with open(counter_file_path, 'w') as counterFile:
        counterFile.write(str(cnt))

def read_counter():
    try:
        with open(counter_file_path, 'r') as counterFile:
            return int(counterFile.read())
    except Exception as error:
        print(error)
        return 0


for root, dirs, files in os.walk(input_dir, topdown=False):
    for name in files:
        dir_ = os.path.join(root, name)
        #dir_ = dir_.lower()
        input_arr.append(dir_)


start_cnt = read_counter()
for index, img_input in enumerate(input_arr):
    # construct the argument parser and parse the arguments
    # load the image, clone it, and setup the mouse callback function
    cv2_image = cv2.imread(img_input)
    cv2.namedWindow("cv2_image")
    cv2.setMouseCallback("cv2_image", click_and_crop)
    clone = cv2_image.copy()
    # display the image and wait for a keypress
    cv2.imshow("cv2_image", cv2_image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            cv2_image = clone.copy()
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

        # if there are two reference points, then crop the region of interest
        # from teh image and display it
    if len(refPt) == 2:
        start_pos_x = refPt[0][0]
        start_pos_y = refPt[0][1]
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        roi_w = refPt[1][0] - refPt[0][0]
        roi_h = refPt[1][1] - refPt[0][1]
        print(roi_w, " ", roi_h)
        #cv2.imshow("ROI", roi)
        imgGray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_dir + "/" + str(start_cnt) + ".pgm", imgGray)

        # Open a file with access mode 'a'
        file_object = open(opencv_face_pos_path, 'a')
        # Append 'hello' at the end of file
        # pos/pos-0.pgm 1 0 0 100 40
        line_content = "pos/pos-" + str(start_cnt) + ".pgm 1 0 0 " +  str(roi_w) + " " + str(roi_h) + "\n"
        file_object.write(line_content)
        # Close the file
        file_object.close()

        # luu file image kem theo thong tin object sau khi xu ly
        # soluongObject_startX_startY_width_height_randomID.jpg
        prcessed_name = "1_" + str(start_pos_x) + "_" + str(start_pos_x) + "_" + str(roi_w) + "_" + str(roi_h) + "_" + str(random.randrange(1, 1000000000, 1))
        imgGray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(processed_image_path + "/" + prcessed_name + ".pgm", imgGray)

        cv2.waitKey(0)
    start_cnt += 1
    update_counter(start_cnt)
    os.unlink(img_input)

# close all open windows
cv2.destroyAllWindows()

# $ python click_and_crop.py --image jurassic_park_kitchen.jpg