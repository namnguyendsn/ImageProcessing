# import the necessary packages
import argparse
import cv2
import os
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

input_dir = r"/home/u2004/Desktop/AI_Image_Classification/opencv_create_sample/input_image"

output_dir = r"/home/u2004/Desktop/AI_Image_Classification/opencv_create_sample/output"
start_pos = [0 ,0]
input_arr = []

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
        cv2.imshow("image", image)

    if cropping is True:
        image = clone.copy()
        cv2.rectangle(image, (start_pos[0], start_pos[1]), (x, y), (255, 0, 0), 2)

for root, dirs, files in os.walk(input_dir, topdown=False):
    for name in files:
        dir_ = os.path.join(root, name)
        #dir_ = dir_.lower()
        input_arr.append(dir_)


start_cnt = 0
for index, input_img in enumerate(input_arr):
    # construct the argument parser and parse the arguments
    # load the image, clone it, and setup the mouse callback function
    cv2_image = cv2.imread(input_img)

    clone = cv2_image.copy()
    cv2.namedWindow("cv2_image")
    cv2.setMouseCallback("cv2_image", click_and_crop)

    # display the image and wait for a keypress
    cv2.imshow("cv2_image", cv2_image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        cv2_image = clone.copy()
    # if the 'c' key is pressed, break from the loop
    elif key == ord("s"):
        break

    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("ROI", roi)
        cv2.imwrite(output_dir + "/" + str(start_cnt) + ".jpg", roi)
        cv2.waitKey(0)
    start_cnt += 1


# close all open windows
cv2.destroyAllWindows()

# $ python click_and_crop.py --image jurassic_park_kitchen.jpg