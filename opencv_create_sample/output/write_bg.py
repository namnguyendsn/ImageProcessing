import os
import cv2

input_dir = r"/home/u2004/Desktop/AI_Image_Classification/opencv_create_sample/input_neg"
bg_out_path = r"/home/u2004/Desktop/AI_Image_Classification/opencv_create_sample/output/neg"
bg_file_path = r"/home/u2004/Desktop/AI_Image_Classification/opencv_create_sample/output/bg.txt"
bg_counter_file_path = r"/home/u2004/Desktop/AI_Image_Classification/opencv_create_sample/output/bg_counter.txt"

def update_bg(cnt):
    with open(bg_file_path, 'a') as counterFile:
        counterFile.write(str(cnt))

def update_counter(cnt):
    with open(bg_counter_file_path, 'w') as counterFile:
        counterFile.write(str(cnt))

def read_counter():
    try:
        with open(bg_counter_file_path, 'r') as counterFile:
            return int(counterFile.read())
    except Exception as error:
        print(error)
        return 0

counter = read_counter()
for root, dirs, files in os.walk(input_dir, topdown=False):
    for name in files:
        dir_ = os.path.join(root, name)
        update_bg("neg/" + str(counter) + ".jpg" + "\n")
        img = cv2.imread(dir_)
        cv2.imwrite(bg_out_path + "/" + str(counter) + ".jpg", img)
        counter += 1
        update_counter(counter)
