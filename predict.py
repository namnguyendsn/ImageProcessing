import tensorflow as tf
import keras as kr
import numpy as np
from keras.models import load_model
from sample_data import sample_data

height_arr = []
sizes_arr = []
size_name_arr = ["XS", "S", "M", "L", "XL"]
output_arr = []

# load trained model
model = load_model('my_model.h5')

while 1:
    testHeighValue = input()
    if type(testHeighValue) == "str":
        break
    testHeighValue = int(testHeighValue)
    heightVal = (testHeighValue - 1540) / 461;
    input_predict = tf.convert_to_tensor([heightVal], dtype=tf.float32)
    predictResult = model.predict(input_predict);
    max_index = np.argmax(predictResult)
    print("size: " + size_name_arr[max_index])




