import tensorflow as tf
import keras as kr
import numpy as np
from keras.models import load_model
from sample_data import sample_data

height_arr = []
sizes_arr = []
size_name_arr = ["XS", "S", "M", "L", "XL"]
output_arr = []

# normalize data
for ele in sample_data:
    height_arr.append((ele["height"] - 1540) / 461)
    size_index = size_name_arr.index(ele["size"])
    sizes_arr.append(size_index)
    tmp = [0, 0, 0, 0, 0]
    tmp[size_index] = 1
    output_arr.append(tmp)

input_tf = tf.convert_to_tensor(height_arr, dtype=tf.float32)
size_tf = tf.convert_to_tensor(sizes_arr, dtype=tf.float32)
output_tf = tf.convert_to_tensor(output_arr, dtype=tf.float32)

tf.print(input_tf)
tf.print(size_tf)
tf.print(output_tf)

model = tf.keras.Sequential();

hidden_layer = tf.keras.layers.Dense(
    units=16,  # so luong tensor
    activation="sigmoid",
#    inputDim=1  # co 1 input la height ?
)

output = tf.keras.layers.Dense(
    units=5,  # so luong tensor
    activation="softmax"
)

model.add(hidden_layer)
model.add(output)

# compile model
#model.compile(optimizer=tf.train.sgd(0.2),  # ? sao lai la 0.2
#              loss="categoricalCrossentropy", metrics=["mae"])

model.compile(optimizer="Adam", loss="mse", metrics=["mae"])


# train model. giong kieu dinh nghia cho no
# neu co input la X thi output se la Y
# vd: neu height=1000 thi output se la XS
model.fit(input_tf,
          output_tf,
          epochs=1000,
          validation_split=0.1,  # su dung 10% data de validation
          shuffle=True);

# save model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

print("------- train xong ------------")

