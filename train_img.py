import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, models

from config import net_encoder, PATH as IMAGE_PATH

input_shape = (32, 32, 3)


def load_img(path_to_img):
    # 读取的图片已转化为 0 -1 之间，不用/255
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (32, 32))
    return img


x_train = np.zeros((1000, 32, 32, 3), dtype='float32')
y_train = np.zeros((1000, net_encoder.__len__()))
for i in range(x_train.shape[0]):
    num_dir = os.listdir(IMAGE_PATH)
    k = i % len(num_dir)
    path = IMAGE_PATH + '\\' + num_dir[k]
    rand_img = path + '\\' + np.random.choice(os.listdir(path))
    x_train[i] = load_img(rand_img)
    y_train[i, net_encoder[int(num_dir[k])]] = 1
x_train = x_train

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())  # 加了bn层结果看不懂了
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(net_encoder.__len__(), activation='softmax'))

model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=4)
model.save('recognize_new')
print('acu history:',history.history['accuracy'])



def show_cnn_result():
    pred_y = model.predict(x_train)
    for ep in range(5):
        for i in range(11):
            plt.subplot(11, 3, i * 3 + 1)
            plt.imshow(x_train[i + ep * 3])
            plt.subplot(11, 3, i * 3 + 2)
            plt.bar(range(len(pred_y[i + ep * 9])), pred_y[i + ep * 3])
            plt.subplot(11, 3, i * 3 + 3)
            plt.bar(range(len(y_train[i + ep * 9])), y_train[i + ep * 3])
        plt.show()
# show_cnn_result()

