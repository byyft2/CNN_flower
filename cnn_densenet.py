from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
import numpy as np
from PIL import Image

import os, re, glob
import cv2
from sklearn.model_selection import train_test_split

accident_dir = "/home/jongmin/project/jongp/datasets/dataset/train_1/"
categories = ["gunzaran", "sugook", "sunflower"]
nb_classes = len(categories)
# 이미지 크기 지정
image_w = 64
image_h = 64
pixels = image_w * image_h * 3
# 이미지 데이터 읽어 들이기
X = []
Y = []
for idx, cat in enumerate(categories):
    # 레이블 지정
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지
    image_dir = accident_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)      # numpy 배열로 변환
        X.append(data)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)
# 학습 전용 데이터와 테스트 전용 데이터 구분
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y)
print(len(X_train))

model = tf.keras.applications.DenseNet121(
    include_top=True, weights=None, input_tensor=None, input_shape=(64,64,3),
    pooling=None, classes=3
)

model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='adam',
    metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=400, verbose=1,
          validation_data=(X_test,y_test))

model.save_weights("/home/jongmin/project/jongp/weight_dir/flower_weight_6_14_2_dense.h5")
print("완료")
