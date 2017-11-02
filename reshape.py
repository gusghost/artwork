import sys
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

for dir in os.listdir("data/train"):
    if dir == ".DS_Store":
        continue
        dir1 = "data/train/" + dir
for file in os.listdir(dir1):
    if file != ".DS_Store":
        # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
        label_list.append(label)
        filepath = dir1 + "/" + file
        # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
        # [R,G,B]はそれぞれが0-255の配列。
        image = np.array(Image.open(filepath).resize((25, 25)))
        print(filepath)
        # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
        image = image.transpose(2, 0, 1)
        # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
        image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
        # 出来上がった配列をimage_listに追加。
        image_list.append(image / 255.)
