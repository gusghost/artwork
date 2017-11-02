from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import random

# 学習用のデータを作る.
image_list = []
label_list = []
epoch_num = 20
list_idx = 0
# ./data/train 以下のorange,appleディレクトリ以下の画像を読み込む。
for dir in os.listdir("X:/library/document/卒業制作/first_prototype04/train"):
    if dir == ".DS_Store":
        continue

    dir1 = "../../first_prototype04/train/" + dir
    label = 0

    if dir == "maru":  # appleはラベル0
        label = 0
    elif dir == "shikaku":  # orangeはラベル1
        label = 1

    for file in os.listdir(dir1):  # listdirってどんな関数なんだ。。。
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
            label_list.append(label)

            filepath = dir1 + "/" + file

            # [R,G,B]はそれぞれが0-255の配列。
            image = np.array(Image.open(filepath))
            print(filepath)
            image = image.flatten()
            # [[number1,n2, n3, ..., nn]]
            # 出来上がった配列をimage_listに追加。
            image_list.append(image / 255.)
            # print(image_list)
            # print(image_shape)
            list_idx += 1

# kerasに渡すためにnumpy配列に変換。
# image_list = random.shuffle(image_list)
print(image_list)
image_list = np.array(image_list)
print(image_list)
# image_list = np.reshape(image_list, (list_idx, 784))
# print(image_list)
# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0], 1 -> [0,1] という感じ。
Y = to_categorical(label_list)
print('Y: ', Y)
# モデルを生成してニューラルネットを構築
model = Sequential()
# input_dimは次元の数らしい。denseとinputの数は最初に設定した次元の数と一致している必要がある
model.add(Dense(784, input_dim=784))
model.add(Activation("relu"))
# model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation("relu"))
# model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation("softmax"))

# オプティマイザにAdamを使用
opt = Adam(lr=0.001)
# モデルをコンパイル
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# validation_split ha
model.fit(image_list, Y, epochs=epoch_num, batch_size=100, validation_split=0.1)

# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
total = 0.
ok_count = 0.

for dir in os.listdir("X:/library/document/卒業制作/first_prototype04/train"):
    if dir == ".DS_Store":
        continue  # continue is nani

    dir1 = "X:/library/document/卒業制作/first_prototype04/train/" + dir
    label = 0

    if dir == "maru":  # appleはラベル0
        label = 0
    elif dir == "shikaku":  # orangeはラベル1
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            label_list.append(label)  # 型宣言が無いとどうもやりづらいというか頭が混乱するなぁ
            filepath = dir1 + "/" + file
            # image = np.array(Image.open(filepath).resize((25, 25)))
            print(filepath)
            # image = image.transpose(2, 0, 1)
            image = np.reshape(image, (1, 784))
            # random.shuffle(image)
            # なぜnp.arrayすると次元というかカッコが一つ増えるのか謎なんだが。が。
            image = np.array(image / 255.)
            # print(image)
            result = model.predict_classes(image)
            print("label:", label, "result:", result[0])
            total += 1.
            if label == result[0]:
                ok_count += 1.

print("correct: ", ok_count / total * 100, "%")
print("epochs: ", epoch_num)
