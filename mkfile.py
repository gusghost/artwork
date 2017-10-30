from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

# 学習用のデータを作る.
image_list = []
label_list = []

# ./data/train 以下のorange,appleディレクトリ以下の画像を読み込む。
for dir in os.listdir("./train"):
    if dir == ".DS_Store":
        continue

    dir1 = "./train/" + dir
    label = 0

    if dir == "maru_b":  # appleはラベル0
        label = 0
    elif dir == "shikaku_g":  # orangeはラベル1
        label = 1
    elif dir == "tri_g":  # orangeはラベル1
        label = 2
    elif dir == "tri_b":  # orangeはラベル1
        label = 3

    for file in os.listdir(dir1):  # listdirってどんな関数なんだ。。。
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
            label_list.append(label)
            filepath = dir1 + "/" + file

            # [R,G,B]はそれぞれが0-255の配列。
            image = np.array(Image.open(filepath))
            print(filepath)

            # image = image.transpose(2, 0, 1)

            # これ白黒画像だからか１ピクセルにつき一つの数値ですね
            # image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            image = np.reshape(image, (1, 784))
            # print(image)
            # 出来上がった配列をimage_listに追加。
            image_list.append(image / 255.)
            # print(image_list)
            # print(image_shape)

# kerasに渡すためにnumpy配列に変換。
image_list = np.array(image_list)
# print(image_list)
# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0], 1 -> [0,1] という感じ。
Y = to_categorical(label_list)
# モデルを生成してニューラルネットを構築
model = Sequential()
# input_dimは次元の数らしい。denseとinputの数は最初に設定した次元の数と一致している必要がある
model.add(Dense(256, input_dim=156568))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(4))
model.add(Activation("softmax"))

# オプティマイザにAdamを使用
opt = Adam(lr=0.001)
# モデルをコンパイル
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(image_list, Y, epochs=200, batch_size=100, validation_split=0.1)

# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
total = 0.
ok_count = 0.

for dir in os.listdir("./test"):
    if dir == ".DS_Store":
        continue  # continue is nani

    dir1 = "./test/" + dir
    label = 0

    if dir == "maru_b":  # appleはラベル0
        label = 0
    elif dir == "shikaku_g":  # orangeはラベル1
        label = 1
    elif dir == "tri_g":  # orangeはラベル1
        label = 2
    elif dir == "tri_b":  # orangeはラベル1
        label = 3

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            label_list.append(label)  # 型宣言が無いとどうもやりづらいというか頭が混乱するなぁ
            filepath = dir1 + "/" + file
            image = np.array(Image.open(filepath).resize((25, 25)))
            print(filepath)
            # image = image.transpose(2, 0, 1)
            image = np.reshape(image, (1, 625))
            result = model.predict_classes(np.array([image / 255.]))
            print("label:", label, "result:", result[0])

            total += 1.

            if label == result[0]:
                ok_count += 1.

print("seikai: ", ok_count / total * 100, "%")
