import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation

data = np.random.rand(250, 5)
labels = np_utils.to_categorical((np.sum(data, axis=1) > 2.5) * 1)
model = Sequential([Dense(20, input_dim=5), Activation('relu'), Dense(2, activation='softmax')])
model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels, nb_epoch=300, validation_split=0.2)

test = np.random.rand(200, 5)
predict = np.argmax(model.predict(test), axis=1)
real = (np.sum(test, axis=1) > 2.5) * 1
print(sum(predict == real) / 200.0)

for i in data:
    if sum((predict == real)/200.) == True
        rue :
        print(data[i])
