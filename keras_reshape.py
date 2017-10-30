# import keras
import numpy as np

ar = np.arange(12)
vr = np.arange(32)
print(ar)

b = np.reshape(ar,(3, 4))

print(b)

c = np.reshape(b,(1,12))
print(c)