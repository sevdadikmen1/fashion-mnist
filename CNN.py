import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv("C:/Users/sevda/OneDrive/Masaüstü/21-22/Yapay-Zeka/Final/input/fashion-mnist_train.csv")
test = pd.read_csv("C:/Users/sevda/OneDrive/Masaüstü/21-22/Yapay-Zeka/Final/input/fashion-mnist_test.csv")


Y_train = train["label"].values
X_train = train.drop(labels = ["label"], axis = 1)
X_train.head()

Y_test = test["label"].values
X_test = test.drop(labels = ["label"], axis = 1)
X_test.head()


#Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

#Reshaping
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)


from keras.utils.np_utils import to_categorical 

Y_train = to_categorical(Y_train, num_classes = 10)
Y_test = to_categorical(Y_test, num_classes = 10)


import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization


model = Sequential()

#first layer
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))

#second layer
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

#third layer
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

#fourth layer
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(MaxPool2D(pool_size=(2, 2)))

#fully connected layer
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))

#output layer
model.add(Dense(10, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])


epochs = 5
batch_size = 100


model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
    verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size)

model.evaluate(X_test, Y_test)

model.evaluate(X_train, Y_train)