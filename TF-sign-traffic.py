#!/usr/bin/env python
# coding: utf-8

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Activation,
    Flatten,
    Conv2D,
    MaxPool2D,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

INIT_LR = 0.0003
EPOCHS = 20
Batch_size = 32

train_loc = "dataset/train/"
test_loc = "dataset/test/"
val_loc = "dataset/val/"

# resize images & Data Agumention
trdata = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
traindata = trdata.flow_from_directory(
    directory=train_loc, target_size=(224, 224), shuffle=True
)
tsdata = ImageDataGenerator(rescale=1.0 / 255)
testdata = tsdata.flow_from_directory(
    directory=test_loc, target_size=(224, 224), shuffle=False
)
vdata = ImageDataGenerator(rescale=1.0 / 255)
valdata = vdata.flow_from_directory(
    directory=val_loc, target_size=(224, 224), shuffle=True
)

traindata.class_indices


input_shape = (224, 224, 3)
img_input = Input(shape=input_shape, name="img_input")

# Build the model
x = Conv2D(32, (3, 3), padding="same", activation="relu", name="layer_1")(img_input)
x = Conv2D(64, (3, 3), padding="same", activation="relu", name="layer_2")(x)
x = MaxPool2D((2, 2), strides=(2, 2), name="layer_3")(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), padding="same", activation="relu", name="layer_4")(x)
x = MaxPool2D((2, 2), strides=(2, 2), name="layer_5")(x)
x = Dropout(0.25)(x)

x = Conv2D(128, (3, 3), padding="same", activation="relu", name="layer_6")(x)
x = MaxPool2D((2, 2), strides=(2, 2), name="layer_7")(x)
x = Dropout(0.25)(x)

x = Flatten(name="layer_8")(x)
x = Dense(64, name="layer_9")(x)
# x = Dropout(0.5)(x)   # جرب تحذفه
x = Dense(24, activation="softmax", name="predections")(x)

# generate the model
model = Model(inputs=img_input, outputs=x, name="CNN-traffic_sign")

# pint network structure
model.summary()


# compile our model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2, verbose=0, mode="auto")
]

# train the head of the network
print("[INFO] training head...")
hist = model.fit(
    traindata,
    batch_size=Batch_size,
    steps_per_epoch=traindata.samples // Batch_size,
    validation_steps=valdata.samples // Batch_size,
    callbacks=my_callbacks,
    epochs=EPOCHS,
    validation_data=valdata,
    verbose=True,
)


plt.plot(hist.history["loss"], label="train")
plt.plot(hist.history["val_loss"], label="val")
plt.title("traffic_sign : Loss & Valdation Loss")
plt.legend()
plt.show()

plt.plot(hist.history["accuracy"], label="train")
plt.plot(hist.history["val_accuracy"], label="val")
plt.title("traffic_sign : Accuracy & Valdation Loss")
plt.legend()
plt.show()

# Confusion Matrix & Precision & recall F1-score


classes = {
    "front_or_left": 0,
    "front_or_right": 1,
    "hump": 2,
    "left_turn": 3,
    "narro_from_lef": 4,
    "narrows_from_right": 5,
    "no_horron": 6,
    "no_parking": 7,
    "no_u_turn": 8,
    "overtaking_is_forbidden": 9,
    "parking": 10,
    "pedestrian_crossing": 11,
    "right_or_left": 12,
    "right_turn": 13,
    "rotor": 14,
    "slow": 15,
    "speed_100": 16,
    "speed_30": 17,
    "speed_40": 18,
    "speed_50": 19,
    "speed_60": 20,
    "speed_80": 21,
    "stop": 22,
    "u_turn": 23,
}

labels_names = list(classes.keys())
target_names = list(classes.values())
Y_pred = model.predict(testdata)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(testdata.classes, y_pred, labels=labels_names)

print("Confusion Matrix")
print(confusion_matrix(testdata.classes, y_pred))

print("Classification_report")
print(classification_report(testdata.classes, y_pred, target_names=target_names))
confusion_matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp = disp.plot(cmap=plt.cm.Blues, values_format="g")

plt.show()

total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

model.save("cnn-traffic_sign-full.h5", save_format="h5")
