# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:16:09 2020
@author: khawla Seddiki
"""
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical, np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, BatchNormalization, MaxPooling1D, LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


nb_classes = nb_classes
ms_input_shape = ms_input_shape

# scaling data 
def ms_data(data_files,csv_name):
    mat_data = []
    labels = []
    mat_data = pd.read_csv(f"{data_files}{csv_name}")
    mat_data = np.asarray(mat_data)
    labels = mat_data[:, 0]
    mat_data = minmax_scale(mat_data[:,1:], axis=0, feature_range=(0, 1))
    mat_data = mat_data.astype("float32")
    labels = labels.astype("int")
    return mat_data, labels

## Model variant_Lecun (model 1 : 4 layers)
def build_model():
    model = Sequential([
        Conv1D(filters=6, kernel_size=21, strides=1, padding='same', activation='relu', input_shape= ms_input_shape,
               kernel_initializer=keras.initializers.he_normal()),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(filters=16, kernel_size=5, strides=1, padding='same',activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84),
        Dense(nb_classes, activation='sigmoid') # or Activation('softmax')
    ])
    return model

## Model variant_LeNet (model 2: 5 layers)
def build_model():
    model = Sequential([
        Conv1D(filters=16, kernel_size=21, strides=1, padding='same', input_shape= ms_input_shape,
               kernel_initializer=keras.initializers.he_normal()),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(filters=32, kernel_size=11, strides=1, padding='same'),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(filters=64, kernel_size=5, strides=1, padding='same'),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Flatten(),
        Dense(2050, activation='relu'),
        Dropout(0.5),
        Dense(nb_classes, activation='sigmoid') # or Activation('softmax')
    ])
    return model

## Model variant_VGG9 (model 3: 9 layers)
def build_model():
    model = Sequential([
        Conv1D(filters=64, kernel_size=21, strides=1, padding='same', activation='relu',
               input_shape= ms_input_shape, kernel_initializer=keras.initializers.he_normal()),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=21, strides=1, padding='same',activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(filters=128, kernel_size=11, strides=1, padding='same',activation='relu'),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=11, strides=1, padding='same',activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(filters=256, kernel_size=5, strides=1, padding='same',activation='relu'),
        BatchNormalization(),
        Conv1D(filters=256, kernel_size=5, strides=1, padding='same',activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(nb_classes, activation='sigmoid') # or Activation('softmax')
    ])
    return model

data, label = ms_data(data_files='../my_data/', csv_name="my_spectra.csv")

# 5 Fold-CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
e=1
test_loss = []
test_acc = []
for train, test in skf.split(data, label):
    x_train = data[train]
    y_train = label[train]
    x_test = data[test]
    y_test = label[test]
    x_train_conv = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
    x_test_conv = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))
    y_train_conv = to_categorical(y_train)
    y_test_conv = to_categorical(y_test)
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                      metrics=['accuracy']) # or categorical_crossentropy
    # model.summary()
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=0.0000001)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
    y_integers = np.argmax(y_train_conv, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))
    history = model.fit(x=x_train_conv, y=y_train_conv, batch_size=256, verbose=1, epochs=50, validation_split=0.2,
                            class_weight=d_class_weights, callbacks=[earlyStopping, reduce_lr])
    loss, acc = model.evaluate(x_test_conv, y_test_conv, verbose=1)
    y_classes = model.predict_classes(x_test_conv, verbose=1)
    matrix = confusion_matrix(y_test, y_classes)
    print(matrix)
    tn, fp, fn, tp = confusion_matrix(y_test, y_classes).ravel()
    specificity = tn / (tn + fp)
    print(specificity)
    sensitivity = tp / (tp + fn)
    print(sensitivity)
    test_loss.append(loss)
    test_acc.append(acc)
    filepath = "./my_model/model" + str(e) + "/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    model.save(filepath + "model" + str(e) + "-weights.h5")
    e = e + 1

model_source = load_model('/home/khawkha/PycharmProjects/DeepMS/my_model/model1/model1-weights.h5')
# Transfer learning with variant_leNet model
# layer.trainable = True allows to fine-tune with weights initialization
# layer.trainable = False allows to fine-tune without weights initialization
def target_model_from_source_model():
    model_source = load_model('../my_model/model../"model...-weights.h5')
    model_source.summary()
    # number of pop layers depend on the layers to train
    model_source.pop()
    model_source.pop()
    model_source.pop()
    model_source.pop()

    for layer in model_source.layers:
        layer.trainable = True

    target_model = Sequential([
        model_source,
        Flatten(),
        Dense(2050, activation='relu'),
        Dropout(0.5),
        Dense(nb_classes, activation='sigmoid')
    ])

model = target_model_from_source_model()
