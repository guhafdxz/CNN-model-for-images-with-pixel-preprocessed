# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 16:41:45 2022
****AlexNet CNN with Keras and cifar10 datasets***
Environment:Intel CPU
python=3.9.X
tensorflow=2.10.0
numpy= 1.23.4
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
# import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.utils import np_utils



# Tensorflow-GPU config
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  



'''
Image Pixel Processing
'''
def process_pixel(x_train,depth,size):
    if depth==32:
        return x_train
    xtrain=np.zeros((x_train.shape[1],x_train.shape[1]),dtype='u')                     
    for i in range(0,size):
        g_xtrain=cv2.cvtColor(x_train[i], cv2.COLOR_RGB2GRAY) # RGB to Grey
        if depth==8:
            xtrain=np.concatenate([xtrain,g_xtrain],axis=0) 
        if depth==4:
            xtrain=np.concatenate([xtrain,np.right_shift(g_xtrain,4)],axis=0)
        if depth==2:
            xtrain=np.concatenate([xtrain,np.right_shift(g_xtrain,6)],axis=0)
        if depth==1:
            _, b_xtrain= cv2.threshold(g_xtrain, 127, 255,cv2.THRESH_BINARY)  #Grey to Binary
            xtrain=np.concatenate([xtrain,b_xtrain],axis=0)  
    xtrain=xtrain[32:].reshape(size,32,32)
    return xtrain
    
'''
Fetching the datasets cifar10 using Keras
'''
def getCifar(size,depth):
 #  x_train and y_train represent the images and labels of the training set 
 #  x_test and y_test represent the images and labels of the test set
    (x_train,y_train), (x_test, y_test) = cifar10.load_data()
 #  x_val and y_val represent the images and labels of the validation set 
    x_train=x_train[:10000]   #total length of x_train is 50000   replace with 30000-50000 if possible
    y_train=y_train[:10000]   #total length of y_train is 50000   replace with 30000-50000 if possible
    x_val= x_test[:size]      #total length of x_test is 10000    
    y_val = y_test[:size]     #total length of y_test is 10000
    x_test = x_test[size:]    
    y_test = y_test[size:]
    x_train=process_pixel(x_train,depth,x_train.shape[0])
    x_val=process_pixel(x_val,depth,size) 
    x_test=process_pixel(x_test,depth,x_test.shape[0]) 
 #  Normalization and data-type processing to improve the training accuracy
    x_train = x_train.astype('float32') / 255
    x_val = x_val.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
 # one-hot coding of the lables of image
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    y_test = np_utils.to_categorical(y_test) 
    return x_train, y_train, x_val, y_val, x_test, y_test

'''
AlexNet Modle
'''
def alexnet(channel):
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(96, (3, 3), strides=(3, 3), input_shape=(32,32,channel), padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    # 2nd Convolutional Layer
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    # 3rd Convolutional Layer
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 4th Convolutional Layer
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 5th Convolutional Layer
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")) 
    
    #Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.3))
    # 2nd Fully Connected Layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.3))
    # 3rd Fully Connected Layer
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(10, activation='softmax'))
    print(model.summary())

    return model

batch_size=300  # num of traing sampling
num_classes=10 # num of category
epochs =20 # training times   epochs*batch_size=length of training set ,eg.10000
channel=3 # 3 for  depth=32 1 for others
depth=8 # pixel depth [1,2,4,8,32]
'''
Compile AlexNet and Train
'''
size=2000  # length of validation sets  replace with 6000-10000 if possible
x_train, y_train, x_val, y_val, x_test, y_test =getCifar(size,depth)
if channel==1:
    x_train=np.reshape(x_train,(x_train.shape[0],32,32,1))  #reshape the single channel image to 4D array to train
    x_val=np.reshape(x_val,(x_val.shape[0],32,32,1))
    x_test=np.reshape(x_test,(x_test.shape[0],32,32,1))
model = alexnet(channel)
# Compile network (loss function, optimizer, evaluation indicator)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Network training (training and validation data, training algebra,  training batch size)
train_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2)
# Sava model
model.save('alexnet_cifar10_grey.h5')


"mixed 8 depth image with 1 depth image"
# depth=1
# xtrain_2, ytrain_2, xval_2, yval_2, xtest_2, ytest_2 =getCifar(size,depth)
# if channel==1:
#     xtrain_2=np.reshape(x_train,(x_train.shape[0],32,32,1))  #reshape the single channel image to 4D array to train
#     xval_2=np.reshape(x_val,(x_val.shape[0],32,32,1))
#     xtest_2=np.reshape(x_test,(x_test.shape[0],32,32,1))

# xtrain_2= xtrain_2.astype('float32') / 255
# xval_2= xval_2.astype('float32') / 255
# xtest_2= x_test.astype('float32') / 255
# xtrain_2=np.reshape(xtrain_2,(xtrain_2.shape[0],32,32,1))
# xval_2=np.reshape(xval_2,(xval_2.shape[0],32,32,1))
# xtest_2=np.reshape(xtest_2,(xtest_2.shape[0],32,32,1))
# concat_xval=np.concatenate([x_val,xval_2],axis=0)
# concat_yval=np.concatenate([y_val,yval_2],axis=0)
# train_history_2 = model.fit(np.concatenate([x_train,xtrain_2],axis=0),np.concatenate([y_train,ytrain_2],axis=0),validation_data=(concat_xval, concat_yval), epochs=epochs, batch_size=batch_size, verbose=2)




'''
training process visualization function 
'''
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()
show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

#The model output of on the test set
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#  results of prediction
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)













